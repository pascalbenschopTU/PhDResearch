import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import cv2
import os
from PIL import Image
import time
import argparse
from ultralytics import YOLO


# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_gaussian_kernel(kernel_size, sigma, device='cuda'):
    """Create a 2D Gaussian kernel."""
    # Make kernel size an odd number to have a center pixel
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create a 2D Gaussian kernel using meshgrid
    x = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2.
    gauss_kernel = torch.exp(-0.5 * (x**2) / (sigma**2))
    gauss_kernel = gauss_kernel / gauss_kernel.sum()

    # Create 2D Gaussian kernel by outer product
    gauss_kernel_2d = torch.outer(gauss_kernel, gauss_kernel)
    gauss_kernel_2d = gauss_kernel_2d / gauss_kernel_2d.sum()  # Normalize

    # Reshape to (1, 1, kernel_size, kernel_size) for convolution
    gauss_kernel_2d = gauss_kernel_2d.view(1, 1, kernel_size, kernel_size)
    return gauss_kernel_2d

def gaussian_blur_torch(img, kernel_size, sigma):
    """Apply Gaussian blur to a 4D tensor (B, C, H, W) using a Gaussian kernel."""
    device = img.device

    # Create Gaussian kernel and move to device
    kernel = create_gaussian_kernel(kernel_size, sigma, device=device)
    
    # Apply the Gaussian kernel to each channel separately
    channels = img.shape[1]
    kernel = kernel.repeat(channels, 1, 1, 1)  # Repeat for each channel

    # Pad the image to keep original size after convolution
    padding = kernel_size // 2
    img_blurred = torch.nn.functional.conv2d(img, kernel, padding=padding, groups=channels)
    
    return img_blurred

def adaptive_blur_patch(img, mask, bounding_box, G_base, alpha_b, alpha_r, is_max, is_full_blur=False):
    # Extract bounding box coordinates
    x1, y1, x2, y2 = map(int, bounding_box[:4])
    img_size = img.shape[-2] * img.shape[-1]

    # Move image and mask to GPU if not already
    img = img.to(device)
    mask = mask.to(device)

    # Extract the patch from the image and mask
    mask_patch = mask[y1:y2, x1:x2]
    img_patch = img[0, :, y1:y2, x1:x2]  # Select batch 0 since batch size is 1

    # Step 1: Calculate mask size and boundary size
    mask_size = torch.count_nonzero(mask_patch).item()
    Zb = torch.tensor([y2 - y1, x2 - x1], device=device)

    # Step 2: Scaling factor `r`
    r = max(alpha_r * np.log(100 * mask_size / img_size), 1)

    # Step 3: Define kernel parameters
    if is_max:
        Ka = torch.round(Zb).int().tolist()
        Ka = [k if k % 2 != 0 else k - 1 for k in Ka]
        sigma_a = 0.3 * (0.5 * (max(Ka) - 1) - 1) + 0.8
    else:
        K_base, sigma_base = G_base
        Ka = torch.round(r * K_base).int()
        Ka = [min(int(Ka[i].item()), max(int(alpha_b * Zb[i].item()), 1)) for i in range(2)]
        Ka = [k if k % 2 != 0 else k - 1 for k in Ka]
        sigma_a = r * sigma_base if is_full_blur else sigma_base

    blurred_patch = gaussian_blur_torch(img_patch.unsqueeze(0), kernel_size=max(Ka), sigma=sigma_a)
    blurred_patch = blurred_patch.squeeze(0)

    # Step 5: Combine blurred patch and original patch using the mask
    mask_3d_patch = mask_patch.unsqueeze(0).repeat(3, 1, 1)  # Expand mask to 3 channels for RGB
    img_patch = torch.where(mask_3d_patch == 1, blurred_patch, img_patch)

    # Place the patch back in the original image
    output_img = img.clone()
    output_img[0, :, y1:y2, x1:x2] = img_patch  # Put patch back into original image

    return output_img


def apply_adaptive_blur(img, model, G_base, alpha_b, alpha_r, is_max, is_full_blur, size=(640, 640)):
    # Convert image to tensor and resize only once to the model input size
    start_time = time.time()
    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    img_tensor_resized = F.interpolate(img_tensor, size=size, mode="bilinear", align_corners=False)

    print(f"Preprocessing in ms: {(time.time() - start_time) * 1000:.2f}")

    # Perform segmentation
    model.to(device)
    results = model(img_tensor_resized)

    print(f"Segmentation in ms: {(time.time() - start_time) * 1000:.2f}")

    # Get segmentation masks and other details
    masks, boxes, classes = get_segmentation_masks(results, size)

    print(f"Extracted masks in ms: {(time.time() - start_time) * 1000:.2f}")
    
    # Process the image directly on the GPU
    final_img = img_tensor_resized.clone()
    
    for mask, bounding_box, cls in zip(masks, boxes, classes):
        # If the class isn't 'person', skip this iteration
        if model.names[int(cls)] != 'person':
            continue

        # Apply adaptive blur only on the relevant areas
        final_img = adaptive_blur_patch(final_img, mask, bounding_box, G_base, alpha_b, alpha_r, is_max, is_full_blur)
    
    print(f"Applied adaptive blur in ms: {(time.time() - start_time) * 1000:.2f}")

    # Resize only once to the original size
    final_img = F.interpolate(final_img, size=img_tensor.shape[2:], mode="bilinear", align_corners=False)

    print(f"Postprocessing in ms: {(time.time() - start_time) * 1000:.2f}")
    
    # Convert to numpy array only once
    final_img_np = (final_img.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    return final_img_np


def get_segmentation_masks(results, img_size):
    result_masks = []
    result_boxes = []
    result_classes = []
    
    for item in results:
        # Get the masks, boxes, and classes for each detected object in the batch
        masks = item.masks
        if masks is not None:
            # Convert masks to tensor and resize in one operation
            data = masks.data.to(device).unsqueeze(1)  # Adding a channel dimension for interpolation

            # Resize all masks in a single batch operation
            masks_resized = F.interpolate(data.float(), size=img_size, mode="bilinear", align_corners=False)
            masks_resized = masks_resized.squeeze(1)  # Remove the added channel dimension
            
            result_masks.extend(masks_resized)  # Append all resized masks at once

        # Process bounding boxes and classes similarly
        boxes = item.boxes
        if boxes is not None:
            data = boxes.data.to(device)
            class_index = boxes.cls.to(device)
            
            for batch_index in range(data.shape[0]):
                result_boxes.append(data[batch_index].cpu().numpy())
                result_classes.append(class_index[batch_index].item())

    return result_masks, result_boxes, result_classes

def simplified_adaptive_blur(img, model):
    # Example parameters
    G_base = (5, 1.0)  # Example base Gaussian kernel size and sigma
    alpha_b = 0.5  # Example alpha value
    Z_ref = 640 * 640  # Reference image size
    alpha_r = (img.size[0] * img.size[1]) / Z_ref  # Example alpha_r value
    is_max = True  # Use maximum adaptive blur
    is_full_blur = True  # Set to True to fully scale sigma with r

    return apply_adaptive_blur(img, model, G_base, alpha_b, alpha_r, is_max, is_full_blur)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Anonymize images using adaptive blur.")
    parser.add_argument('--dataset_path', type=str, default=r"Action-Recognition/data/UCF101-frames", help='Path to the dataset folder')
    args = parser.parse_args()

    dataset_path = args.dataset_path


    model = YOLO('yolov8m-seg.pt')

    # Walk (recursively) through all folders and files in the dataset path
    # Anonymize all images using the function apply_adaptive_blur
    # Save the anonymized images in a new folder called "anonymized"
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".jpg"):
                # Load the image
                img_path = os.path.join(root, file)
                save_path = img_path.replace("UCF101-frames", "UCF101-frames-anonymized")

                if os.path.exists(save_path):
                    continue

                img = Image.open(img_path)

                start_time = time.time()

                # Apply adaptive blur
                final_img = simplified_adaptive_blur(img, model) # numpy array

                print(f"Anonymized in ms: {(time.time() - start_time) * 1000:.2f}")

                # Save the anonymized image
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                Image.fromarray(final_img).save(save_path)