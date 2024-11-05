import os
import torch
import torchvision.transforms as transforms
from torchvision.transforms import GaussianBlur
from tqdm import tqdm
from torchvision.io import read_image
from PIL import Image
import argparse
from helpers import similarity_from_descriptors  # Import modified function
from extractor import ViTExtractor
import time

import matplotlib.pyplot as plt

def process_video_frames(video_frames, descriptors_folder, selected_descriptors, model_type="dino_vits8", stride=4, device="cuda"):
    """
    Processes each frame of a video, applying saliency-based adaptive blurring.
    
    :param video_frames: List of file paths for video frames.
    :param descriptors_folder: Path to folder containing descriptor .pt files.
    :param selected_descriptors: List of descriptor names to match and process.
    :param model_type: Model type for descriptor extraction.
    :param device: Device to perform computations on (e.g., "cuda", "mps", or "cpu").
    
    :return: List of processed frames with privacy-preserving blur applied.
    """
    device = torch.device(device)
    extractor = ViTExtractor(model_type, stride=stride, device=device)

    # Load descriptors
    descriptor_files = [f for f in os.listdir(descriptors_folder) if f.endswith(".pt")]
    descriptors_batch = [
        torch.load(os.path.join(descriptors_folder, f), map_location=device)
        for f in descriptor_files if any(name in f for name in selected_descriptors)
    ]
    descriptors_batch = torch.stack(descriptors_batch, dim=0).to(device)  # Shape: (num_descriptors, descriptor_dim)

    # Process each frame
    processed_frames = []
    with torch.no_grad():
        for frame_path in tqdm(video_frames, desc="Processing frames"):
            # Load and prepare the frame
            orig_image = read_image(frame_path).to(device)
            orig_image_float = orig_image.float()
            
            # Compute similarity map using the batched descriptors
            similarity_maps = similarity_from_descriptors(
                descriptors_batch, [frame_path], extractor, device=device
            )
            batched_visual_features = (similarity_maps[0] * 255.0).clamp(80, 255).squeeze()

            # Generate saliency map
            saliency_map = (batched_visual_features - batched_visual_features.min()) / \
                           (batched_visual_features.max() - batched_visual_features.min())
            
            # Generate a blurred version of the image
            blur_kernel_size = 49  # Kernel size for blurring
            blur_sigma = 6.0       # Sigma value for blurring
            blurred_image = GaussianBlur(blur_kernel_size, sigma=blur_sigma)(orig_image_float)

            # Adjust saliency map for blending
            saliency_map_adjusted = torch.where(saliency_map > 0.1, torch.ones_like(saliency_map), torch.zeros_like(saliency_map))
            saliency_map_adjusted = saliency_map_adjusted.to(device)

            # Blend original and blurred images using adjusted saliency map
            noisy_image = orig_image_float * (1 - saliency_map_adjusted.unsqueeze(0)) + blurred_image * saliency_map_adjusted.unsqueeze(0)

            # Convert image back to byte format
            noisy_image = noisy_image.clamp(0, 255).byte()
            processed_frames.append(noisy_image.permute(1, 2, 0).cpu().byte())
    
    return processed_frames

def process_video(args):
    """
    Processes video frames with privacy-preserving blur.

    :param args: Command-line arguments.
    """

    # Get list of video frames from the specified directory
    video_frames = [os.path.join(args.frames_dir, f) for f in os.listdir(args.frames_dir) if f.endswith(('.jpg', '.png'))]
    video_frames.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))  # Sort frames numerically

    # video_frames = video_frames[:10]  # Process only the first 10 frames for demonstration

    if args.selected_descriptors is not None:
        # Process video frames
        processed_frames = process_video_frames(
            video_frames, 
            args.descriptors_folder, 
            args.selected_descriptors, 
            model_type=args.model_type, 
            device=args.device
        )
        print(f"Processed {len(processed_frames)} frames.")
    else:
        processed_frames = [read_image(f).permute(1, 2, 0).cpu() for f in video_frames]

    # Save files to directory
    frames_dir = os.path.join(args.output_dir, "processed_frames")
    os.makedirs(frames_dir, exist_ok=True)
    for i, frame in enumerate(processed_frames):
        frame_path = os.path.join(frames_dir, f"{i:04d}.png")
        Image.fromarray(frame.numpy()).save(frame_path)


    # Convert each processed frame (tensor) to a PIL Image
    processed_frames_pil = [Image.fromarray(frame.numpy()) for frame in processed_frames]

    # Save as GIF
    processed_gif_path = os.path.join(args.output_dir, "processed_frames.gif")
    processed_frames_pil[0].save(
        processed_gif_path,
        save_all=True,
        append_images=processed_frames_pil[1:],
        duration=200,  # Duration between frames in milliseconds
        loop=0
    )
    print(f"Processed frames saved to: {processed_gif_path}")

def process_image(args):
    """
    Processes a single image with privacy-preserving blur.

    :param args: Command-line arguments.
    """
    # Load and prepare the image
    image_path = args.image_path
    orig_image = read_image(image_path)
    orig_image_float = orig_image.float()

    # Load descriptors
    descriptor_files = [f for f in os.listdir(args.descriptors_folder) if f.endswith(".pt")]
    descriptors_batch = [
        torch.load(os.path.join(args.descriptors_folder, f), map_location=args.device)
        for f in descriptor_files if any(name in f for name in args.selected_descriptors)
    ]
    descriptors_batch = torch.stack(descriptors_batch, dim=0).to(args.device)  # Shape: (num_descriptors, descriptor_dim)

    # Process the image
    processed_frame = process_video_frames(
        [image_path], 
        args.descriptors_folder, 
        args.selected_descriptors, 
        model_type=args.model_type, 
        device=args.device
    )[0]

    # Save the processed image
    processed_image_path = os.path.join(args.output_dir, "processed_image.png")
    Image.fromarray(processed_frame.numpy()).save(processed_image_path)
    print(f"Processed image saved to: {processed_image_path}")



if __name__ == "__main__":
    # Define paths and parameters
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process video frames with privacy-preserving blur.")
    parser.add_argument("-f", "--frames_dir", type=str, required=False, help="Path to directory containing video frames.")
    parser.add_argument("-i", "--image_path", type=str, required=False, help="Path to image for descriptor extraction.")
    parser.add_argument("-df", "--descriptors_folder", type=str, required=True, help="Path to folder containing descriptor .pt files.")
    parser.add_argument("-ds", "--selected_descriptors", type=str, nargs='+', required=False, help="List of descriptor names to match and process.")
    parser.add_argument("-m", "--model_type", type=str, default="dino_vits8", help="Model type for descriptor extraction.")
    parser.add_argument("-dev", "--device", type=str, default="cuda", help="Device to perform computations on (e.g., 'cuda', 'mps', or 'cpu').")
    parser.add_argument("-o", "--output_dir", type=str, default="output", help="Output directory for processed frames.")
    
    args = parser.parse_args()

    if args.image_path is not None:
        # Process a single image
        process_image(args)
    elif args.frames_dir is not None:
        # Process video frames
        process_video(args)
    else:
        print("Please provide either an image path or a directory containing video frames.")
        