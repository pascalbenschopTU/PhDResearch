import os
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from tqdm import tqdm
from torchvision.io import read_image
from PIL import Image
import argparse
from helpers import similarity_from_descriptors, flow_from_video, firstframe_warp
from extractor import ViTExtractor
from RAFT.core.raft import RAFT


def process_frame_batch(orig_images_float, descriptors_batch, extractor, flow_model=None, device="cuda", seed_image=None):
    """
    Process a batch of video frames by applying saliency-based adaptive blurring.

    :param orig_images_float: Tensor of frames in float format.
    :param descriptors_batch: Tensor of loaded descriptors.
    :param extractor: Feature extractor model.
    :param flow_model: Pre-trained RAFT flow model.
    :param device: Device to perform computations on.

    :return: Processed frames as a tensor.
    """
    original_width = orig_images_float.shape[3]
    original_height = orig_images_float.shape[2]

    # Resize frames to 224x224 for descriptor matching
    images_224 = F.interpolate(orig_images_float, size=(224, 224), mode="bilinear")

    # Compute similarity map using loaded descriptors
    similarity_maps = similarity_from_descriptors(
        descriptors_batch, images_224, extractor, device=device
    )

    # Upscale similarity maps to original frame size
    similarity_maps = F.interpolate(
        similarity_maps, size=(original_height, original_width), mode="bilinear"
    ).squeeze(1)

    # Flow can be used in video to create motion consistent noise
    if flow_model:
        # Generate optical flow for temporal consistency
        flow = flow_from_video(flow_model, images_224, upsample_factor=1.0, device=device, iters=12)
        afd = firstframe_warp(flow, usecolor=True, seed_image=seed_image, device=device)

        # Blur the frames using afd
        blurred_images = afd.permute(0, 3, 1, 2).to(device)
        blurred_images = F.interpolate(blurred_images, size=(original_height, original_width))
    else:
        blurred_images = GaussianBlur(kernel_size=(69, 69), sigma=(10, 10))(orig_images_float)

    # Scale and normalize similarity maps
    batched_visual_features = (similarity_maps * 255.0).clamp(80, 255).unsqueeze(1)
    batched_visual_features = F.interpolate(batched_visual_features, size=(original_height, original_width))
    min_vals = torch.amin(batched_visual_features, dim=(1, 2), keepdim=True)
    max_vals = torch.amax(batched_visual_features, dim=(1, 2), keepdim=True)

    # The saliency maps are used as soft mask (can be converted to hard mask by thresholding)
    saliency_maps = (batched_visual_features - min_vals) / (max_vals - min_vals + 1e-8)

    # Blend original and blurred images
    noisy_images = orig_images_float * (1 - saliency_maps) + blurred_images * saliency_maps
    noisy_images = noisy_images.clamp(0, 255).permute(0, 2, 3, 1).cpu().byte()

    # Seed image is used to warp the consecutive frames in a batch
    if seed_image is not None:
        new_seed_image = afd[-1].permute(2, 0, 1).unsqueeze(0)
        # Normalize
        new_seed_image = (new_seed_image - new_seed_image.min()) / (new_seed_image.max() - new_seed_image.min())
        return noisy_images, new_seed_image
    else:
        return noisy_images, None


def process_video_frames(
        video_frames, 
        descriptors_folder, 
        selected_descriptors, 
        model_type="dino_vits8", 
        stride=4, 
        device="cuda",
        use_flow=False
    ):
    """
    Processes each frame of a video, applying saliency-based adaptive blurring.

    :param video_frames: List of file paths for video frames.
    :param descriptors_folder: Path to folder containing descriptor .pt files.
    :param selected_descriptors: List of descriptor names to match and process.
    :param model_type: Model type for descriptor extraction.
    :param stride: Stride for sampling frames.
    :param device: Device to perform computations on (e.g., "cuda", "mps", or "cpu").

    :return: List of processed frames with privacy-preserving blur applied.
    """
    device = torch.device(device)
    extractor = ViTExtractor(model_type, stride=stride, device=device)

    print(f"Number of parameters in {model_type} model: {sum(p.numel() for p in extractor.model.parameters())}")

    if use_flow:
        # Load pre-trained RAFT model
        flow_model = torch.nn.DataParallel(RAFT(args))
        print(f"Number of parameters in RAFT model: {sum(p.numel() for p in flow_model.parameters())}")
        flow_model.load_state_dict(torch.load("RAFT/models/raft-sintel.pth", map_location=device))
        flow_model = flow_model.module
        flow_model.to(device)
        flow_model.eval()
        seed_image = torch.randn(1, 3, 224, 224).to(device)
    else:
        flow_model = None
        seed_image = None

    # Load descriptors
    descriptor_files = [f for f in os.listdir(descriptors_folder) if f.endswith(".pt")]
    descriptors_batch = [
        torch.load(os.path.join(descriptors_folder, f), map_location=device)
        for f in descriptor_files if any(name in f for name in selected_descriptors)
    ]
    descriptors_batch = torch.stack(descriptors_batch, dim=0).to(device)

    processed_frames = []
    batch_size = 8

    with torch.no_grad():
        for i in tqdm(range(0, len(video_frames), batch_size), desc="Processing frames"):
            batch_frames = video_frames[i:i + batch_size]
            if len(batch_frames) < batch_size:
                batch_frames += [batch_frames[-1]] * (batch_size - len(batch_frames))

            # Convert file paths to tensor
            orig_images = [read_image(frame_path).to(device) for frame_path in batch_frames]
            orig_images_float = torch.stack([img.float() for img in orig_images], dim=0)

            # Process the current batch of frames
            processed_frames_batch, seed_image = process_frame_batch(
                orig_images_float, descriptors_batch, extractor, flow_model, device, seed_image=seed_image
            )
            processed_frames.extend(processed_frames_batch)

    return processed_frames

def process_frames(args):
    """
    Processes video frames with privacy-preserving blur.

    :param args: Command-line arguments.
    """

    # Get list of video frames from the specified directory
    video_frames = [os.path.join(args.frames_dir, f) for f in os.listdir(args.frames_dir) if f.endswith(('.jpg', '.png'))]
    video_frames.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.splitext(os.path.basename(x))[0]))))  # Sort frames numerically

    # video_frames = video_frames[:10]  # Process only the first 10 frames for demonstration

    if args.selected_descriptors is not None:
        # Process video frames
        processed_frames = process_video_frames(
            video_frames, 
            args.descriptors_folder, 
            args.selected_descriptors, 
            model_type=args.model_type, 
            device=args.device,
            use_flow=args.use_flow,
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

    # Save as Video with compression (e.g., using H.264 codec)
    processed_video_path = os.path.join(args.output_dir, "processed_video.mp4")

    # Get the height and width of the frames
    h, w = processed_frames[0].shape[0], processed_frames[0].shape[1]

    # Define the video codec (using H.264 compression with .mp4 format)
    video_fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4
    video_writer = cv2.VideoWriter(processed_video_path, fourcc, video_fps, (w, h))

    # Write each frame to the video
    for frame in processed_frames:
        # Ensure the frame is in uint8 format and convert to BGR
        frame_np = frame.byte().cpu().numpy()  # Convert to NumPy array
        if frame_np.shape[-1] == 3:  # Check if RGB
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        if frame_np.shape[:2] != (h, w):
            frame_np = cv2.resize(frame_np, (w, h), interpolation=cv2.INTER_LINEAR)
            print(f"Resized frame to ({w}, {h})")
        video_writer.write(frame_np)

        # Save an image for debug
        cv2.imwrite("frame.png", frame_np)

    video_writer.release()
    print(f"Processed frames saved as video to: {processed_video_path}")

def process_video(args):
    """
    Processes a video file with privacy-preserving blur and saves the anonymized video.

    :param args: Command-line arguments.
    """
    # Define paths for input and output videos
    video_path = args.video_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Capture video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Gather all frames
    video_frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Convert frame from BGR (OpenCV) to RGB (PyTorch format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # Save frame as temporary file to feed into process_video_frames
        temp_frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
        frame_pil.save(temp_frame_path)
        video_frames.append(temp_frame_path)
        frame_idx += 1

    cap.release()  # Release the video capture as we no longer need it

    # Process all frames
    processed_frames = process_video_frames(
        video_frames, 
        args.descriptors_folder, 
        args.selected_descriptors, 
        model_type=args.model_type, 
        device=args.device,
        use_flow=args.use_flow,
    )

    # Set up output video writer, and create .mp4 video file
    output_video_path = os.path.join(output_dir, f"anonymized_{os.path.splitext(os.path.basename(video_path))[0]}.mp4")
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Write processed frames to output video
    for processed_frame in processed_frames:
        # Convert processed frame from tensor to numpy array
        frame_np = processed_frame.numpy()
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()  # Release video writer

    # Clean up temporary files
    for temp_path in video_frames:
        os.remove(temp_path)  # Remove temporary frame file

    print(f"Anonymized video saved to: {output_video_path}")

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
        device=args.device,
        use_flow=args.use_flow,
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
    parser.add_argument("-v", "--video_path", type=str, required=False, help="Path to video file for processing.")
    parser.add_argument("-df", "--descriptors_folder", type=str, required=True, help="Path to folder containing descriptor .pt files.")
    parser.add_argument("-ds", "--selected_descriptors", type=str, nargs='+', required=False, help="List of descriptor names to match and process.")
    parser.add_argument("-m", "--model_type", type=str, default="dino_vits8", help="Model type for descriptor extraction.")
    parser.add_argument("-dev", "--device", type=str, default="cuda", help="Device to perform computations on (e.g., 'cuda', 'mps', or 'cpu').")
    parser.add_argument("-o", "--output_dir", type=str, default="output", help="Output directory for processed frames.")
    parser.add_argument("-flow", "--use_flow", action="store_true", help="Use optical flow for temporal consistency.")
    
    args = parser.parse_args()

    args.alternate_corr = False
    args.mixed_precision = True
    args.small = False
    args.upsample = 1

    if args.image_path is not None:
        # Process a single image
        process_image(args)
    elif args.frames_dir is not None:
        # Process video frames
        process_frames(args)
    elif args.video_path is not None:
        # Process video file
        process_video(args)
    else:
        print("Please provide either an image path or a directory containing video frames.")
        