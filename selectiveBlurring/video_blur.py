import os
import torch
import cv2
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from tqdm import tqdm
from torchvision.io import read_image
from PIL import Image
import argparse
from helpers import similarity_from_descriptors, flow_from_video, firstframe_warp
from extractor import ViTExtractor
from RAFT.core.raft import RAFT

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

    flow_model = torch.nn.DataParallel(RAFT(args))
    flow_model.load_state_dict(torch.load("RAFT/models/raft-sintel.pth", map_location=torch.device('cpu')))
    flow_model = flow_model.module
    flow_model.to(device)
    flow_model.eval()

    # Generate an instance of the GaussianBlur class
    blur_kernel_size = 49  # Kernel size for blurring
    blur_sigma = 6.0       # Sigma value for blurring
    gauss_blur = GaussianBlur(blur_kernel_size, sigma=blur_sigma)

    # Load descriptors
    descriptor_files = [f for f in os.listdir(descriptors_folder) if f.endswith(".pt")]
    descriptors_batch = [
        torch.load(os.path.join(descriptors_folder, f), map_location=device)
        for f in descriptor_files if any(name in f for name in selected_descriptors)
    ]
    descriptors_batch = torch.stack(descriptors_batch, dim=0).to(device)  # Shape: (num_descriptors, descriptor_dim)

    # Process each frame
    processed_frames = []
    batch_size = 8
    with torch.no_grad():
        for i in tqdm(range(0, len(video_frames), batch_size), desc="Processing frames"):
            batch_frames = video_frames[i:i + batch_size]

            print(len(batch_frames))
            if len(batch_frames) < batch_size:
                # Pad the batch to match the batch size
                batch_frames += [batch_frames[-1]] * (batch_size - len(batch_frames))

            
            
            # Load and prepare the frames
            orig_images = [read_image(frame_path).to(device) for frame_path in batch_frames]
            orig_images_float = [img.float() for img in orig_images]
            
            # Compute similarity map using the batched descriptors
            similarity_maps = similarity_from_descriptors(
                descriptors_batch, batch_frames, extractor, device=device
            )

            similarity_maps = F.interpolate(
                similarity_maps,
                size=(orig_images_float[0].shape[1], orig_images_float[0].shape[2]),
                mode="bilinear",
            ).squeeze(1)

            # Create temporally consistent noise
            flow = flow_from_video(flow_model, orig_images_float, upsample_factor=args.upsample)
            afd = firstframe_warp(
                flow,
                usecolor=True,
            )
            print(len(flow))

            for j, frame_path in enumerate(batch_frames):
                batched_visual_features = (similarity_maps[j] * 255.0).clamp(80, 255).squeeze()

                # Generate saliency map
                saliency_map = (batched_visual_features - batched_visual_features.min()) / \
                               (batched_visual_features.max() - batched_visual_features.min())
                
                # Generate a blurred version of the image
                # blurred_image = gauss_blur(orig_images_float[j])
                blurred_image = afd[j].permute(2, 0, 1).to(device)

                # Adjust saliency map for blending
                saliency_map_adjusted = torch.where(saliency_map > 0.1, 0.75 + saliency_map, torch.zeros_like(saliency_map))
                saliency_map_adjusted = torch.clamp(saliency_map_adjusted, 0.0, 1.0)
                saliency_map_adjusted = saliency_map_adjusted.to(device)
                saliency_map_adjusted = saliency_map_adjusted.unsqueeze(0)

                # Blend original and blurred images using adjusted saliency map
                noisy_image = orig_images_float[j] * (1 - saliency_map_adjusted) + blurred_image * saliency_map_adjusted

                # Convert image back to byte format
                noisy_image = noisy_image.clamp(0, 255).byte()
                processed_frames.append(noisy_image.permute(1, 2, 0).cpu().byte())
    
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
        duration=100,  # Duration between frames in milliseconds
        loop=0
    )
    print(f"Processed frames saved to: {processed_gif_path}")

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
    
    # Set up output video writer
    output_video_path = os.path.join(output_dir, f"anonymized_{os.path.basename(video_path)}")
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Process video in batches of frames
    video_frames = []
    batch_size = 32  # You can adjust batch size based on available GPU memory
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

        # Process in batches
        if len(video_frames) == batch_size or frame_idx == total_frames:
            processed_frames = process_video_frames(
                video_frames, 
                args.descriptors_folder, 
                args.selected_descriptors, 
                model_type=args.model_type, 
                device=args.device
            )

            # Write processed frames to output video
            for processed_frame in processed_frames:
                # Convert processed frame from tensor to numpy array
                frame_np = processed_frame.numpy()
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

            # Clear processed video frames from memory
            for temp_path in video_frames:
                os.remove(temp_path)  # Remove temporary frame file
            video_frames = []  # Reset list for next batch

    # Release video reader and writer
    cap.release()
    out.release()
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
    parser.add_argument("-v", "--video_path", type=str, required=False, help="Path to video file for processing.")
    parser.add_argument("-df", "--descriptors_folder", type=str, required=True, help="Path to folder containing descriptor .pt files.")
    parser.add_argument("-ds", "--selected_descriptors", type=str, nargs='+', required=False, help="List of descriptor names to match and process.")
    parser.add_argument("-m", "--model_type", type=str, default="dino_vits8", help="Model type for descriptor extraction.")
    parser.add_argument("-dev", "--device", type=str, default="cuda", help="Device to perform computations on (e.g., 'cuda', 'mps', or 'cpu').")
    parser.add_argument("-o", "--output_dir", type=str, default="output", help="Output directory for processed frames.")
    
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
        