import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.io import read_image
from extractor import ViTExtractor

def parse():
    parser = argparse.ArgumentParser(
        description="Save multiple descriptors by clicking on the image"
    )
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        help="The template image from which to extract descriptors",
    )
    parser.add_argument(
        "-n",
        "--descriptorname",
        type=str,
        help="Base name for descriptors, e.g., 'face_descriptor'. Each will be saved with a unique suffix",
    )
    parser.add_argument(
        "--model_type",
        default="dino_vits8",
        type=str,
        help="""type of model to extract. 
                Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
                vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]""",
    )
    parser.add_argument(
        "--facet",
        default="key",
        type=str,
        help="""facet to create descriptors from. 
                options: ['key' | 'query' | 'value' | 'token']""",
    )
    parser.add_argument(
        "--layer", default=11, type=int, help="layer to create descriptors from."
    )

    args = parser.parse_args()
    return args

def get_descriptor(
    image_path_a: str,
    load_size: int = 224,
    layer: int = 11,
    facet: str = "key",
    bin: bool = False,
    stride: int = 4,
    model_type: str = "dino_vits8",
    descriptorname="descriptor",
    prefix_savepath="output/descriptors/",
):
    # Set up device and extractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = ViTExtractor(model_type, stride, device=device)
    patch_size = extractor.model.patch_embed.patch_size
    image_batch_a, image_pil_a = extractor.preprocess(image_path_a, load_size)
    descr = extractor.extract_descriptors(
        image_batch_a.to(device), layer, facet, bin, include_cls=False
    )
    num_patches_a, load_size_a = extractor.num_patches, extractor.load_size

    # Prepare output directory
    os.makedirs(prefix_savepath, exist_ok=True)

    # Set up figure for interactive selection
    fig, ax = plt.subplots(figsize=(10, 6))
    interactive_title = "Click to save descriptor (right click to exit)"
    fig.suptitle(interactive_title)
    ax.imshow(image_pil_a)
    visible_patches = []
    radius = patch_size // 2
    descriptor_count = 1

    # Define the path for the reference image
    reference_image_path = os.path.join(prefix_savepath, f"{descriptorname}_reference.png")

    # Start the loop for interactive selection
    while True:
        # Wait for a click and get coordinates
        pts = np.asarray(
            plt.ginput(1, timeout=-1, mouse_stop=plt.MouseButton.RIGHT, mouse_pop=None)
        )
        if len(pts) != 1:  # Exit loop if right-click
            break

        # Process coordinates
        y_coor, x_coor = int(pts[0, 1]), int(pts[0, 0])

        # Calculate the patch indices directly from the coordinates
        patch_y = y_coor // stride
        patch_x = x_coor // stride

        # Calculate raveled descriptor index directly
        raveled_desc_idx = patch_y * num_patches_a[1] + patch_x

        # Ensure that patch_x and patch_y are within bounds
        if patch_x < 0 or patch_x >= num_patches_a[1] or patch_y < 0 or patch_y >= num_patches_a[0]:
            print(f"Clicked outside valid range. Clicked patch: ({patch_x}, {patch_y})")
            continue

        print(f"Selected patch at: ({patch_x}, {patch_y}), idx: {raveled_desc_idx}, descriptor shape: {descr.shape}")
        point_descriptor = descr[0, 0, raveled_desc_idx]

        # Calculate the center for drawing the patch circle
        center_x = patch_x * stride + patch_size // 2
        center_y = patch_y * stride + patch_size // 2
        patch = plt.Circle((center_x, center_y), radius, color=(1, 0, 0, 0.75))
        ax.add_patch(patch)
        fig.canvas.draw()

        # Save the descriptor with a unique name
        descriptor_filename = f"{descriptorname}_{descriptor_count}.pt"
        torch.save(point_descriptor, os.path.join(prefix_savepath, descriptor_filename))
        print(f"Saved descriptor to: {os.path.join(prefix_savepath, descriptor_filename)}")
        descriptor_count += 1

        # Change the title for the saved image
        fig.suptitle(f"Descriptors saved as '{descriptorname}' from selected points")
        # Save the updated figure as a reference image after each point is added
        fig.savefig(reference_image_path)
        print(f"Updated reference image saved to: {reference_image_path}")
        # Revert the title back to the interactive instruction
        fig.suptitle(interactive_title)




if __name__ == "__main__":
    args = parse()
    load_size = 224
    bin = False
    stride = 4
    with torch.no_grad():
        get_descriptor(
            args.image,
            load_size,
            args.layer,
            args.facet,
            bin,
            stride,
            args.model_type,
            args.descriptorname,
            prefix_savepath="output/descriptors/",
        )
