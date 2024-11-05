import os

import numpy as np
import PIL
import torch
from torchvision import transforms
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image

from extractor import ViTExtractor


def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y)
    """
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Cannot convert to boolean")


# get the best matching descriptor in the first image of the video to the descriptor
def get_best_matching_descriptor(descriptor, image_path):
    with torch.no_grad():
        # extract descriptors
        device = "cuda" if torch.cuda.is_available() else "cpu"
        extractor = ViTExtractor(device=device)
        image_batch, _ = extractor.preprocess(image_path, load_size=224)

        descs = extractor.extract_descriptors(
            image_batch.to(device), layer=11, facet="key", bin=False, include_cls=False
        )

        # compute similarity
        sim = chunk_cosine_sim(descriptor[None, None, None], descs)
        sim_image = sim.reshape(extractor.num_patches)
        sim_image = sim_image.cpu().numpy()

        # get best matching descriptor
        best_matching_descriptor = np.argmax(sim_image)
        return descs[:, :, best_matching_descriptor].squeeze()


def save_similarity_from_descriptor(
    descriptor,
    videoname: str,
    images: str,
    load_size: int = 224,
    layer: int = 11,
    facet: str = "key",
    bin: bool = False,
    stride: int = 4,
    model_type: str = "dino_vits8",
    prefix_savedir="output/similarities/",
    name=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = ViTExtractor(model_type, stride, device=device)
    patch_size = extractor.model.patch_embed.patch_size
    img_size = PIL.Image.open(images[0]).size[::-1]

    similarities = []

    for en, image_path_b in enumerate(images):
        print(f"Computing Descriptors {en}")
        image_batch_b, image_pil_b = extractor.preprocess(image_path_b, load_size)
        descs_b = extractor.extract_descriptors(
            image_batch_b.to(device), layer, facet, bin, include_cls=False
        )
        num_patches_b, load_size_b = extractor.num_patches, extractor.load_size
        sim = chunk_cosine_sim(descriptor[None, None, None], descs_b)
        similarities.append(sim)

        sim_image = sim.reshape(num_patches_b)
        os.makedirs(prefix_savedir + f"/{name}_{videoname}", exist_ok=True)
        sim_image = transforms.Resize(img_size, antialias=True)(sim_image.unsqueeze(0))
        save_image(sim_image, f"{prefix_savedir}/{name}_{videoname}/{en:04d}.png")


def similarity_from_descriptor(
    descriptor,
    images: str,
    load_size: int = 224,
    layer: int = 11,
    facet: str = "key",
    bin: bool = False,
    stride: int = 4,
    model_type: str = "dino_vits8",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = ViTExtractor(model_type, stride, device=device)
    similarities = []
    img_size = PIL.Image.open(images[0]).size[::-1]

    ret = []
    for en, image_path_b in tqdm(enumerate(images), total=len(images), position=1):
        # print(f"Computing Descriptors {en}")
        image_batch_b, image_pil_b = extractor.preprocess(image_path_b, load_size)
        descs_b = extractor.extract_descriptors(
            image_batch_b.to(device), layer, facet, bin, include_cls=False
        )
        num_patches_b, load_size_b = extractor.num_patches, extractor.load_size
        print(f"descriptor shape: {descriptor.shape}, descs_b shape: {descs_b.shape}")
        sim = chunk_cosine_sim(descriptor[None, None, None], descs_b)
        similarities.append(sim)

        sim_image = sim.reshape(num_patches_b)
        print(f"sim_image shape: {sim_image.shape}, img shape: {img_size}")
        ret_img = transforms.Resize(img_size, antialias=True)(sim_image.unsqueeze(0))

        ret.append(ret_img.squeeze())
    return ret


def batch_chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes cosine similarity between all possible pairs in two sets of vectors in a fully batched mode.
    
    :param x: A tensor of descriptors with shape Bx1x(t_x)x(d') where d' is the dimensionality of the descriptors and t_x
              is the number of tokens in x.
    :param y: A tensor of descriptors with shape Bx1x(t_y)x(d') where d' is the dimensionality of the descriptors and t_y
              is the number of tokens in y.
    :return: Cosine similarity between all descriptors in x and all descriptors in y, with shape Bx1x(t_x)x(t_y).
    """
    # Normalize x and y for cosine similarity calculation
    x_norm = x / x.norm(dim=-1, keepdim=True)  # Shape: (B, 1, t_x, d')
    y_norm = y / y.norm(dim=-1, keepdim=True)  # Shape: (B, 1, t_y, d')
    
    # Compute cosine similarity by taking dot product along the descriptor dimension
    # Result shape will be (B, 1, t_x, t_y) after matmul
    similarity = torch.matmul(x_norm, y_norm.transpose(-1, -2))  # Shape: (B, 1, t_x, t_y)
    
    return similarity  # returns Bx1x1x(t_x)x(t_y)


# def similarity_from_descriptors(
#     descriptors_batch,  # Shape: (num_descriptors, descriptor_dim)
#     images: list,
#     extractor: ViTExtractor,
#     load_size: int = 224,
#     layer: int = 11,
#     facet: str = "key",
#     bin: bool = False,
#     device="cuda",
# ):
#     """
#     Computes similarity between a batch of descriptors and descriptors extracted from multiple images.
    
#     :param descriptors_batch: Tensor of descriptors, shape (num_descriptors, descriptor_dim).
#     :param images: List of paths to image files.
#     :param load_size: Image load size for the model.
#     :param layer: Layer from which to extract descriptors.
#     :param facet: Facet type for descriptor extraction (e.g., "key", "query", etc.).
#     :param bin: Boolean flag for binning descriptors.
#     :param stride: Stride value for the model.
#     :param model_type: Type of model to extract descriptors with (e.g., "dino_vits8").
    
#     :return: List of resized similarity maps for each image.
#     """
#     similarities = []
#     img_size = Image.open(images[0]).size[::-1]  # Original image size for resizing
    
#     # Move descriptors batch to device for parallelized similarity calculation
#     descriptors_batch = descriptors_batch.to(device)  # Shape: (num_descriptors, descriptor_dim)

#     for image_path in images:
#         # Preprocess image and extract descriptors
#         image_batch, image_pil = extractor.preprocess(image_path, load_size)

#         descs_image = extractor.extract_descriptors(
#             image_batch.to(device), layer, facet, bin, include_cls=False
#         )  # Shape: (1, num_patches, descriptor_dim)

#         # Reshape descriptors for batch similarity calculation
#         descs_image = descs_image.squeeze(0)  # Shape: (num_patches, descriptor_dim)
        
#         # Expand and reshape to align with batch_chunk_cosine_sim requirements
#         # Add batch and singleton dimension to descriptors_batch and descs_image
#         # Result shapes: (num_descriptors, 1, 1, descriptor_dim) and (1, 1, num_patches, descriptor_dim)
#         descriptors_expanded = descriptors_batch[:, None, None, :] # Bx1x1xd'
#         descs_image_expanded = descs_image[None, :, :] # 1xtxd'

#         # Calculate cosine similarity using the batched chunk cosine sim function
#         similarity = batch_chunk_cosine_sim(descriptors_expanded, descs_image_expanded)  # Shape: (num_descriptors, 1, 1, num_patches)

#         # Aggregate similarity across descriptors 
#         combined_similarity = similarity.max(dim=0).values  # Shape: (1, num_patches)

#         similarity_image = combined_similarity.reshape(extractor.num_patches)

#         # similarity_map_resized = transforms.Resize(img_size, antialias=False)(similarity_image.unsqueeze(0))
#         similarity_map_resized = F.interpolate(
#             similarity_image.unsqueeze(0).unsqueeze(0),
#             size=img_size,
#             mode='nearest',  # `nearest` is faster than `bilinear`
#         ).squeeze()

#         # Append the similarity map for this image
#         similarities.append(similarity_map_resized.squeeze())
    
#     return similarities


def similarity_from_descriptors(
    descriptors_batch,  # Shape: (num_descriptors, descriptor_dim)
    image_paths: list,
    extractor: ViTExtractor,
    load_size: int = 224,
    layer: int = 11,
    facet: str = "key",
    bin: bool = False,
    device="cuda",
):
    """
    Computes similarity between a batch of descriptors and descriptors extracted from multiple images, in parallel.
    
    :param descriptors_batch: Tensor of descriptors, shape (num_descriptors, descriptor_dim).
    :param image_paths: List of paths to image files.
    :param load_size: Image load size for the model.
    :param layer: Layer from which to extract descriptors.
    :param facet: Facet type for descriptor extraction (e.g., "key", "query").
    :param bin: Boolean flag for binning descriptors.
    
    :return: Tensor of resized similarity maps, shape (batch_size, height, width).
    """
    # Preprocess images in a batch and get original sizes
    image_batch, original_size = extractor.preprocess_batch(image_paths, load_size)  # Shape: (batch_size, C, H, W)
    image_batch = image_batch.to(device)
    descriptors_batch = descriptors_batch.to(device)  # Move descriptors to device

    # Extract descriptors for the entire batch
    descs_images = extractor.extract_descriptors(
        image_batch, layer, facet, bin, include_cls=False
    )  # Shape: (batch_size, num_patches, descriptor_dim)

    # Expand dimensions for batch-wise cosine similarity calculation
    descriptors_expanded = descriptors_batch[:, None, None, :]  # Shape: (num_descriptors, 1, 1, descriptor_dim)
    descs_images_expanded = descs_images[:, None, :, :]  # Shape: (batch_size, 1, num_patches, descriptor_dim)

    # Compute cosine similarity for all descriptors and image patches in a single call
    similarity = batch_chunk_cosine_sim(descriptors_expanded, descs_images_expanded)  # Shape: (batch_size, 1, num_descriptors, num_patches)

    # Aggregate similarity by taking the max similarity across descriptors for each image patch
    combined_similarity = similarity.max(dim=1).values  # Shape: (batch_size, 1, num_patches)

    # Reshape for batch-wise resizing
    combined_similarity = combined_similarity.reshape(-1, 1, *extractor.num_patches)  # Shape: (batch_size, 1, num_patches_height, num_patches_width)

    # Resize all similarity maps in one operation to match each original image size
    resized_similarity_maps = F.interpolate(
        combined_similarity,  # Shape: (batch_size, 1, sqrt(num_patches), sqrt(num_patches))
        size=original_size,  # Each image's original size
        mode="bilinear",  # Bilinear interpolation for resizing
    ).squeeze(1)  # Remove channel dimension, final shape: (batch_size, height, width)

    return resized_similarity_maps

