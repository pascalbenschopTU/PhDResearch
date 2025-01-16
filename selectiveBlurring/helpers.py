import os

import numpy as np
import PIL
import torch
from torchvision import transforms
from scipy import interpolate
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import save_image
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
import cv2


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


def similarity_from_descriptors(
    descriptors_batch,  # Shape: (num_descriptors, descriptor_dim)
    images: torch.Tensor,  # Shape: (batch_size, C, H, W)
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
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    
    image_batch, original_size = extractor.preprocess_batch_from_tensor(images, load_size=(load_size, load_size))  # Shape: (batch_size, C, H, W)
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

    return combined_similarity

def batch_warp(img, flow, mode="bilinear", padding_mode="zeros", device="cuda"):
    # img.shape -> B, 3, H, W
    # flow.shape -> B, H, W, 2
    (
        b,
        c,
        h,
        w,
    ) = img.shape
    y_coords = torch.linspace(-1, 1, h)
    x_coords = torch.linspace(-1, 1, w)
    f0 = (
        torch.stack(torch.meshgrid(x_coords, y_coords))
        .permute(2, 1, 0)
        .repeat(b, 1, 1, 1)
    ).to(device)

    f = f0 + torch.stack([2 * (flow[..., 0] / w), 2 * (flow[..., 1] / h)], dim=3)
    warped = F.grid_sample(img, f, mode=mode, padding_mode=padding_mode)
    return warped.squeeze()

def firstframe_warp(
    flows, interpolation_mode="nearest", usecolor=True, seed_image=None, device="cuda"
):
    if len(flows) == 0:
        return None
    
    h, w, _ = flows[0].shape
    c = 3 if usecolor == True else 1

    t = len(flows)
    # flows = torch.stack(flows)

    if usecolor:
        inits = (torch.rand((1, c, h, w))).repeat(t, 1, 1, 1).to(device)
    else:
        inits = (torch.rand((1, 1, h, w))).repeat(t, 3, 1, 1).to(device)

    if seed_image != None:
        if isinstance(seed_image, str):
            inits = read_image(seed_image, ImageReadMode.RGB) / 255.0
        else:
            inits = seed_image
        
        inits = Resize((h, w), interpolation=InterpolationMode.NEAREST)(inits)
        inits = inits.repeat(t, 1, 1, 1)

    warped = batch_warp(inits, flows, mode=interpolation_mode, device=device)
    masks = ~(warped.any(dim=1))
    masks = masks.unsqueeze(1).repeat(1, 3, 1, 1)
    warped[masks] = inits[masks]
    warped = torch.cat([inits[0, ...].unsqueeze(dim=0), warped], dim=0)
    warped = (warped).clip(0, 1)
    # warped = Resize(size=(int(h * 0.5), int(w * 0.5)))(warped)
    # warped = warped[:, :, 10:-10, 10:-10]
    warped = (warped.permute(0, 2, 3, 1) * 255).float()
    return warped

def upsample_flow(flow, h, w):
    # usefull function to bring the flow to h,w shape
    # so that we can warp effectively an image of that size with it
    h_new, w_new, _ = flow.shape
    flow_correction = torch.Tensor((h / h_new, w / w_new))
    f = flow * flow_correction[None, None, :]

    f = (
        Resize((h, w), interpolation=InterpolationMode.BICUBIC)(f.permute(2, 0, 1))
    ).permute(1, 2, 0)
    return f

def flow_from_video(model, frames_list, upsample_factor=1, device="cpu", iters=12):
    num_frames = len(frames_list)
    h, w = frames_list[0].shape[1:3]  # Height and width of frames

    # Prepare frames as a batch
    # frames_tensor = torch.stack(frames_list).to(device)
    frames_tensor = frames_list.to(device)

    # Optionally upsample frames once
    if upsample_factor != 1:
        frames_tensor = F.interpolate(frames_tensor, scale_factor=upsample_factor, mode="bilinear")

    # Create batched frame pairs
    image0 = frames_tensor[:-1]  # All except the last frame
    image1 = frames_tensor[1:]   # All except the first frame

    # Pad all pairs together
    padder = InputPadder(image0.shape)
    image0, image1 = padder.pad(image0, image1)

    # Process all pairs in one batch
    # with torch.no_grad():
    _, flow_up = model(image0, image1, iters=iters, test_mode=True)

    # Remove padding and reshape flows
    flow_output = padder.unpad(flow_up)

    # Optionally resize flows to the original resolution
    if upsample_factor != 1:
        flow_output = F.interpolate(flow_output, size=(h, w), mode="bilinear")

    return flow_output.permute(0, 2, 3, 1)  # Convert to (H, W, 2)


# def flow_from_video(model, frames_list, upsample_factor=1, device="cuda", iters=12):
#     num_frames = len(frames_list)
#     all_flows = []
#     c, h, w = frames_list[0].shape
#     with torch.no_grad():
#         for i in range(num_frames - 1):
#             image0 = frames_list[i].unsqueeze(0).to(device)
#             image1 = frames_list[i + 1].unsqueeze(0).to(device)

#             if upsample_factor != 1:
#                 image0 = nn.Upsample(scale_factor=upsample_factor, mode="bilinear")(
#                     image0
#                 )
#                 image1 = nn.Upsample(scale_factor=upsample_factor, mode="bilinear")(
#                     image1
#                 )

#             padder = InputPadder(image0.shape)
#             image0, image1 = padder.pad(image0, image1)
#             _, flow_up = model(image0, image1, iters=iters, test_mode=True)
#             flow_output = (
#                 padder.unpad(flow_up).detach().cpu().squeeze().permute(1, 2, 0)
#             )
#             fl = flow_output.detach().cpu().squeeze()
#             fl = upsample_flow(fl, h, w)
#             all_flows.append(fl)
#     return all_flows


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)
