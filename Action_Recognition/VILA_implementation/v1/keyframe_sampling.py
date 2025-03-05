import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from transformers import CLIPProcessor, CLIPModel, XCLIPProcessor, XCLIPModel
from typing import List
from PIL import Image
import matplotlib.pyplot as plt


anomaly_labels = [
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
]


# Select the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pre-trained ResNet model for feature extraction and move it to device
resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove classification layer
resnet = resnet.to(device).eval()  # Move to GPU if available

# FPS = 15

# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# model_name = "microsoft/xclip-large-patch14-16-frames"
# x_clip_model = XCLIPModel.from_pretrained(model_name).to(device)
# x_clip_processor = XCLIPProcessor.from_pretrained(model_name)

# Image transformation for ResNet
transform = transforms.Compose([
    transforms.Resize((336, 336)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_resnet_features(frame: Image.Image, device: torch.device):
    """Extract ResNet feature vector for a single frame."""
    frame = transform(frame).unsqueeze(0).to(device)  # Move to GPU if available
    with torch.no_grad():
        features = resnet(frame).squeeze().cpu().numpy()  # Move back to CPU before converting to NumPy
    return features

def extract_clip_features(input, model, processor, device, model_type='x-clip', index=0):
    if model_type == 'clip':
        text_input = processor(text=["Anomaly"], return_tensors="pt").to(device)
        text_features = model.get_text_features(**text_input)
        img_input = processor(images=input, return_tensors="pt").to(device)
        with torch.no_grad():
            img_features = model.get_image_features(**img_input)
        return torch.nn.CosineSimilarity(dim=-1)(text_features, img_features).item()
    elif model_type == 'x-clip':
        if len(input) < 16:
            return 0
        
        # Create a 4x4 grid of subplots (16 images)
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))  # 4x4 grid
        
        # Flatten axes for easy indexing
        axes = axes.flatten()

        # Loop over the images and plot them
        for i, ax in enumerate(axes):
            ax.imshow(input[i])
            ax.axis('off')  # Hide the axes for a cleaner image display
        
        plt.tight_layout()
        plt.savefig(f"G:/PHDResearch/Action_Recognition/VILA/stupid/frame_test_{index}.png")
        plt.close()

        tensor_frames = [transform(frame).unsqueeze(0) for frame in input]  # Add batch dim
        tensor_frames = torch.cat(tensor_frames, dim=0).to(device)  # Stack into a batch

        text_features_list = []

        # Compute text features for all labels
        for text_query in anomaly_labels:
            text_input = processor(text=[text_query], return_tensors="pt").to(device)
            text_features = model.get_text_features(**text_input)
            text_features_list.append(text_features)

        text_features_tensor = torch.cat(text_features_list, dim=0)  # Shape: (num_labels, feature_dim)

        # Process video input
        video_input = {"pixel_values": tensor_frames.unsqueeze(0)}  # Add batch dim for video
        with torch.no_grad():
            video_features = model.get_video_features(**video_input)

        # Compute cosine similarity for all labels
        similarity_scores = torch.nn.functional.cosine_similarity(text_features_tensor, video_features, dim=-1)
        
        print(f"Frame index: {index}, Scores: {similarity_scores.tolist()}")

        return similarity_scores.tolist()
    else:
        raise ValueError("Unsupported model type")

def compute_motion_score(prev_frame: np.ndarray, curr_frame: np.ndarray):
    """Compute motion score using absolute difference."""
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
    return np.sum(cv2.absdiff(gray_prev, gray_curr))  # Sum of pixel changes

def filter_motion_frames(frames: List[Image.Image], motion_threshold: float = 10000):
    """Remove static frames by comparing motion scores."""
    filtered_indices = [0]  # Always keep the first frame
    prev_frame = np.array(frames[0])

    for i in range(1, len(frames)):
        curr_frame = np.array(frames[i])
        motion_score = compute_motion_score(prev_frame, curr_frame)

        if motion_score > motion_threshold:
            filtered_indices.append(i)
            prev_frame = curr_frame  # Update reference frame
    
    return [frames[i] for i in filtered_indices], filtered_indices

def cluster_keyframes(frames: List[Image.Image], num_keyframes: int, device: torch.device):
    """Cluster frames using K-Means on ResNet features."""
    # Extract features for filtered frames
    feature_vectors = np.array([extract_resnet_features(frame, device) for frame in frames])
    
    # Perform K-Means clustering (KMeans in sklearn only works on CPU)
    kmeans = KMeans(n_clusters=min(num_keyframes, len(frames)), random_state=42, n_init=10)
    kmeans.fit(feature_vectors)
    
    # Get representative frames closest to each cluster center
    cluster_indices = []
    for cluster in range(kmeans.n_clusters):
        cluster_points = np.where(kmeans.labels_ == cluster)[0]
        center = kmeans.cluster_centers_[cluster]
        closest_frame = min(cluster_points, key=lambda idx: np.linalg.norm(feature_vectors[idx] - center))
        cluster_indices.append(closest_frame)

    return [frames[i] for i in sorted(cluster_indices)], sorted(cluster_indices)

def meanstd(len_scores, dic_scores, n, fns, t1, t2, all_depth):
    split_scores = []
    split_fn = []
    no_split_scores = []
    no_split_fn = []
    for dic_score, fn in zip(dic_scores, fns):
        score = dic_score['score']
        depth = dic_score['depth']
        mean = np.mean(score)
        std = np.std(score)
        # top_n = heapq.nlargest(n, range(len(score)), score.__getitem__)
        top_n = np.argsort(score)[-n:][::-1]  # Get indices of the top `n` scores in descending order
        top_score = [score[t] for t in top_n]
        mean_diff = np.mean(top_score) - mean
        if mean_diff > t1 and std > t2:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
        elif depth < all_depth:
            score1 = score[:len(score)//2]
            score2 = score[len(score)//2:]
            fn1 = fn[:len(score)//2]
            fn2 = fn[len(score)//2:]
            split_scores.append(dict(score=score1, depth=depth+1))
            split_scores.append(dict(score=score2, depth=depth+1))
            split_fn.append(fn1)
            split_fn.append(fn2)
        else:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
    if len(split_scores) > 0:
        all_split_score, all_split_fn = meanstd(len_scores, split_scores, n, split_fn, t1, t2, all_depth)
    else:
        all_split_score = []
        all_split_fn = []
    all_split_score = no_split_scores + all_split_score
    all_split_fn = no_split_fn + all_split_fn
    return all_split_score, all_split_fn

def remove_background(frames: List[Image.Image]):
    # Initialize background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    # Preprocess frames with background subtraction
    processed_frames = []
    for frame in frames:  # `input` is a list of PIL images
        np_frame = np.array(frame)  # Convert PIL to NumPy array
        np_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

        fg_mask = bg_subtractor.apply(np_frame)  # Get foreground mask
        fg = cv2.bitwise_and(np_frame, np_frame, mask=fg_mask)  # Apply mask

        fg = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)  # Convert BGR back to RGB
        pil_frame = Image.fromarray(fg)  # Convert NumPy array back to PIL image

        processed_frames.append(pil_frame)

    return processed_frames

def adaptive_keyframe_sampling(frames: List[Image.Image], num_keyframes: int, model, processor, device: torch.device, t1=0.8, t2=-100, all_depth=5, logger=None):
    fps = len(frames) / num_keyframes
    frame_indices = [i* int(fps) for i in range(int(len(frames) / fps))]
    # scores = [extract_clip_features(frame, model, processor, device) for frame in frames]

    frames = remove_background(frames)

    batch_size = 16
    stride = 1
    batches = [frames[i:i + batch_size] for i in range(0, len(frames) - batch_size + 1, stride * batch_size)]
    scores = [extract_clip_features(batch, model, processor, device, index=i) for i, batch in enumerate(batches)]

    # Convert to NumPy for easier processing
    scores_array = np.array(scores)  # Shape: (num_frames, num_labels)

    # # Aggregate scores per frame (you can use max, mean, or variance)
    # frame_scores = np.max(scores_array, axis=1)  # Taking max across anomaly labels per frame
    # Normalize along the frame axis (axis=0)
    min_vals = scores_array.min(axis=0, keepdims=True)
    max_vals = scores_array.max(axis=0, keepdims=True)
    normalized_scores = (scores_array - min_vals) / (max_vals - min_vals + 1e-8)  # Avoid division by zero

    # Aggregate scores per frame after normalization
    frame_scores = np.max(normalized_scores, axis=1)  # Taking max across anomaly labels per frame


    plt.figure(figsize=(10, 5))
    plt.plot(frame_scores, marker='o', linestyle='-', color='b', label="CLIP Similarity Scores")
    plt.xlabel("Frame Index")
    plt.ylabel("Similarity Score")
    plt.title("CLIP Text-Image Similarity Over Frames")
    plt.legend()
    plt.grid(True)
    plt.savefig("G:/PHDResearch/Action_Recognition/VILA/test.png")

    top_scores = np.argsort(frame_scores)[-10:][::-1]
    print("Top 10 scores; ", top_scores)

    for i, index_batch in enumerate(top_scores):
        frame_range = frames[index_batch * batch_size:(index_batch + 1) * batch_size]
        print(f"Frame {i+1}: {index_batch * batch_size} - {(index_batch + 1) * batch_size - 1}")

    # if len(scores) >= num_keyframes: always the case
    normalized_scores = [(score - np.min(scores)) / (np.max(scores) - np.min(scores)) for score in scores]
    segmented_scores, segmented_fns = meanstd(
        len(normalized_scores), 
        [{'score': normalized_scores, 'depth': 0}], 
        num_keyframes, 
        [frame_indices], 
        t1, 
        t2, 
        all_depth
    )

    # selected_frames = []
    selected_indices = []
    
    for seg_score, seg_fn in zip(segmented_scores, segmented_fns):
        f_num = int(num_keyframes / 2**(seg_score['depth']))
        topk = np.argsort(seg_score['score'])[-f_num:][::-1]
        f_nums = [seg_fn[t] for t in topk if t < len(seg_fn)]
        selected_indices.extend(f_nums)
    
    selected_indices.sort()
    selected_frames = [frames[i] for i in selected_indices]
    
    return selected_frames, selected_indices


def get_sampled_frames(all_frames: List[Image.Image], num_frames: int = 10, device: str = "cuda", logger = None):
    """Pipeline: Motion filtering → Feature extraction → K-Means clustering."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    filtered_frames, filtered_indices = filter_motion_frames(all_frames)
    keyframes, keyframe_indices = cluster_keyframes(filtered_frames, num_frames, device)
    keyframe_indices = [filtered_indices[i] for i in keyframe_indices]

    # clip_keyframes, clip_keyframe_indices = adaptive_keyframe_sampling(
    #     all_frames, 
    #     num_keyframes=num_frames, 
    #     model=x_clip_model, 
    #     processor=x_clip_processor, 
    #     device=device,
    #     logger=logger
    # )

    # print(f"final indices (Length: {len(keyframe_indices)}): {keyframe_indices}")
    # print(f"Clip indices: {len(clip_keyframe_indices)}, : {clip_keyframe_indices}")
    
    return keyframes, keyframe_indices


# Example Usage:
# sampled_frames, indices = get_sampled_frames(all_frames, num_frames=10)



# import typing
# from torchvision.models.resnet import Bottleneck, BasicBlock, ResNet
# import torch.utils.model_zoo as model_zoo
# import time
# from tqdm import tqdm

# if typing.TYPE_CHECKING:
#     from loguru import Logger
# else:
#     Logger = None

# __all__ = ["logger"]


# def __get_logger() -> Logger:
#     from loguru import logger

#     logger.add("logs/logfile.log", rotation="1 MB", retention="10 days", level="INFO")
#     return logger


# logger = __get_logger()

# RES_MODEL = 'resnet18'
# PICK_LAYER = 'avg'
# MAX_FRAMES = 10000
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Clustering config
# MEANS = np.array([103.939, 116.779, 123.68]) / 255.  # mean of 3 channels (BGR)
# SIMILARITY_THRESHOLD = 0.98 # 0.98
# CLU_MIN_THRESH = 2

# # Chunk config
# FRAME_SELECTION = 15  # 1/FRAME_SELECTION * FPS frames sampled per second (for 15 that equals to 2)
# UNIFORMLY_SAMPLED_FRAMES = -1 # -1 = full video, X < num_video_frames = X uniformly sampled frames
# FPS = 30  # frames per second
# OVERLAP_FRAMES = 4 # frames overlap between chunks
# NUM_VIDEO_FRAMES = 16 # Number of frames for model


# # from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }


# class ResidualNet(ResNet):
#     def __init__(self, model=RES_MODEL, pretrained=True):
#         if model == "resnet18":
#             super().__init__(BasicBlock, [2, 2, 2, 2], 1000)
#             if pretrained:
#                 self.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
#         elif model == "resnet34":
#             super().__init__(BasicBlock, [3, 4, 6, 3], 1000)
#             if pretrained:
#                 self.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
#         elif model == "resnet50":
#             super().__init__(Bottleneck, [3, 4, 6, 3], 1000)
#             if pretrained:
#                 self.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
#         elif model == "resnet101":
#             super().__init__(Bottleneck, [3, 4, 23, 3], 1000)
#             if pretrained:
#                 self.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
#         elif model == "resnet152":
#             super().__init__(Bottleneck, [3, 8, 36, 3], 1000)
#             if pretrained:
#                 self.load_state_dict(model_zoo.load_url(model_urls['resnet152']))

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)  # x after layer4, shape = N * 512 * H/32 * W/32
#         max_pool = torch.nn.MaxPool2d((x.size(-2), x.size(-1)), stride=(x.size(-2), x.size(-1)), padding=0, ceil_mode=False)
#         Max = max_pool(x)  # avg.size = N * 512 * 1 * 1
#         Max = Max.view(Max.size(0), -1)  # avg.size = N * 512
#         avg_pool = torch.nn.AvgPool2d((x.size(-2), x.size(-1)), stride=(x.size(-2), x.size(-1)), padding=0, ceil_mode=False, count_include_pad=True)
#         avg = avg_pool(x)  # avg.size = N * 512 * 1 * 1
#         avg = avg.view(avg.size(0), -1)  # avg.size = N * 512
#         fc = self.fc(avg)  # fc.size = N * 1000
#         output = {
#             'max': Max,
#             'avg': avg,
#             'fc': fc
#         }
#         return output


# class ResNetFeat(object):
#     def make_samples(self, frame_set):
#         start_time = time.time()
#         res_model = ResidualNet(model=RES_MODEL)
#         res_model.eval()
#         res_model.to(DEVICE)
#         global init_time
#         init_time = time.time() - start_time
#         print(f'ResNet initiation cost {init_time} seconds')

#         # features
#         samples = []
#         # for idx in tqdm(frame_set):
#         for idx in tqdm(range(len(frame_set))):
#             img = frame_set[idx]
#             img = np.transpose(img, (2, 0, 1)) / 255.
#             img[0] -= MEANS[0]  # reduce B's mean
#             img[1] -= MEANS[1]  # reduce G's mean
#             img[2] -= MEANS[2]  # reduce R's mean
#             img = np.expand_dims(img, axis=0)
#             try:
#                 inputs = torch.autograd.Variable(torch.from_numpy(img).to(DEVICE).float())
#                 d_hist = res_model(inputs)[PICK_LAYER]
#                 d_hist = d_hist.data.cpu().numpy().flatten()
#                 d_hist /= np.sum(d_hist)  # normalize
#                 samples.append({
#                                 'img_idx': idx,
#                                 'hist': d_hist
#                                 })

#             except ValueError as e:
#                 print("Wrong", e)
#                 raise

#         return samples, init_time


# def extract_high_lev_f(frame_set):
#     samples, init_time = ResNetFeat().make_samples(frame_set)
#     f_mat = np.vstack([d['hist'] for d in samples])

#     return f_mat, init_time


# def clustering(f_mat, clu_min_thres=CLU_MIN_THRESH, similarity_threshold=SIMILARITY_THRESHOLD):
#     # dynamic clustering of projected frame histograms to find shots
#     cluster_set = dict()
#     for i in range(f_mat.shape[0]):
#         cluster_set[i] = np.empty((0, f_mat.shape[1]), int)

#     # initialize the cluster
#     cluster_set[0] = np.vstack((cluster_set[0], f_mat[0]))
#     cluster_set[0] = np.vstack((cluster_set[0], f_mat[1]))

#     centroid_set = dict()  # to store centroids of each cluster
#     for i in range(f_mat.shape[0]):
#         centroid_set[i] = np.empty((0, f_mat.shape[1]), int)
#     # finding centroid of centroid_set[0] cluster
#     centroid_set[0] = np.mean(cluster_set[0], axis=0)

#     count = 0
#     for i in range(2, f_mat.shape[0]):
#         if np.all(f_mat[i] == 0) or np.all(centroid_set[count] == 0):
#             logger.warning(f"Zero vector found at index {i} or centroid {count}. Skipping.")
#         similarity2 = np.dot(f_mat[i], centroid_set[count])**2/(np.dot(f_mat[i], f_mat[i])*np.dot(centroid_set[count], centroid_set[count]))
#         # logger.info(f"Similarity: {similarity2}")
#         if similarity2 < similarity_threshold:
#             count += 1
#             cluster_set[count] = np.vstack((cluster_set[count], f_mat[i]))
#             centroid_set[count] = np.mean(cluster_set[count], axis=0)
#         else:
#             cluster_set[count] = np.vstack((cluster_set[count], f_mat[i]))
#             centroid_set[count] = np.mean(cluster_set[count], axis=0)

#     num = []  # find the number of data points in each cluster formed.
#     for i in range(f_mat.shape[0]):
#         # logger.info(f"Cluster {i} has {cluster_set[i].shape[0]} frames.")
#         num.append(cluster_set[i].shape[0])
#         if cluster_set[i].shape[0] == 0:
#             break
    

#     KF_idx = []
#     KF_vec = []
#     i = 0
#     s = 0
#     while num[i] != 0:
#         if num[i] >= clu_min_thres:
#             new_KF_idx = int(s + (num[i]+1)//2 - 1)  # ceiling
#             # new_KF_idx = s + num[i] - 1  # ceiling (the second last frame)
#             KF_idx.append(new_KF_idx)  # python idx start from 0
#             KF_vec.append(f_mat[new_KF_idx])
#         s += num[i]
#         i += 1
#     KF_vec = np.array(KF_vec)
#     return KF_vec, KF_idx

# def _sample_frames(frame_set, num_frames):
#     f_mat, init_time = extract_high_lev_f(frame_set)
#     KF_vec, KFs = clustering(f_mat)
#     # Without sampling, put middle frame in KFs[0]
#     # KFs = [len(frame_set) // 2]

#      # Adjust to ensure exactly `num_frames`
#     if len(KFs) < num_frames and len(KFs) > 1:
#         # Interpolate additional frames
#         num_frames_to_interpolate = num_frames - len(KFs)
#         for i in range(num_frames_to_interpolate):
#             # Find biggest #num_frames_to_interpolate gaps
#             gap_sizes = np.diff(KFs)
#             largest_gap_idx = np.argmax(gap_sizes)
#             new_frame_idx = KFs[largest_gap_idx] + gap_sizes[largest_gap_idx] // 2
#             KFs.insert(largest_gap_idx + 1, new_frame_idx)
#             KFs = [int(idx) for idx in KFs]  # Convert to int
#             KFs.sort()  # Sort to maintain order
#     elif len(KFs) > num_frames:
#         # Prune frames to match `num_frames`
#         step = len(KFs) / num_frames
#         KFs = [KFs[int(i * step)] for i in range(num_frames)]
#     elif len(KFs) == 1:
#         # Special case: only one keyframe
#         logger.warning(f"Only one keyframe found, spreading around keyframe {KFs[0]}.")
#         center_frame = KFs[0]
#         half_span = min(center_frame, len(frame_set) - center_frame - 1)  # Avoid out-of-bounds
#         range_start = max(0, center_frame - half_span)
#         range_end = min(len(frame_set) - 1, center_frame + half_span)
#         KFs = np.linspace(range_start, range_end, num_frames).astype(int).tolist()
#     else:
#         # uniform sampling
#         logger.warning(f"Sampled frames uniformly because there are no items in KFs.")
#         logger.info(f"Sampled frames: {KFs}")
#         logger.info(f"Number of frames: {len(frame_set)}")
#         logger.info(f"KF_vec: {KF_vec}")
#         logger.info(f"Featuress shape: {f_mat.shape}")
#         KFs = [int(i) for i in np.linspace(0, len(frame_set) - 1, num_frames)]

#     return [int(idx) for idx in KFs]

# def get_sampled_frames(all_frames, num_frames: int) -> List[PIL.Image.Image]:
#     # Convert all_frames to list of numpy array
#     numpy_frames = [np.array(frame) for frame in all_frames]

#     # Use `_sample_frames` to determine which frames to return
#     indices = _sample_frames(numpy_frames, num_frames=num_frames)
#     # indices = motion_guided_sampling(all_frames, num_frames=num_frames)
#     print(colored(f"Sampled frames: {indices}", "green"))
#     sampled_frames = []
#     for index in indices:
#         frame = all_frames[index]
#         sampled_frames.append(frame)

#     return sampled_frames, indices