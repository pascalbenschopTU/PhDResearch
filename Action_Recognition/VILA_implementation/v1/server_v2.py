import argparse
import base64
import glob
import json
import os
import re
import time
from collections import defaultdict
from typing import List, Literal, Optional, Union, Dict, Any, Union

import easydict
import numpy as np
import PIL
import matplotlib.pyplot as plt
import requests
import torch

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from termcolor import colored
from pydantic import BaseModel
from tqdm import tqdm
from transformers import PretrainedConfig
from transformers.generation import GenerationConfig

import decord
from decord import VideoReader

from llava import conversation as clib
from llava.constants import MEDIA_TOKENS
from llava.model.builder import load_pretrained_model
from llava.media import Video, Image
from llava.utils import disable_torch_init, make_list
# from llava.utils.logging import logger
from llava.utils.tokenizer import tokenize_conversation
from llava.mm_utils import process_image, process_images

from keyframe_sampling import get_sampled_frames

# Constants
DEFAULT_MODEL_PATH = "Efficient-Large-Model/VILA1.5-3B"
DEFAULT_CONV_MODE = "auto"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
MAX_FRAMES = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chunk config
FRAME_SELECTION = 15  # 1/FRAME_SELECTION * FPS frames sampled per second (for 15 that equals to 2)
UNIFORMLY_SAMPLED_FRAMES = -1 # -1 = full video, X < num_video_frames = X uniformly sampled frames
FPS = 30  # frames per second
OVERLAP_FRAMES = 4 # frames overlap between chunks
NUM_VIDEO_FRAMES = 16 # Number of frames for model

# # Generation config
DEFAULT_GENERATION_CONFIG = {
    "temperature": 0.05, # Lower is more deterministic
    "do_sample": True,
    "max_new_tokens": 64, # Less tokens possible in answers
    "min_new_tokens": 20, # Min 20 tokens in answer
    "repetition_penalty": 1.7, # Less repetitions in answer
}


model = None
model_name = None
tokenizer = None
image_processor = None
context_len = None

app = FastAPI()

class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ImageURL(BaseModel):
    url: str


class ImageContent(BaseModel):
    type: Literal["image_url"]
    image_url: ImageURL


IMAGE_CONTENT_BASE64_REGEX = re.compile(r"^data:image/(png|jpe?g);base64,(.*)$")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[TextContent, ImageContent]]]


class ChatCompletionRequest(BaseModel):
    model: Literal[
        "VILA1.5-3B",
        "VILA1.5-3B-AWQ",
        "VILA1.5-3B-S2",
        "VILA1.5-3B-S2-AWQ",
        "Llama-3-VILA1.5-8B",
        "Llama-3-VILA1.5-8B-AWQ",
        "VILA1.5-13B",
        "VILA1.5-13B-AWQ",
        "VILA1.5-40B",
        "VILA1.5-40B-AWQ",
        "NVILA-8B",
    ]
    messages: List[ChatMessage]
    max_tokens: Optional[int] = DEFAULT_GENERATION_CONFIG["max_new_tokens"]
    top_p: Optional[float] = 0.9
    temperature: Optional[float] = DEFAULT_GENERATION_CONFIG["temperature"]
    stream: Optional[bool] = False
    use_cache: Optional[bool] = True
    num_beams: Optional[int] = 1

import typing

if typing.TYPE_CHECKING:
    from loguru import Logger
else:
    Logger = None

__all__ = ["logger"]


def __get_logger() -> Logger:
    from loguru import logger

    logger.add("logs/logfile.log", rotation="1 MB", retention="10 days", level="INFO")
    return logger


logger = __get_logger()

def _load_video(video_path: str, *, num_sample_frames: int, max_frames=MAX_FRAMES) -> List[PIL.Image.Image]:
    # Load video frames from a directory
    if os.path.isdir(video_path):
        frame_paths = sorted(glob.glob(os.path.join(video_path, "*")))[:max_frames]
        indices = np.round(np.linspace(0, len(frame_paths) - 1, num_sample_frames)).astype(int)
        return [Image.open(frame_paths[index]) for index in indices]
    
    decord.bridge.set_bridge("torch")  # Use PyTorch-backed decoding
    vr = VideoReader(video_path)
    num_video_frames = len(vr)  # Get frame count efficiently

    if num_video_frames <= 0:
        raise ValueError(f"Video '{video_path}' has no frames.")
    
    indices = np.arange(num_video_frames)
    
    # If num_sample_frames is an integer between 2 and num_video_frames
    # Sample the amount of num_sample_frames, acording to strategy uniform
    if num_sample_frames > 1 and num_sample_frames < num_video_frames:
        indices = np.round(np.linspace(0, num_video_frames, num_sample_frames)).astype(int)
    elif num_video_frames > max_frames:
        logger.info(f"Reducing {num_video_frames} to {max_frames}")
        indices = np.round(np.linspace(0, num_video_frames, max_frames)).astype(int)

    return load_video_frames(video_reader=vr, indices=indices)

def load_video_frames(video_reader, indices):
    # Ensure indices are within bounds
    valid_indices = [i for i in indices if i < len(video_reader)]

    frames = video_reader.get_batch(valid_indices).numpy()  # Batch decoding
    return [PIL.Image.fromarray(frame) for frame in frames]

def split_sampled_videos_into_chunks(sampled_frames, timestamps, chunk_duration: int):
    chunks = []
    timestamps_per_chunk = []

    for i in range(0, len(sampled_frames), chunk_duration - OVERLAP_FRAMES):
        chunk = sampled_frames[i:i+chunk_duration]
        chunk_timestamps = timestamps[i:i+chunk_duration]
        chunks.append(chunk)
        timestamps_per_chunk.append(chunk_timestamps)

    return chunks, timestamps_per_chunk


def _extract_image(image: Union[Image, PIL.Image.Image]) -> PIL.Image.Image:
    if isinstance(image, Image):
        if image.path.startswith("http://") or image.path.startswith("https://"):
            image = PIL.Image.open(requests.get(image.path, stream=True).raw)
        else:
            image = PIL.Image.open(image.path)
    return image

def _extract_video(video: Video, config: PretrainedConfig) -> List[PIL.Image.Image]:
    num_sample_frames = config.num_video_frames
    if getattr(config, "fps") != 0:
        logger.warning("Extracting frames from video with specified FPS is not supported yet. Ignored.")

    frames = _load_video(video.path, num_sample_frames=num_sample_frames)
    logger.info(f"Extracted {len(frames)} frames from video '{video.path}'.")
    return frames


def extract_media(
    messages: List[Dict[str, Any]],
    config: Optional[PretrainedConfig] = None,
    draft: bool = False,
) -> Dict[str, List[Any]]:
    media = defaultdict(list)
    for message in messages:
        text = ""
        for part in make_list(message["value"]):
            if isinstance(part, str):
                for token in MEDIA_TOKENS.values():
                    if token in part:
                        logger.warning(f"Media token '{token}' found in text: '{part}'. Removed.")
                        part = part.replace(token, "").strip()
                text += part
            elif isinstance(part, (Image, PIL.Image.Image)):
                if draft:
                    media["image"].append(part)
                else:
                    media["image"].append(_extract_image(part))
                text += MEDIA_TOKENS["image"]
            elif isinstance(part, Video):
                if draft:
                    media["video"].append(part)
                else:
                    media["video"].append(_extract_video(part, config))
                text += MEDIA_TOKENS["video"]
            else:
                raise ValueError(f"Unsupported prompt part type: {type(part)}")
        message["value"] = text
    return media


@torch.inference_mode()
def generate_predictions_NVILA(images, conversation, generation_config=None):
    media_config = defaultdict(dict)
    generation_config = generation_config or GenerationConfig(**DEFAULT_GENERATION_CONFIG)

     # Ensure each chunk is a list of frames
    processed_images = [
        process_images(images, model.vision_tower.image_processor, model.config).half()
    ]

    with torch.no_grad():
        try:
            input_ids = tokenize_conversation(
                conversation,
                model.tokenizer,
                add_generation_prompt=True
            ).cuda().unsqueeze(0)

            output_ids = model.generate(
                input_ids=input_ids,
                media={"video": processed_images},
                media_config=media_config,
                generation_config=generation_config,
            )

            # Decode the response
            response = model.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

            # Update conversation with the model's response
            # conversation.append({"from": "gpt", "value": response})
            return response
        except ValueError as e:
            logger.error(f"Failed to generate response: {e}")
            return ""
        

@torch.inference_mode()
def generate_content_for_video(
    prompt: Union[str, List],
    generation_config: Optional[GenerationConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Generates predictions for a video by splitting it into chunks and predicting on each chunk.
    
    Args:
        video (torch.Tensor): Input video as a tensor.
        generation_config (Optional[GenerationConfig]): Configuration for text generation.
        chunk_duration (int): Duration of each chunk in seconds.
    
    Returns:
        List[Dict[str, Any]]: A list of predictions with timestamps.
    """
    conversation = [{"from": "human", "value": prompt}]

    # model.config.num_video_frames = -1
    # Create an EasyDict object
    media_config = easydict.EasyDict({
        "num_video_frames": UNIFORMLY_SAMPLED_FRAMES,
        "fps": FPS
    })

    # Extract media from the conversation
    media = extract_media(conversation, media_config)

    if "video" in media:
        video_frames = media["video"][0]
    elif "image" in media:
        video_frames = [media["image"][0]]

    num_sampled_frames = len(video_frames) // FRAME_SELECTION
    sampled_frames, frame_indices = get_sampled_frames(video_frames, num_sampled_frames, logger=logger)

    timestamps = np.array(frame_indices) / FPS
    video_chunks, timestamps = split_sampled_videos_into_chunks(
        sampled_frames,
        timestamps=timestamps,
        chunk_duration=model.config.num_video_frames #NUM_VIDEO_FRAMES
    )

    predictions = []
    for i, chunked_sampled_images in enumerate(video_chunks):
        assert len(chunked_sampled_images) > 0, f"Empty video chunk at index {i}"

        # Include timestamp in the prompt for this chunk
        timestamp_range = f"[{timestamps[i][0]:.2f}s - {timestamps[i][-1]:.2f}s]" if len(timestamps[i]) > 1 else f"[{timestamps[i][0]:.2f}s]"
  
        logger.info(f"Generating response for chunk {i}/{len(video_chunks)}, timestamp: {timestamp_range}, length: {len(chunked_sampled_images)} frames")

        response = generate_predictions_NVILA(chunked_sampled_images, conversation, generation_config=generation_config)
        predictions.append({
            "timestamp": timestamp_range,
            "response": response
        })


    return predictions

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, image_processor
    disable_torch_init()
    model_path = os.getenv("VILA_MODEL_PATH", "Efficient-Large-Model/VILA1.5-3B")
    model_name = os.path.basename(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, model_name, None)
    print(f"Model {model_name} loaded successfully.")

    # Set conversation mode
    clib.default_conversation = clib.conv_templates[args.conv_mode].copy()


@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle inference requests with text and optional video."""
    try:
        # Prepare multi-modal prompt
        messages = request.messages

        prompt = []
        for message in messages:
            if message.role == "user":
                if isinstance(message.content, str):
                    prompt.append(message.content)
                if isinstance(message.content, list):
                    for content in message.content:
                        if content.type == "text":
                            prompt.append(content.text)
                        elif content.type == "image_url":
                            if content.image_url.url.endswith((".jpg", ".jpeg", ".png", ".heic")):
                                media = Image(content.image_url.url)
                                prompt.append(media)
                            elif content.image_url.url.endswith((".mp4", ".mkv", ".webm", ".avi")):
                                media = Video(content.image_url.url)
                                prompt.append(media)
                        else:
                            raise ValueError(f"Unsupported media type: {content}")

        # Generate response
        response = generate_content_for_video(prompt)
        logger.info(f"Response: {response}")

        return response
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-path", type=str, default="NVILA-8B")
    parser.add_argument("--conv-mode", type=str, default="auto")
    args = parser.parse_args()

    os.environ["VILA_MODEL_PATH"] = args.model_path
    uvicorn.run(app, host=args.host, port=args.port)
