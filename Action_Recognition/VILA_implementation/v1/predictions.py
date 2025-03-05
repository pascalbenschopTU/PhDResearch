import argparse
import os
import requests
import time
from tqdm import tqdm
from transformers import pipeline
from openai import OpenAI
from llava.utils.logging import logger
import json
import re

url = "http://localhost:8000/chat/completions"

client = OpenAI(
    base_url="http://localhost:8000",
    api_key="fake-key",
)


def get_response(question="What is happening in this video?", video_path=None):

    response = client.chat.completions.create(
        model="NVILA-8B",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": video_path,
                        },
                    },
                ],
            }
        ],
    )

    return response


labels = [
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "Normal",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
]

# response3 = get_response(question="Name any anomalous actions you see in the video.", video_path=abuse_video_path)
# response3 = get_response(question="Is there any anomaly in the video? ", video_path=abuse_video_path)

# prompt = f"""You are analyzing a video for anomalous actions based on the following labels: {", ".join(labels)}.
#     Your task is to answer YES or NO to whether the current video segment contains anomalies. 
#     Provide a label if you see any.
#     Answer in the following format: YES/NO, [label if YES]
# """

label_desc_map = {
    # "Abuse": "Abuse",
    # "Arrest": "Arrest",
    # "Arson": "Arson",
    # "Assault": "Assault",
    # "Burglary": "Burglary",
    # "Explosion": "Explosion",
    # "Fighting": "Fighting",
    # "Normal": "Normal",
    # "RoadAccidents": "RoadAccidents",
    # "Robbery": "Robbery",
    # "Shooting": "Shooting",
    # "Shoplifting": "Shoplifting",
    # "Stealing": "Stealing",
    # "Vandalism": "Vandalism",

    # Normal activities to balance anomalies
    "Walking on the sidewalk": "Normal",
    "Standing and waiting": "Normal",
    "Talking to someone calmly": "Normal",
    "Checking phone": "Normal",
    "Looking around casually": "Normal",
    "Sitting on a bench": "Normal",
    "Paying for items at checkout": "Normal",
    "Getting into or out of a vehicle": "Normal",
    "Car stopping at a traffic light": "Normal",
    "Waiting in line": "Normal",
    "Petting a dog or walking a pet": "Normal",
    "Cyclists riding in a bike lane": "Normal",
    "Pedestrians crossing with a green light": "Normal",

    # Anomaly activities
    "Car theft": "Stealing",
    "Hitting, Punching or Kicking people": "Fighting, Assault, Abuse",
    "Attacking someone": "Assault",
    "Setting fire": "Arson",
    "Pouring gasoline": "Arson",
    "Abusing animals": "Abuse",
    "Mistreatment of elderly": "Abuse",
    "Kneeling with arms behind back": "Arrest",
    "Pushing someone on the ground": "Abuse, Arrest",
    "Firing a gun or firearm": "Shooting",
    "Pointing a gun at someone": "Robbery, Shooting",
    "Handcuffing someone": "Arrest",
    "Destruction": "Vandalism",
    "Damage to property": "Vandalism",
    "Collision with vehicle": "RoadAccidents",
    "Crashing": "RoadAccidents",
    "Hitting pedestrian with vehicle": "RoadAccidents",
    "Taking goods without paying": "Shoplifting",
    "Hiding supermarket items in clothing": "Shoplifting",
    "Threatening a cashier": "Robbery",
    "Taking items from someone without permission": "Stealing",
    "Theft": "Stealing, Shoplifting, Robbery",
    "Forced entry of property": "Burglary"
}

# prompt = f"""You are analyzing a video for anomalous actions. Below is a list of possible descriptions:

# {', '.join(label_desc_map.keys())} 

# Your task is to determine what is anomalous about the video.  

# Respond in the following format:  
# **[anomaly], [explanation]** (if you see any anomaly)
# **Normal** (if no anomaly is present)
# """

prompt = f"""Analyze a video segment and classify actions as normal or anomalous.  
Use the descriptions below if they are similar to actions in the video:  

{', '.join(label_desc_map.keys())}  

### **Response Format:**  
- **[Anomaly Label(s)] - [Brief explanation]**  
  Example: **Shooting - A person is firing a gun.**  
- **Normal** (if no anomaly is detected)  

### **Guidelines:**  
- Assign **all relevant labels** if multiple apply.
- Do Not mention labels that are not applicable.   
"""

print(f"Prompt: {prompt}")

# print(f"Prompt: {prompt}")

# Load the zero-shot-classification pipeline
classifier_func = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

VIDEO_PATH = "../datasets/UCF_Crime/videos/"
# VIDEO_PATH = "../datasets/XD-Violence/videos/videos/"

test_annotation_file = "../labeling/labeled_test_001.txt"
# test_annotation_file = "../labeling/xd_labeled_text.txt"
# Action_Recognition\datasets\UCF_Crime\Annotations\Anomaly_Test.txt
# test_annotation_file = "../datasets/UCF_Crime/Annotations/Anomaly_Test.txt"
# example: Abuse/Abuse028_x264 0 1411 0
with open(test_annotation_file, "r") as f:
    lines = f.readlines()

label_map = {
    0: "Abuse",
    1: "Arrest",
    2: "Arson",
    3: "Assault",
    4: "Burglary",
    5: "Explosion",
    6: "Fighting",
    7: "Normal",
    8: "RoadAccidents",
    9: "Robbery",
    10: "Shooting",
    11: "Shoplifting",
    12: "Stealing",
    13: "Vandalism",
}

correlation_map = {
    "Stealing": {"Shoplifting": 0.8, "Burglary": 0.5, "Robbery": 0.3},
    "Shoplifting": {"Stealing": 0.8, "Burglary": 0.5, "Robbery": 0.3},
    "Arson": {"Explosion": 0.7},
    "Explosion": {"Arson": 0.7},
    "Robbery": {"Burglary": 0.75, "Stealing": 0.6},
    "Burglary": {"Robbery": 0.75, "Shoplifting": 0.5},
    "Assault": {"Fighting": 0.7, "Abuse": 0.6},
    "Abuse": {"Assault": 0.6, "Fighting": 0.1},
    "Fighting": {"Assault": 0.8},
    "Arrest": {"Fighting": 0.2},
    "RoadAccidents": {"Normal": 0.1},
}

correct_per_class = {label: 0 for label in label_map.values()}
incorrect_per_class = {label: 0 for label in label_map.values()}

adjusted_correct_per_class = {label: 0.0 for label in label_map.values()}

correct_per_class_top_3 = {label: 0 for label in label_map.values()}

def get_video_prediction(video_path):
    """Fetches the response from the classifier for a given video."""
    response = get_response(question=prompt, video_path=video_path)
    output_dict = {}

    for chatCompletionMessage in response:
        response_text = chatCompletionMessage.response

        # Append labels to matched descriptions
        for desc, mapped_label in sorted(label_desc_map.items(), key=lambda x: -len(x[0])):  # Sort by length to avoid substring conflicts
            pattern = re.escape(desc)  # Escape special characters
            response_text = re.sub(rf"\b{pattern}\b", f"{desc}: {mapped_label}", response_text, flags=re.IGNORECASE)

        # Append ": Normal" after "No" or "no"
        response_text = re.sub(r"\bNo\b", "No: Normal", response_text)
        response_text = re.sub(r"\bno\b", "no: Normal", response_text)


        # # Construct the question prompt
        # question = (
        #     "\n\n Does the text describe an anomaly? If yes, select the most relevant label. "
        #     "If not, select 'Normal'.\n"
        # )

        #  Construct the question prompt
        question = (
            "\n\n Does the text describe an anomaly? If yes, select the most relevant label. "
            "If not, select 'Normal'.\n"
        )

        # Combine everything into the final text sent to the classifier
        # response_text = chatCompletionMessage.response + likely_labels_text + question
        text_prompt = question + response_text

        prediction = classifier_func(text_prompt, candidate_labels=list(label_map.values()))

        predicted_labels = prediction["labels"][:3]
        predicted_scores = prediction["scores"][:3]

        logger.info(f"{response_text}, : prediction: {predicted_labels}")

        output_dict[chatCompletionMessage.timestamp] = {
            "LLM_output": response_text,
            "LLM_classes": predicted_labels,
            "LLM_confidence": predicted_scores,
        }

    return output_dict

def evaluate_prediction(label, video_prediction):
    """Evaluates the model's predictions and updates accuracy counters."""
    actual_label = label_map[label]
    top_1_score = 0.0
    top_1_adjusted_score = 0.0
    top_3_score = 0.0
    
    for timestamp, prediction_data in video_prediction.items():
        predicted_labels = prediction_data["LLM_classes"]
        predicted_confidences = prediction_data["LLM_confidence"]
        
        top_1_pred = predicted_labels[0]
        
        if actual_label == top_1_pred:
            top_1_score = 1.0
            top_1_adjusted_score = 1.0
        else:
            # Get the correlation between the actual and predicted label
            correlation = correlation_map[actual_label].get(top_1_pred, 0.0) if actual_label in correlation_map else 0.0
            # Make sure a correct label is not downscored to a correlated label
            top_1_adjusted_score = max(top_1_adjusted_score, correlation)
        
        # Track top-3 correctness
        if actual_label in predicted_labels:
            top_3_score = 1.0
        
    correct_per_class[actual_label] += top_1_score
    incorrect_per_class[actual_label] += (1.0 - top_1_score)
    adjusted_correct_per_class[actual_label] += top_1_adjusted_score
    correct_per_class_top_3[actual_label] += top_3_score

    logger.info(f"Top 1: {top_1_score}, Adjusted Top 1: {top_1_adjusted_score}, Top 3: {top_3_score}")

    return {
        "actual_label": actual_label,
        "top_1_score": top_1_score,
        "top_3_score": top_3_score,
        "top_1_adjusted_score": top_1_adjusted_score
    }



def load_existing_eval(eval_dataset_json):
    """Load existing evaluation data if the file exists."""
    if os.path.exists(eval_dataset_json):
        with open(eval_dataset_json, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def main(args):
    start_idx = 0
    if args.eval_json:
        eval_dataset_json = args.eval_json
        eval_dict = load_existing_eval(eval_dataset_json)
        start_idx = max(map(int, eval_dict.keys()), default=-1) + 1
    else:
        dataset_name = args.dataset_name
        model_name = args.model_name
        timestr = time.strftime("%Y%m%d-%H%M%S")
        eval_dataset_json = f'logs/{dataset_name}/{model_name}/eval_{timestr}.json'
        os.makedirs(os.path.dirname(eval_dataset_json), exist_ok=True)
        eval_dict = {}

    # Main Loop
    for i, line in enumerate(tqdm(lines)):
        if i < start_idx:
            continue

        if args.class_to_test is not None and not args.class_to_test in line:
            continue

        line_parts = line.strip().split(" ")
        if len(line_parts) == 2:
            video_name, label = line_parts[0], line_parts[1]
        elif len(line_parts) == 4:
            video_name, start_frame, end_frame, label = line_parts
        else:
            continue  # Skip malformed lines

        total_correct = sum(correct_per_class.values())
        total_incorrect = sum(incorrect_per_class.values())
        total_samples = total_correct + total_incorrect

        accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    
        print(f"Video name: {video_name}, label: {label}, acc: {accuracy}")

        if "Normal/" in video_name:
            video_name = video_name.replace("Normal/", "Normal_Videos_event/")

        video_label = int(label)
        video_path = f"{VIDEO_PATH}{video_name}"
        if not ".mp4" in video_path:
            video_path = video_path + ".mp4"

        video_prediction = get_video_prediction(video_path)
        evaluation_result = evaluate_prediction(video_label, video_prediction)

        eval_dict[i] = {
            "video_name": video_name,
            "video_label": video_label,
            "video_prediction": video_prediction,
            "top_1_score": evaluation_result["top_1_score"]
        }

        if i % 10 == 0:
            with open(eval_dataset_json, 'w', encoding='utf-8') as f:
                json.dump(eval_dict, f, indent=2)

    with open(eval_dataset_json, 'w', encoding='utf-8') as f:
        json.dump(eval_dict, f, indent=2)

    # Calculate overall accuracy
    total_correct = sum(correct_per_class.values())
    total_incorrect = sum(incorrect_per_class.values())
    total_samples = total_correct + total_incorrect

    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0

    # Calculate adjusted accuracy
    total_adjusted_correct = sum(adjusted_correct_per_class.values())

    adjusted_accuracy = (total_adjusted_correct / total_samples) * 100 if total_samples > 0 else 0

    # Calculate top-3 accuracy
    total_top3 = sum(correct_per_class_top_3.values())
    top3_accuracy = (total_top3 / total_samples) * 100 if total_samples > 0 else 0

    # log results
    logger.info("\n--- Accuracy Results ---")
    logger.info(f"Overall Accuracy: {accuracy:.2f}%")
    logger.info(f"Adjusted Accuracy: {adjusted_accuracy:.2f}%")
    logger.info(f"Top-3 Accuracy: {top3_accuracy:.2f}%\n")

    # log per-class results
    logger.info("Per-Class Correct Predictions:")
    logger.info(correct_per_class)
    logger.info("\nPer-Class Incorrect Predictions:")
    logger.info(incorrect_per_class)
    logger.info("\nPer-Class Top-3 Predictions:")
    logger.info(correct_per_class_top_3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str, default="UCF_Crime")
    parser.add_argument("-m", "--model_name", type=str, default="NVILA")
    parser.add_argument("-e", "--eval_json", type=str, default=None, help="Path to existing json")
    parser.add_argument("-c", "--class_to_test", type=str, default=None, help="Specific class to test")
    args = parser.parse_args()

    main(args)