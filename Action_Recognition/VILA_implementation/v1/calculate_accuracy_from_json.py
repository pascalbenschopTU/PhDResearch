import json
import os
import argparse
from sklearn.metrics import roc_auc_score
from transformers import pipeline
from tqdm import tqdm

# Load the zero-shot-classification pipeline
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
# tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
# classifier_func = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")


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

def calculate_accuracy_from_json(args):
    json_filepath = args.json_filepath
    if not os.path.exists(json_filepath):
        print(f"File not found: {json_filepath}")
        return
    
    with open(json_filepath, 'r', encoding='utf-8') as f:
        eval_dict = json.load(f)
    
    correct_per_class = {}
    incorrect_per_class = {}
    adjusted_correct_per_class = {}
    correct_per_class_top_3 = {}

    y_true = []  # Ground truth for VAD
    y_scores = []  # Predicted anomaly scores

    pbar = tqdm(eval_dict.values())
    
    for entry in pbar:
        actual_label = entry["video_label"]
        actual_label = label_map[actual_label]
        top_1_score = entry["top_1_score"]
        video_prediction = entry["video_prediction"]
        
        if actual_label not in correct_per_class:
            correct_per_class[actual_label] = 0
            incorrect_per_class[actual_label] = 0
            adjusted_correct_per_class[actual_label] = 0.0
            correct_per_class_top_3[actual_label] = 0
        
        correct_per_class[actual_label] += top_1_score
        incorrect_per_class[actual_label] += (1 - top_1_score)
        
        top_1_adjusted_score = top_1_score
        top_3_score = 0.0
        anomaly_scores = []
        for prediction in video_prediction.values():
            if args.recalculate:
                LLM_output = prediction["LLM_output"]
                contains_label = any(label.lower() in LLM_output.lower() for label in label_map.values())
                question = (
                    "Does the following text describe an anomaly? If yes, select the most relevant label. "
                    "If not, select 'Normal'.\n\n"
                    f"Text: {LLM_output}\n"
                )

                if not contains_label:
                    question += "This text is probably Normal as there is no label in the text."

                prediction = classifier_func(LLM_output + question, candidate_labels=list(label_map.values()))

                predicted_labels = prediction["labels"][:3]
                predicted_scores = prediction["scores"][:3]
                # print(f"Question: {question} \n\n LLM output: {LLM_output} \n\n prediction: {predicted_labels} \n\n")
            else:
                predicted_labels = prediction["LLM_classes"]
            top_1_pred = predicted_labels[0] if predicted_labels else None
            
            if actual_label in predicted_labels:
                top_3_score = 1.0
            
            if top_1_pred:
                correlation = correlation_map.get(actual_label, {}).get(top_1_pred, 0.0)
                top_1_adjusted_score = max(top_1_adjusted_score, correlation)

            if top_1_pred != "Normal":
                # y_score += 1.0
                anomaly_scores.append(1.0)
            else:
                anomaly_scores.append(0.0)

        # y_score /= len(video_prediction.values())
        y_score = anomaly_scores
        
        adjusted_correct_per_class[actual_label] += top_1_adjusted_score
        correct_per_class_top_3[actual_label] += top_3_score

        # VAD AUC Calculation - Treat "Normal" as non-anomalous, others as anomalous
        # y_true.append(0 if actual_label == "Normal" else 1)
        anomaly_label = 0 if actual_label == "Normal" else 1
        y_true.extend([anomaly_label for _ in range(len(y_score))])
        # y_scores.append(y_score)  # Use 1 - top_1_score as an anomaly score
        y_scores.extend(y_score)

        pbar.set_description(f"y: {y_score}, 1: {top_1_score}, 1a: {top_1_adjusted_score}")

    import numpy as np
    # y_scores = np.array(y_scores).flatten()
    print(np.array(y_true).shape, np.array(y_scores).shape)

    auc_score = roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else 0.0
    
    total_correct = sum(correct_per_class.values())
    total_incorrect = sum(incorrect_per_class.values())
    total_samples = total_correct + total_incorrect
    total_adjusted_correct = sum(adjusted_correct_per_class.values())
    
    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    adjusted_accuracy = (total_adjusted_correct / total_samples) * 100 if total_samples > 0 else 0
    total_top3 = sum(correct_per_class_top_3.values())
    top3_accuracy = (total_top3 / total_samples) * 100 if total_samples > 0 else 0
    
    print("\n--- Accuracy Results ---")
    print(f"Top-1 Accuracy: {accuracy:.2f}%")
    print(f"Adjusted Accuracy: {adjusted_accuracy:.2f}%")
    print(f"Top-3 Accuracy: {top3_accuracy:.2f}%")
    print(f"VAD AUC Score: {auc_score:.4f}\n")

    # Convert dict_values to lists first
    correct_array = np.array(list(correct_per_class.values()))
    incorrect_array = np.array(list(incorrect_per_class.values()))

    # Compute accuracy per class
    accuracy = correct_array / (correct_array + incorrect_array)
    
    print("Per-Class Accuracy:")
    print(accuracy)
    print("\nPer-Class Top-3 Predictions:")
    print(correct_per_class_top_3)
    print("\nPer-Class Adjusted Correct Predictions:")
    print(adjusted_correct_per_class)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate accuracy from a JSON file.")
    parser.add_argument("-j", "--json_filepath", type=str, help="Path to the JSON file containing evaluation results.")
    parser.add_argument("-r", "--recalculate", action='store_true', help="recalculate the class predictions from LLM texts")
    args = parser.parse_args()

    if args.recalculate:
        classifier_func = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    calculate_accuracy_from_json(args)
