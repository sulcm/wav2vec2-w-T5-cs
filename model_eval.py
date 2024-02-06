from datasets import load_dataset, Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import evaluate

import json
import re
import os
from datetime import datetime
from itertools import chain

import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--model_name_or_path")
parser.add_argument("--output_dir")
parser.add_argument("--dataset_name")
parser.add_argument("--dataset_config_name")
parser.add_argument("--text_column_name")
parser.add_argument("--audio_column_name")
parser.add_argument("--test_split_name")
parser.add_argument("--metrics")
parser.add_argument("--chars_to_ignore")

test_args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"

model = Wav2Vec2ForCTC.from_pretrained(test_args.model_name_or_path).to(device)
processor = Wav2Vec2Processor.from_pretrained(test_args.model_name_or_path)

sampling_rate = processor.feature_extractor.sampling_rate

eval_metrics = {metric: evaluate.load(metric) for metric in test_args.metrics.strip().split()}

eval_dataset = load_dataset(test_args.dataset_name, test_args.dataset_config_name, split=test_args.test_split_name)
eval_dataset = eval_dataset.cast_column(test_args.audio_column_name, Audio(sampling_rate=sampling_rate))

chars_to_ignore_regex = f'[{"".join(test_args.chars_to_ignore.strip().split())}]'
text_column_name = test_args.text_column_name

def remove_special_characters(batch):
    if chars_to_ignore_regex is not None:
        batch["target_text"] = re.sub(chars_to_ignore_regex, "", batch[text_column_name]).lower()
    else:
        batch["target_text"] = batch[text_column_name].lower()
    return batch

print("Removing special characters and normalizing text...")
eval_dataset = eval_dataset.map(remove_special_characters, remove_columns=[text_column_name])

def map_to_pred(batch):
    inputs = processor(batch["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    batch["pred"] = transcription
    return batch

print("Evaluating model...")
result = eval_dataset.map(map_to_pred, remove_columns=[test_args.audio_column_name])
pred = list(chain.from_iterable(result['pred']))

metrics = {k: v.compute(predictions=pred, references=result['target_text']) for k, v in eval_metrics.items()}
metrics.update({
    "model": test_args.model_name_or_path,
    "dataset": {
        "name": test_args.dataset_name,
        "language": test_args.dataset_config_name,
        "split": test_args.test_split_name
    }
})
print("\nEvaluation results:\n" + json.dumps(metrics, indent=2))

if not os.path.isdir(test_args.output_dir):
    os.makedirs(test_args.output_dir)

with open(test_args.output_dir + datetime.now().strftime("%Y%m%d_%H%M%S") + '.json', "w") as f:
    json.dump(metrics, f)