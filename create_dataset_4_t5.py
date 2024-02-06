from datasets import load_dataset, Audio, Dataset, DatasetDict
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
parser.add_argument("--split_names")
parser.add_argument("--metrics")
parser.add_argument("--chars_to_ignore")

args = parser.parse_args()

ASR_TRANSCRIPTION = "asr_transcription"
TARGET_OUTPUT = "target_output"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Wav2Vec2ForCTC.from_pretrained(args.model_name_or_path).to(device)
processor = Wav2Vec2Processor.from_pretrained(args.model_name_or_path)

sampling_rate = processor.feature_extractor.sampling_rate

eval_metrics = {metric: evaluate.load(metric) for metric in args.metrics.strip().split()}

chars_to_ignore_regex = f'[{"".join(args.chars_to_ignore.strip().split())}]'
text_column_name = args.text_column_name

def remove_special_characters(batch):
    if chars_to_ignore_regex is not None:
        batch["target_text"] = re.sub(chars_to_ignore_regex, "", batch[text_column_name]).lower()
    else:
        batch["target_text"] = batch[text_column_name].lower()
    return batch

def map_to_pred(batch):
    inputs = processor(batch["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    batch["pred"] = transcription
    return batch

dataset_metrics = {
    "model": args.model_name_or_path,
    "dataset": {
        "name": args.dataset_name,
        "language": args.dataset_config_name,
    }
}
prep_dataset = DatasetDict()
for split in args.split_names.strip().split():
    current_dataset_split = load_dataset(args.dataset_name, args.dataset_config_name, split=split)
    current_dataset_split = current_dataset_split.cast_column(args.audio_column_name, Audio(sampling_rate=sampling_rate))

    print(f"Removing special characters and normalizing text for split {split}...")
    current_dataset_split = current_dataset_split.map(remove_special_characters, remove_columns=[text_column_name])

    print(f"Evaluating model for split {split}...")
    result = current_dataset_split.map(map_to_pred, remove_columns=[args.audio_column_name])
    pred = list(chain.from_iterable(result['pred']))

    prep_dataset[split] = Dataset.from_dict({
        ASR_TRANSCRIPTION: pred,
        TARGET_OUTPUT: result['target_text']
    })

    metrics = {k: v.compute(predictions=pred, references=result['target_text']) for k, v in eval_metrics.items()}
    dataset_metrics['dataset'][split] = metrics
    print(f"\nEvaluation results for {split}:\n" + json.dumps(metrics, indent=2))

if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

with open(os.path.join(args.output_dir, 'asr_transcription_metrics.json'), "w") as f:
    json.dump(dataset_metrics, f)

prep_dataset.save_to_disk(args.output_dir)