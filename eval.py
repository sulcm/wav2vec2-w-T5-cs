from datasets import load_dataset, Audio, concatenate_datasets
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import evaluate
import json
from datetime import datetime
from itertools import chain


with open("./eval_config.json", "r") as f:
    config = json.load(f)

eval_metrics = {metric: evaluate.load(metric) for metric in ['wer', 'cer']}

eval_dataset_loaded = load_dataset(config['dataset']['path'], config['dataset']['name'], split=config['dataset']['split'])
if "train" in eval_dataset_loaded.keys():
    eval_dataset = concatenate_datasets([eval_dataset_loaded[split] for split in eval_dataset_loaded])
else:
    eval_dataset = eval_dataset_loaded
eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=config['dataset']['sampling_rate']))

model = Wav2Vec2ForCTC.from_pretrained(config['model_path']).to("cuda")
processor = Wav2Vec2Processor.from_pretrained(config['model_path'])

def map_to_pred(batch):
    inputs = processor(batch["audio"]["array"], sampling_rate=config['dataset']['sampling_rate'], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    batch["pred"] = transcription
    return batch

result = eval_dataset.map(map_to_pred, remove_columns=["audio"])
pred = list(chain.from_iterable(result['pred']))
metrics = {k: v.compute(predictions=pred, references=result['transcription']) for k, v in eval_metrics.items()}

with open(config['save_path'] + datetime.now().strftime("%Y%m%d_%H%M%S") + '.json', "w") as f:
    json.dump(metrics, f)