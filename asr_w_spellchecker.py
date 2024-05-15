import os
import torch
import logging
import json
import numpy as np

from torchaudio import load as load_wav_file
from torchaudio.functional import resample
from dataclasses import dataclass, field
from typing import Optional, Union
from pathlib import Path
from tqdm import tqdm

from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    T5Tokenizer,
    T5ForConditionalGeneration,
    HfArgumentParser,
    GenerationConfig
)


@dataclass
class DataArguments:
    """
    Arguments for loading data file or dataset. If loading data from file it must be .wav file.
    """
    data_name_or_path: str = field(
        metadata={"help": "Path to local data file or dataset (via the datasets library). If loading data from local file then it must be .wav file."}
    )
    dataset_config_name: Optional[str] = field(
        default="cs",
        metadata={"help": "The configuration name of the dataset to use via the datasets library (applies only if using datasets library)."}
    )
    split_name: Optional[str] = field(
        default="test",
        metadata={"help": "The name of the dataset split to use via the datasets library (applies only if using datasets library)."}
    )
    audio_column_name: Optional[str] = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data (applies only if using datasets library)."}
    )
    text_column_name: Optional[str] = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the ground truth (applies only if using datasets library)."}
    )

@dataclass
class ST6Arguments:
    """
    Arguments for ST6
    """
    wav2vec2_path: str = field(
        metadata={"help": "Path to Wav2Vec2.0 model."}
    )
    t5_path: Optional[str] = field(
        default="",
        metadata={"help": "Path to T5 model."}
    )
    t5_generation_config: Optional[Union[str, Path]] = field(
        default=None,
        metadata={
            "help": "File path pointing to a custom GenerationConfig json file, to use during prediction."
        },
    )
    use_cuda: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use CUDA (has to be available when checking with `torch.cuda.is_available()`)."},
    )
    logging_level: Optional[str] = field(
        default="INFO",
        metadata={"help": "Set logging level of ST6 class."},
    )
    return_asr_output: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to return Wav2Vec2.0 transcription along side with ST6 output in tuple where `(output, asr_output)`."},
    )
    batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "Speed up inference by increasing `batch_size`."},
    )


class ST6():
    framework = "pt"
    sampling_rate = 16_000

    def __init__(self,
                 wav2vec2_path: str,
                 t5_path: str="",
                 t5_generation_config: dict={"max_new_tokens": 64, "num_beams": 4, "do_sample": True},
                 use_cuda: bool=True,
                 logging_level=logging.INFO
                ) -> None:
        self.logger = self.get_logger(name=__name__, logging_level=logging_level)

        self.w2v2_path = wav2vec2_path
        self.t5_path = t5_path
        self.t5_config = t5_generation_config

        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")
        self.logger.info(f"Using device '{self.device}' with framework '{self.framework}'")

        self.logger.info(f"Loading Wav2Vec2 model from {self.w2v2_path} ...")
        self.w2v2_processor = Wav2Vec2Processor.from_pretrained(self.w2v2_path)
        self.w2v2_model = Wav2Vec2ForCTC.from_pretrained(self.w2v2_path).to(self.device)
        self.logger.info(f"Wav2Vec2.0 was initialized")

        if self.t5_path != "":
            self.logger.info(f"Loading T5 model from {self.t5_path} ...")
            self.t5_tokenizer = T5Tokenizer.from_pretrained(self.t5_path)
            self.t5_model = T5ForConditionalGeneration.from_pretrained(self.t5_path).to(self.device)
            self.prefix = "spell check: "
            self.logger.info(f"T5 was initialized with generation configuration: {self.t5_config}")
        
        self.logger.info("ST6 was successfully initialized")
        pass

    def infer(self, input_audio: list|list[list], return_asr_output: bool=False) -> list:
        inputs = self.w2v2_processor(input_audio, sampling_rate=self.sampling_rate, return_tensors=self.framework, padding=True).to(self.device)
        with torch.no_grad():
            logits = self.w2v2_model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = self.w2v2_processor.batch_decode(pred_ids, skip_special_tokens=True)

        if self.t5_path == "":
            return transcription
        self.logger.debug(f"Wav2Vec2 transcription: {transcription}")

        inputs = self.t5_tokenizer([self.prefix + sentence for sentence in transcription], return_tensors=self.framework, padding=True).to(self.device)
        output_sequences = self.t5_model.generate(**inputs, **self.t5_config)
        corrected_input = self.t5_tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

        return corrected_input if not return_asr_output else (corrected_input, transcription)
    
    def __call__(self, input_audio: list|list[list], return_asr_output: bool=False) -> list:
        return self.infer(input_audio=input_audio, return_asr_output=return_asr_output)
    
    def get_logger(self, name: str=__name__, logging_level=logging.NOTSET) -> logging.Logger:
        logger = logging.getLogger(name)
        formatter = logging.Formatter('%(asctime)s - %(module)s.%(funcName)s - %(levelname)s: %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        try:
            logger.setLevel(logging_level)
        except Exception:
            logger.setLevel(logging.INFO)
            logger.warning(f'No logging level named "{logging_level}" setting to "INFO"')
        return logger


def main():
    parser = HfArgumentParser((ST6Arguments, DataArguments))
    st6_args, data_args = parser.parse_args_into_dataclasses()

    # loads data from .wav file or tries to pull it from huggingface dataset
    print("Loading data ...")
    if os.path.isfile(data_args.data_name_or_path) and data_args.data_name_or_path.endswith(".wav"):
        print(f'Trying to load local file from "{data_args.data_name_or_path}" ...')
        samples, wav_sampling_rate = load_wav_file(data_args.data_name_or_path)
        if wav_sampling_rate != ST6.sampling_rate:
            print(f'Resampling local file "{data_args.data_name_or_path}" from {wav_sampling_rate}Hz to {ST6.sampling_rate}Hz')
            audio_files_array = [resample(samples, orig_freq=wav_sampling_rate, new_freq=ST6.sampling_rate).numpy().flatten()]
        else:
            audio_files_array = [samples.numpy().flatten()]
    else:
        print(f'Trying to load dataset from "{data_args.data_name_or_path}" with name "{data_args.dataset_config_name}" and split {data_args.split_name}')
        try:
            dataset = load_dataset(data_args.data_name_or_path, data_args.dataset_config_name, split=data_args.split_name)
            print(f'Resampling dataset "{data_args.data_name_or_path}" to {ST6.sampling_rate}Hz')
            dataset = dataset.cast_column(data_args.audio_column_name, Audio(sampling_rate=ST6.sampling_rate))
        except Exception as e:
            print(f'There is no dataset "{data_args.data_name_or_path}" or try checking other parameters of `DataArguments` class. Error:\n{str(e)}')
            return 1
        if isinstance(dataset[data_args.audio_column_name][0], dict):
            audio_files_array = [ex["array"] for ex in dataset[data_args.audio_column_name]]
        else:
            audio_files_array = dataset[data_args.audio_column_name]
    print(f"{len(audio_files_array)} audio file/s successfully loaded.")

    # gets generation config for T5 model and validates it
    print(f'Loading generation config for T5 from path "{st6_args.t5_generation_config}" ...')
    if st6_args.t5_generation_config and os.path.isfile(st6_args.t5_generation_config):
        with open(st6_args.t5_generation_config, "r") as f:
            t5_generation_config = json.load(f)
        print("Validating generation config ...")
        generation_config_template = GenerationConfig()
        t5_generation_config = {attr: val for attr, val in t5_generation_config.items() if hasattr(generation_config_template, attr)}
    else:
        print("No such file. Default generation config will be used.")
        t5_generation_config = {}
    print("Generation config for T5 was successfully loaded.")

    print("Initialing ST6 ...")
    st6 = ST6(
        wav2vec2_path=st6_args.wav2vec2_path,
        t5_path=st6_args.t5_path,
        t5_generation_config=t5_generation_config,
        use_cuda=st6_args.use_cuda,
        logging_level=st6_args.logging_level.upper() if isinstance(st6_args.logging_level, str) else st6_args.logging_level
    )

    print(f"\n{'=' * 50}\n")
    prev_batch_start = 0
    predictions = []
    for _ in tqdm(range(int(np.ceil(len(audio_files_array) / st6_args.batch_size))), unit="batch"):
        output = st6(input_audio=audio_files_array[prev_batch_start:prev_batch_start+st6_args.batch_size], return_asr_output=st6_args.return_asr_output)
        predictions.extend(output)
        prev_batch_start += st6_args.batch_size
    predictions_str = '\n'.join([f"{str(i)}: " + str(p) + "\n" for i, p in enumerate(predictions)])
    print(f"Predictions:\n{predictions_str}")
    
    return 0


if __name__ == "__main__":
    main()