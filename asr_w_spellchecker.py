import torch
import logging

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    T5Tokenizer,
    T5ForConditionalGeneration
)


class ST6():
    framework = "pt"

    def __init__(self, wav2vec2_path: str, t5_path: str="", use_cuda: bool=True) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(module)s.%(funcName)s - %(levelname)s: %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.w2v2_path = wav2vec2_path
        self.t5_path = t5_path

        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")
        self.logger.info(f"Using device '{self.device}' with framework '{self.framework}'")

        self.logger.info(f"Loading Wav2Vec2 model from {self.w2v2_path} ...")
        self.w2v2_processor = Wav2Vec2Processor.from_pretrained(self.w2v2_path)
        self.w2v2_model = Wav2Vec2ForCTC.from_pretrained(self.w2v2_path)
        self.w2v2_model.to(self.device)
        self.sampling_rate = self.w2v2_processor.feature_extractor.sampling_rate

        if self.t5_path != "":
            self.logger.info(f"Loading T5 model from {self.t5_path} ...")
            self.t5_tokenizer = T5Tokenizer.from_pretrained(self.t5_path)
            self.t5_model = T5ForConditionalGeneration.from_pretrained(self.t5_path)
            self.t5_model.to(self.device)
            self.prefix = "spell check: "
            self.max_new_tokens = 64
        
        self.logger.info("ST6 was successfully initialized")
        pass

    def forward(self, input_audio: list|list[list]) -> list:
        inputs = self.w2v2_processor(input_audio, sampling_rate=self.sampling_rate, return_tensors=self.framework).to(self.device)
        with torch.no_grad():
            logits = self.w2v2_model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = self.w2v2_processor.batch_decode(pred_ids)

        if self.t5_path == "":
            return transcription
        
        self.logger.debug(f"Wav2Vec2 transcription: {transcription}")

        inputs = self.t5_tokenizer([self.prefix + sentence for sentence in transcription], return_tensors=self.framework).to(self.device)
        output_sequences = self.t5_model.generate(**inputs, max_new_tokens=self.max_new_tokens, num_beams=8, do_sample=True)
        corrected_input = self.t5_tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

        return corrected_input
    
    def __call__(self, input_audio: list|list[list]) -> list:
        return self.forward(input_audio=input_audio)