import torch
import re
import unicodedata
import evaluate
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import librosa
import numpy as np

try:
    from num2words import num2words
    NUM2WORDS_AVAILABLE = True
except ImportError:
    NUM2WORDS_AVAILABLE = False

def normalize_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = unicodedata.normalize('NFKC', text)

    text = re.sub(r'[^\w\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

class ASREngine:
    def __init__(self, model_name="openai/whisper-large-v3", device="cuda"):
        self.device = device
        self.torch_dtype = torch.float16 if device == "cuda" else torch.float32

        print(f"Loading ASR model: {model_name}")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.cer_metric = evaluate.load("cer")

    def transcribe(self, audio_path):
        audio, sr = librosa.load(audio_path, sr=16000)
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt").to(self.device)
        input_features = inputs.input_features.to(dtype=self.torch_dtype)

        generated_ids = self.model.generate(input_features, language="french")
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return transcription

    def compute_cer(self, reference, hypothesis):
        ref_norm = normalize_text(reference)
        hyp_norm = normalize_text(hypothesis)

        if not ref_norm:
            return 1.0

        cer = self.cer_metric.compute(predictions=[hyp_norm], references=[ref_norm])
        return cer
