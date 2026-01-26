"""
Model Cache - Singleton pattern to avoid repeated model loading.

This module provides a centralized cache for expensive models (SONAR, Whisper, ZipVoice)
to eliminate the 2-4 minute overhead per sentence from repeated loading.

Usage:
    from model.model_cache import ModelCache
    
    cache = ModelCache()
    text_encoder = cache.get_text_encoder()
    whisper_model = cache.get_whisper_model()
"""

import torch
import os
import sys
from pathlib import Path


class ModelCache:
    """Singleton cache for expensive ML models."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize model cache (only once)."""
        if ModelCache._initialized:
            return
        
        self._text_encoder = None
        self._whisper_model = None
        self._whisper_processor = None
        self._zipvoice_model = None
        
        ModelCache._initialized = True
        print("[ModelCache] Initialized singleton instance")
    
    def get_text_encoder(self, device='cuda'):
        """Get cached SONAR text encoder or load if not cached.
        
        Args:
            device (str): Device to load model on ('cuda' or 'cpu').
            
        Returns:
            TextEncoder: SONAR text encoder instance.
        """
        if self._text_encoder is None:
            print("[ModelCache] Loading SONAR text encoder...")
            sys.path.append(str(Path(__file__).parent))
            from text_encoder import TextEncoder
            
            self._text_encoder = TextEncoder(device=device)
            print(f"[ModelCache] SONAR loaded on {device}")
        
        return self._text_encoder
    
    def get_whisper_model(self, model_size='large-v3', device='cuda'):
        """Get cached Whisper ASR model or load if not cached.
        
        Args:
            model_size (str): Whisper model size (default: 'large-v3').
            device (str): Device to load model on ('cuda' or 'cpu').
            
        Returns:
            tuple: (whisper_model, whisper_processor)
        """
        if self._whisper_model is None or self._whisper_processor is None:
            print(f"[ModelCache] Loading Whisper {model_size}...")
            
            try:
                from transformers import WhisperProcessor, WhisperForConditionalGeneration
                
                model_name = f"openai/whisper-{model_size}"
                self._whisper_processor = WhisperProcessor.from_pretrained(model_name)
                self._whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name)
                self._whisper_model = self._whisper_model.to(device)
                self._whisper_model.eval()
                
                print(f"[ModelCache] Whisper {model_size} loaded on {device}")
            
            except Exception as e:
                print(f"[ModelCache] Failed to load Whisper: {e}")
                raise
        
        return self._whisper_model, self._whisper_processor
    
    def get_zipvoice_model(self, checkpoint_path=None, device='cuda'):
        """Get cached ZipVoice model or load if not cached.
        
        Args:
            checkpoint_path (str): Path to ZipVoice checkpoint (optional).
            device (str): Device to load model on ('cuda' or 'cpu').
            
        Returns:
            ZipVoice model instance.
        """
        if self._zipvoice_model is None:
            print("[ModelCache] Loading ZipVoice...")
            
            # Note: Adjust based on actual ZipVoice loading code
            # The actual implementation would need to import and initialize ZipVoice
            try:
                # Example structure (adapt to actual ZipVoice code):
                # from zipvoice import ZipVoice
                # self._zipvoice_model = ZipVoice.from_pretrained(checkpoint_path)
                # self._zipvoice_model = self._zipvoice_model.to(device)
                # self._zipvoice_model.eval()
                
                print("[ModelCache] ZipVoice loading not implemented")
                print("[ModelCache] Please adapt get_zipvoice_model() to your ZipVoice setup")
                
            except Exception as e:
                print(f"[ModelCache] Failed to load ZipVoice: {e}")
                raise
        
        return self._zipvoice_model
    
    def clear_cache(self):
        """Clear all cached models (useful for testing or memory management)."""
        print("[ModelCache] Clearing all cached models...")
        
        self._text_encoder = None
        
        if self._whisper_model is not None:
            del self._whisper_model
            self._whisper_model = None
        
        if self._whisper_processor is not None:
            del self._whisper_processor
            self._whisper_processor = None
        
        if self._zipvoice_model is not None:
            del self._zipvoice_model
            self._zipvoice_model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[ModelCache] Cache cleared")
    
    def get_cache_status(self):
        """Get status of all cached models.
        
        Returns:
            dict: Status of each model (loaded=True/False).
        """
        return {
            'text_encoder': self._text_encoder is not None,
            'whisper_model': self._whisper_model is not None,
            'whisper_processor': self._whisper_processor is not None,
            'zipvoice_model': self._zipvoice_model is not None,
        }


# Convenience functions for direct access
def get_text_encoder(device='cuda'):
    """Get cached SONAR text encoder."""
    cache = ModelCache()
    return cache.get_text_encoder(device=device)


def get_whisper_model(model_size='large-v3', device='cuda'):
    """Get cached Whisper model and processor."""
    cache = ModelCache()
    return cache.get_whisper_model(model_size=model_size, device=device)


def get_zipvoice_model(checkpoint_path=None, device='cuda'):
    """Get cached ZipVoice model."""
    cache = ModelCache()
    return cache.get_zipvoice_model(checkpoint_path=checkpoint_path, device=device)


def clear_model_cache():
    """Clear all cached models."""
    cache = ModelCache()
    cache.clear_cache()


if __name__ == '__main__':
    # Test the cache
    print("Testing ModelCache...")
    
    cache = ModelCache()
    print(f"Initial status: {cache.get_cache_status()}")
    
    # Test SONAR loading
    encoder = cache.get_text_encoder(device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"After SONAR load: {cache.get_cache_status()}")
    
    # Test Whisper loading
    whisper_model, whisper_processor = cache.get_whisper_model(
        model_size='large-v3',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"After Whisper load: {cache.get_cache_status()}")
    
    # Test cache persistence (second instance should reuse)
    cache2 = ModelCache()
    print(f"Second instance status: {cache2.get_cache_status()}")
    assert cache is cache2, "Singleton pattern failed!"
    
    print("\nModelCache test passed!")
