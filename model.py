import torch
import torch.nn as nn
from transformers import WhisperModel

class AnimalStyleEncoder(nn.Module):
    """Style encoder that learns animal-specific audio characteristics"""
    def __init__(self, whisper_model_name='openai/whisper-base', 
                 n_animal_classes=4, style_dim=128, unfreeze_layers=2):
        super().__init__()
        
        # Load pre-trained Whisper encoder
        print("Loading Whisper model...")
        self.whisper = WhisperModel.from_pretrained(whisper_model_name)
        self.whisper_dim = self.whisper.config.d_model  # 512 for base
        
        # Freeze all Whisper encoder layers first
        for param in self.whisper.encoder.parameters():
            param.requires_grad = False

        # Unfreeze the last N layers for fine-tuning
        if unfreeze_layers > 0:
            print(f"Unfreezing the last {unfreeze_layers} Whisper encoder layers for fine-tuning.")
            for block in self.whisper.encoder.layers[-unfreeze_layers:]:
                for param in block.parameters():
                    param.requires_grad = True
        else:
            print("All Whisper encoder layers frozen.")

        # Style encoder: learns what makes each animal unique
        self.style_encoder = nn.Sequential(
            nn.Linear(self.whisper_dim, style_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(style_dim * 2, style_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Animal classifier
        self.animal_classifier = nn.Linear(style_dim, n_animal_classes)
        self.style_dim = style_dim
        
    def forward(self, mel_spectrogram):
        """
        Args:
            mel_spectrogram: (batch, n_mels, time)
        Returns:
            style_embedding: (batch, style_dim)
            animal_logits: (batch, n_classes)
        """
        # Whisper expects (batch, n_mels, time)
        encoder_outputs = self.whisper.encoder(mel_spectrogram)
        whisper_features = encoder_outputs.last_hidden_state  # (batch, time, whisper_dim)
        pooled_features = whisper_features.mean(dim=1)        # (batch, whisper_dim)
        style_embedding = self.style_encoder(pooled_features) # (batch, style_dim)
        animal_logits = self.animal_classifier(style_embedding)
        return style_embedding, animal_logits

    def extract_style(self, mel_spectrogram):
        """Get only the style embedding (no classifier)"""
        encoder_outputs = self.whisper.encoder(mel_spectrogram)
        whisper_features = encoder_outputs.last_hidden_state
        pooled_features = whisper_features.mean(dim=1)
        style_embedding = self.style_encoder(pooled_features)
        return style_embedding