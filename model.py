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
            nn.Dropout(0.3),
            nn.Linear(style_dim * 2, style_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
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

class AnimalStyleDecoder(nn.Module):
    """Decoder that reconstructs a Mel spectrogram from a style embedding"""
    def __init__(self, style_dim=128, n_mels=80, n_frames=2581, n_classes=None):  # Changed from 3000 to 2581
        super().__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames
        if n_classes is not None:
            self.class_emb = nn.Embedding(n_classes, style_dim)
            input_dim = style_dim * 2
        else:
            self.class_emb = None
            input_dim = style_dim
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, style_dim * 4),
            nn.ReLU(),
            nn.Linear(style_dim * 4, n_mels * n_frames)
        )

    def forward(self, style_embedding, class_idx=None):
        if self.class_emb is not None and class_idx is not None:
            class_embedding = self.class_emb(class_idx)
            x = torch.cat([style_embedding, class_embedding], dim=-1)
        else:
            x = style_embedding
        x = self.decoder(x)
        x = x.view(-1, self.n_mels, self.n_frames)
        return x

class AnimalAutoencoder(nn.Module):
    """Frozen encoder + trainable decoder for Mel reconstruction"""
    def __init__(self, encoder: AnimalStyleEncoder, decoder: AnimalStyleDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, mel_spectrogram):
        with torch.no_grad():
            style_embedding = self.encoder.extract_style(mel_spectrogram)
        mel_recon = self.decoder(style_embedding)
        return mel_recon