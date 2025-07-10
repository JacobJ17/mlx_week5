from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
import pandas as pd

ANIMAL_CLASSES = [
    'dog',
    # 'chirping_birds',
    'crow',
    'sheep',
    # 'frog',
    'cow',
    # 'hen',
    'pig',
    # 'rooster',
    'cat',
    # 'crickets',
]

class AnimalSoundDataset(Dataset):
    """Dataset for ESC-50 animal sounds"""
    def __init__(self, data_dir='data', meta_csv='ESC-50-master/meta/esc50.csv', audio_subdir='ESC-50-master/audio',
                 animal_classes=ANIMAL_CLASSES, sr=16000, n_mels=80, max_length=5.0, fold=None):
        self.data_dir = Path(data_dir)
        self.meta_csv = self.data_dir / meta_csv
        self.audio_dir = self.data_dir / audio_subdir
        self.animal_classes = animal_classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(animal_classes)}
        self.sr = sr
        self.n_mels = n_mels
        self.max_length = max_length

        # Load metadata
        meta = pd.read_csv(self.meta_csv)
        animal_meta = meta[meta['category'].isin(self.animal_classes)].reset_index(drop=True)
        if fold is not None:
            animal_meta = animal_meta[animal_meta['fold'] == fold].reset_index(drop=True)

        # Store audio file paths and new labels
        self.audio_files = []
        self.labels = []

        for _, row in animal_meta.iterrows():
            wav_path = self.audio_dir / row['filename']
            if wav_path.exists():
                self.audio_files.append(wav_path)
                self.labels.append(self.class_to_idx[row['category']])

        print(f"Found {len(self.audio_files)} audio files across {len(self.animal_classes)} classes")
        for i, animal_class in enumerate(self.animal_classes):
            count = self.labels.count(i)
            print(f"  {animal_class}: {count} files")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        # Load audio
        waveform, orig_sr = torchaudio.load(audio_path)

        # Resample if needed
        if orig_sr != self.sr:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sr)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad or trim to fixed length (30 seconds for Whisper)
        target_length = int(30 * self.sr)
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        else:
            padding = target_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))

        # Convert to mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_mels=self.n_mels,
            n_fft=400,
            hop_length=160
        )
        mel_spec = mel_transform(waveform)  # (1, n_mels, time)

        # Log-mel (no normalization)
        mel_spec = torch.log(mel_spec + 1e-8)
        mel_spec = mel_spec.squeeze(0)  # (n_mels, time)

        # Pad or trim mel_spec to 3000 frames (time dimension)
        target_frames = 3000
        if mel_spec.shape[1] > target_frames:
            mel_spec = mel_spec[:, :target_frames]
        else:
            pad_amt = target_frames - mel_spec.shape[1]
            mel_spec = F.pad(mel_spec, (0, pad_amt))

        return mel_spec, label, waveform.squeeze(0)