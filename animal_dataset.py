from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
import pandas as pd

ANIMAL_CLASSES = [
    'dog',
    'chirping_birds',
    'crow',
    'sheep',
    'frog',
    'cow',
    'hen',
    'pig',
    'rooster',
    'cat',
    'crickets',
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

        # Whisper-style Mel spec (for encoder)
        whisper_mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=80, n_fft=400, hop_length=160
        )
        whisper_mel = whisper_mel_transform(waveform)
        whisper_mel = torch.log(whisper_mel + 1e-8).squeeze(0)
        whisper_mel = F.pad(whisper_mel, (0, max(0, 3000 - whisper_mel.shape[1])))[:, :3000]

        # Vocoder-style Mel spec (for decoder target)
        vocoder_mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050, n_mels=80, n_fft=1024, hop_length=256
        )
        vocoder_waveform = torchaudio.functional.resample(waveform, orig_freq=16000, new_freq=22050)
        vocoder_mel = vocoder_mel_transform(vocoder_waveform)
        vocoder_mel = torch.log(vocoder_mel + 1e-8).squeeze(0)
        vocoder_mel = F.pad(vocoder_mel, (0, max(0, 2581 - vocoder_mel.shape[1])))[:, :2581]

        return whisper_mel, vocoder_mel, label, waveform.squeeze(0)