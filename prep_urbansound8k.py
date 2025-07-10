import numpy as np
import torch
from datasets import load_dataset
import librosa

def extract_mel(waveform, sr, n_mels=64, hop_length=512):
    mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)

def prepare_urbansound8k_allfolds(n_mels=64, max_len_seconds=4.0, hop_length=512):
    ds = load_dataset(
        path="data/urbansound8K",
        cache_dir="data/urbansound8K/cache",
    )['train']

    features, labels, folds = [], [], []

    for sample in ds:
        waveform = sample['audio']['array']
        sr = sample['audio']['sampling_rate']
        # Truncate/pad waveform to max_len_seconds
        max_wave_len = int(max_len_seconds * sr)
        if len(waveform) < max_wave_len:
            waveform = np.pad(waveform, (0, max_wave_len - len(waveform)))
        else:
            waveform = waveform[:max_wave_len]
        mel = extract_mel(waveform, sr, n_mels=n_mels, hop_length=hop_length)
        features.append(mel)
        labels.append(sample['classID'])
        folds.append(sample['fold'])

    # Find max time dimension
    max_frames = max(m.shape[1] for m in features)
    # Pad all to max_frames
    features = [np.pad(m, ((0,0), (0, max_frames - m.shape[1])), mode='constant') for m in features]

    features = np.stack(features)
    labels = np.array(labels)
    folds = np.array(folds)

    # Add channel dimension for PyTorch CNNs
    features = features[:, None, :, :]

    # Global normalization (mean/std over all features)
    mean = features.mean()
    std = features.std()
    # features = (features - mean) / std

    return features, labels, folds, mean, std

if __name__ == "__main__":
    features, labels, folds, mean, std = prepare_urbansound8k_allfolds()
    print("Features:", features.shape)
    print("Labels:", labels.shape)
    print("Folds:", folds.shape)
    print("Mean:", mean, "Std:", std)

    # Save to disk
    np.savez_compressed(
        "urbansound8k_mel.npz",
        features=features,
        labels=labels,
        folds=folds,
        mean=mean,
        std=std
    )
    print("Saved to urbansound8k_mel.npz")