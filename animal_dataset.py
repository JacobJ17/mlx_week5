from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
import warnings

# Suppress torchaudio warnings about MPEG headers - these files still load correctly
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

# ESC-50 to combined class mapping
ESC50_CLASS_MAPPING = {
    'hen': 'chicken',
    'rooster': 'chicken',
    'cat': 'cat',
    'cow': 'cow', 
    'dog': 'dog',
    'frog': 'frog',
    'pig': 'pig',
    'chirping_birds': 'bird',
    'crow': 'bird'
}

# Animal-Sound dataset folder to class mapping (lowercase for consistency)
ANIMAL_SOUND_MAPPING = {
    'bear': 'bear',
    'cat': 'cat',
    'chicken': 'chicken',
    'cow': 'cow',
    'dog': 'dog',
    'dolphin': 'dolphin',
    'donkey': 'donkey',
    'elephant': 'elephant',
    'frog': 'frog',
    'horse': 'horse',
    'lion': 'lion',
    'monkey': 'monkey',
    'sheep': 'sheep'
}

# Animal-Sound-Dataset-master (Turkish names) to class mapping
TURKISH_ANIMAL_MAPPING = {
    'aslan': 'lion',           # Aslan = Lion
    'esek': 'donkey',          # Esek = Donkey  
    'inek': 'cow',             # Inek = Cow
    'kedi-part1': 'cat',       # Kedi = Cat
    'kedi-part2': 'cat',       # Kedi = Cat
    'kopek-part1': 'dog',      # Kopek = Dog
    'kopek-part2': 'dog',      # Kopek = Dog
    'koyun': 'sheep',          # Koyun = Sheep
    'kurbaga': 'frog',         # Kurbaga = Frog
    'kus-part1': 'bird',       # Kus = Bird
    'kus-part2': 'bird',       # Kus = Bird
    'maymun': 'monkey',        # Maymun = Monkey
    'tavuk': 'chicken'         # Tavuk = Chicken
}

class AnimalSoundDataset(Dataset):
    """Combined dataset for ESC-50 and Animal-Sound datasets"""
    def __init__(self, data_dir='data', 
                 esc50_meta_csv='ESC-50-master/meta/esc50.csv', 
                 esc50_audio_subdir='ESC-50-master/audio',
                 animal_sound_subdir='animal_sound/Animal-Soundprepros',
                 turkish_animal_subdir='Animal-Sound-Dataset-master',
                 sr=16000, fold=None):
        
        self.data_dir = Path(data_dir)
        self.esc50_meta_csv = self.data_dir / esc50_meta_csv
        self.esc50_audio_dir = self.data_dir / esc50_audio_subdir
        self.animal_sound_dir = self.data_dir / animal_sound_subdir
        self.turkish_animal_dir = self.data_dir / turkish_animal_subdir
        self.sr = sr
        
        # Collect all unique class names
        all_classes = set()
        
        # Add ESC-50 mapped classes
        all_classes.update(ESC50_CLASS_MAPPING.values())
        
        # Add Animal-Sound classes
        all_classes.update(ANIMAL_SOUND_MAPPING.values())
        
        # Add Turkish Animal-Sound classes
        all_classes.update(TURKISH_ANIMAL_MAPPING.values())
        
        # Create sorted class list and mapping
        self.animal_classes = sorted(list(all_classes))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.animal_classes)}
        
        print(f"Combined animal classes: {self.animal_classes}")
        
        # Store audio file paths and labels
        self.audio_files = []
        self.labels = []
        self.sources = []  # Track which dataset each sample comes from
        
        # Load ESC-50 samples
        self._load_esc50_samples(fold)
        
        # Load Animal-Sound samples
        self._load_animal_sound_samples()
        
        # Load Turkish Animal-Sound samples
        self._load_turkish_animal_samples()
        
        print(f"Found {len(self.audio_files)} total audio files across {len(self.animal_classes)} classes")
        for i, animal_class in enumerate(self.animal_classes):
            count = self.labels.count(i)
            esc50_count = sum(1 for j, label in enumerate(self.labels) if label == i and self.sources[j] == 'esc50')
            animal_sound_count = sum(1 for j, label in enumerate(self.labels) if label == i and self.sources[j] == 'animal_sound')
            turkish_count = sum(1 for j, label in enumerate(self.labels) if label == i and self.sources[j] == 'turkish')
            print(f"  {animal_class}: {count} files (ESC-50: {esc50_count}, Animal-Sound: {animal_sound_count}, Turkish: {turkish_count})")

    def find_corrupted_files(self, output_file='corrupted_files.txt'):
        """Scan all files in the dataset and identify corrupted ones"""
        print(f"Scanning {len(self.audio_files)} files for corruption...")
        corrupted_files = []
        warning_files = []
        
        import warnings
        import io
        import sys
        
        for i, file_path in enumerate(self.audio_files):
            # Capture warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                try:
                    # Try to load the full audio file
                    waveform, orig_sr = torchaudio.load(file_path)
                    
                    # Check for warnings (MPEG header issues)
                    mpeg_warnings = [warning for warning in w if "MPEG" in str(warning.message) or "Illegal Audio" in str(warning.message)]
                    if mpeg_warnings:
                        warning_msg = str(mpeg_warnings[0].message)
                        warning_files.append((file_path, self.sources[i], f"MPEG warning: {warning_msg}"))
                    
                    # Check for basic issues
                    if waveform.numel() == 0:
                        corrupted_files.append((file_path, self.sources[i], "Empty audio"))
                    elif torch.isnan(waveform).any():
                        corrupted_files.append((file_path, self.sources[i], "Contains NaN values"))
                    elif torch.isinf(waveform).any():
                        corrupted_files.append((file_path, self.sources[i], "Contains infinite values"))
                    elif orig_sr == 0:
                        corrupted_files.append((file_path, self.sources[i], "Invalid sample rate"))
                        
                except Exception as e:
                    error_msg = str(e)
                    corrupted_files.append((file_path, self.sources[i], f"Loading error: {error_msg}"))
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"Scanned {i + 1}/{len(self.audio_files)} files, found {len(corrupted_files)} corrupted, {len(warning_files)} with warnings")
        
        print(f"\nFound {len(corrupted_files)} corrupted files and {len(warning_files)} files with warnings out of {len(self.audio_files)} total")
        
        # Write results to file
        with open(output_file, 'w') as f:
            f.write(f"Audio Files Report - {len(corrupted_files)} corrupted, {len(warning_files)} with warnings\n")
            f.write("=" * 80 + "\n\n")
            
            # Group by source
            by_source = {'esc50': [], 'animal_sound': [], 'turkish': []}
            
            # Add corrupted files
            for file_path, source, error in corrupted_files:
                by_source[source].append((file_path, error, "CORRUPTED"))
            
            # Add warning files
            for file_path, source, warning in warning_files:
                by_source[source].append((file_path, warning, "WARNING"))
            
            for source, files in by_source.items():
                if files:
                    f.write(f"{source.upper()} Dataset ({len(files)} problematic files):\n")
                    f.write("-" * 50 + "\n")
                    for file_path, issue, issue_type in files:
                        f.write(f"[{issue_type}] {file_path}\n  Issue: {issue}\n\n")
                    f.write("\n")
        
        print(f"Results written to {output_file}")
        return corrupted_files, warning_files

    def remove_corrupted_files(self, corrupted_files_list=None):
        """Remove corrupted files from the dataset"""
        if corrupted_files_list is None:
            print("Finding corrupted files first...")
            corrupted_files_list = self.find_corrupted_files()
        
        corrupted_paths = [str(file_path) for file_path, _, _ in corrupted_files_list]
        
        # Remove corrupted files (in reverse order to maintain indices)
        removed_count = 0
        for i in reversed(range(len(self.audio_files))):
            if str(self.audio_files[i]) in corrupted_paths:
                print(f"Removing: {self.audio_files[i]}")
                self.audio_files.pop(i)
                self.labels.pop(i)
                self.sources.pop(i)
                removed_count += 1
        
        print(f"Removed {removed_count} corrupted files. Dataset now has {len(self.audio_files)} files.")
        return removed_count

    def _load_esc50_samples(self, fold):
        """Load samples from ESC-50 dataset"""
        if not self.esc50_meta_csv.exists():
            print("ESC-50 metadata not found, skipping...")
            return
            
        meta = pd.read_csv(self.esc50_meta_csv)
        
        # Filter for animal classes we want
        animal_meta = meta[meta['category'].isin(ESC50_CLASS_MAPPING.keys())].reset_index(drop=True)
        
        if fold is not None:
            animal_meta = animal_meta[animal_meta['fold'] == fold].reset_index(drop=True)

        for _, row in animal_meta.iterrows():
            wav_path = self.esc50_audio_dir / row['filename']
            if wav_path.exists():
                self.audio_files.append(wav_path)
                mapped_class = ESC50_CLASS_MAPPING[row['category']]
                self.labels.append(self.class_to_idx[mapped_class])
                self.sources.append('esc50')

    def _load_animal_sound_samples(self):
        """Load samples from Animal-Sound dataset"""
        if not self.animal_sound_dir.exists():
            print("Animal-Sound directory not found, skipping...")
            return
            
        for folder in self.animal_sound_dir.iterdir():
            if not folder.is_dir():
                continue
                
            # Map folder name to class name (convert to lowercase for consistency)
            folder_name = folder.name.lower()
            if folder_name in ANIMAL_SOUND_MAPPING:
                class_name = ANIMAL_SOUND_MAPPING[folder_name]
                
                if class_name not in self.class_to_idx:
                    continue
                    
                # Load all audio files from this folder
                for audio_file in folder.glob('*.wav'):
                    self.audio_files.append(audio_file)
                    self.labels.append(self.class_to_idx[class_name])
                    self.sources.append('animal_sound')

    def _load_turkish_animal_samples(self):
        """Load samples from Turkish Animal-Sound dataset"""
        if not self.turkish_animal_dir.exists():
            print("Turkish Animal-Sound directory not found, skipping...")
            return
            
        for folder in self.turkish_animal_dir.iterdir():
            if not folder.is_dir():
                continue
                
            # Map folder name to class name (convert to lowercase for consistency)
            folder_name = folder.name.lower()
            if folder_name in TURKISH_ANIMAL_MAPPING:
                class_name = TURKISH_ANIMAL_MAPPING[folder_name]
                
                if class_name not in self.class_to_idx:
                    continue
                    
                # Load all audio files from this folder
                for audio_file in folder.glob('*.wav'):
                    self.audio_files.append(audio_file)
                    self.labels.append(self.class_to_idx[class_name])
                    self.sources.append('turkish')

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        # Load audio with robust error handling
        try:
            waveform, orig_sr = torchaudio.load(audio_path)
            
            # Check if audio loaded properly
            if waveform.numel() == 0 or torch.isnan(waveform).any() or torch.isinf(waveform).any():
                raise ValueError(f"Invalid audio data in {audio_path}")
                
        except Exception as e:
            print(f"Warning: Skipping corrupted file {audio_path}: {e}")
            # Return a zero tensor if loading fails
            waveform = torch.zeros(1, int(30 * self.sr))
            orig_sr = self.sr

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