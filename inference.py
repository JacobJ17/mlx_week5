import torch
import torchaudio
import numpy as np
from model import AnimalStyleEncoder, AnimalStyleDecoder
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class labels and style means
with open("animal_class_labels.json") as f:
    animal_classes = json.load(f)
style_means = np.load("animal_style_means.npy", allow_pickle=True).item()

# Load models
encoder = AnimalStyleEncoder(
    whisper_model_name='openai/whisper-base',
    n_animal_classes=len(animal_classes),
    style_dim=128,
    unfreeze_layers=0
)
encoder.load_state_dict(torch.load("animal_style_encoder.pt", map_location="cpu"))
encoder.eval().to(device)

# Update decoder loading to include n_classes
decoder = AnimalStyleDecoder(style_dim=128, n_mels=80, n_frames=3000, n_classes=len(animal_classes))
decoder.load_state_dict(torch.load("animal_style_decoder.pt", map_location="cpu"))
decoder.eval().to(device)

def preprocess_audio(audio_path):
    waveform, orig_sr = torchaudio.load(audio_path)
    # Whisper-style Mel spec (for encoder)
    if orig_sr != 16000:
        waveform_16k = torchaudio.transforms.Resample(orig_sr, 16000)(waveform)
    else:
        waveform_16k = waveform
    if waveform_16k.shape[0] > 1:
        waveform_16k = waveform_16k.mean(dim=0, keepdim=True)
    target_length = int(30 * 16000)
    if waveform_16k.shape[1] > target_length:
        waveform_16k = waveform_16k[:, :target_length]
    else:
        waveform_16k = torch.nn.functional.pad(waveform_16k, (0, target_length - waveform_16k.shape[1]))
    whisper_mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_mels=80, n_fft=400, hop_length=160
    )
    whisper_mel = whisper_mel_transform(waveform_16k)
    whisper_mel = torch.log(whisper_mel + 1e-8).squeeze(0)
    whisper_mel = torch.nn.functional.pad(whisper_mel, (0, max(0, 3000 - whisper_mel.shape[1])))[:, :3000]
    return whisper_mel

def animal_translate(audio_path, animal_name):
    mel_spec = preprocess_audio(audio_path).unsqueeze(0).to(device)
    class_idx = torch.tensor([animal_classes.index(animal_name)], device=device)
    with torch.no_grad():
        human_style = encoder.extract_style(mel_spec)
        # Optionally combine with animal style embedding, or just use human_style
        combined_style = human_style
        mel_out = decoder(combined_style, class_idx=class_idx)
    return mel_out.squeeze(0).cpu().numpy()

# Example usage:
# output_mel = animal_translate("your_audio.wav", "dog")
# np.save("output_mel.npy", output_mel)