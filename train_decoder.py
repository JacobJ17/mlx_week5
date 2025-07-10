import torch
from torch.utils.data import DataLoader
from animal_dataset import AnimalSoundDataset
from model import AnimalStyleEncoder, AnimalStyleDecoder, AnimalAutoencoder
from tqdm import tqdm

# Hyperparameters
batch_size = 8
num_epochs = 20
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
full_dataset = AnimalSoundDataset()
train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Load encoder
encoder = AnimalStyleEncoder(
    whisper_model_name='openai/whisper-base',
    n_animal_classes=len(full_dataset.animal_classes),
    style_dim=128,
    unfreeze_layers=0  # All frozen!
)
encoder.load_state_dict(torch.load("animal_style_encoder.pt", map_location="cpu"))
encoder.eval()
encoder.to(device)

# Decoder with class conditioning - now uses correct n_frames=2581
decoder = AnimalStyleDecoder(
    style_dim=128,
    n_mels=80,
    n_frames=2581,  # Changed from 3000 to 2581
    n_classes=len(full_dataset.animal_classes)
).to(device)

optimizer = torch.optim.AdamW(decoder.parameters(), lr=learning_rate)
criterion = torch.nn.L1Loss()

for epoch in range(num_epochs):
    decoder.train()
    running_loss = 0.0
    for whisper_mel, vocoder_mel, label, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} [train]"):
        whisper_mel = whisper_mel.to(device)
        vocoder_mel = vocoder_mel.to(device)
        label = label.to(device)
        
        with torch.no_grad():
            style_emb = encoder.extract_style(whisper_mel)
        
        optimizer.zero_grad()
        mel_recon = decoder(style_emb, class_idx=label)
        loss = criterion(mel_recon, vocoder_mel)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * whisper_mel.size(0)
    
    avg_loss = running_loss / len(full_dataset)
    print(f"Epoch {epoch+1}/{num_epochs} - Recon Loss: {avg_loss:.4f}")

torch.save(decoder.state_dict(), "animal_style_decoder.pt")
print("Decoder training complete!")