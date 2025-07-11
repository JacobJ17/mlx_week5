import torch
from torch.utils.data import DataLoader, random_split
from animal_dataset import AnimalSoundDataset
from model import AnimalStyleEncoder
import wandb
from tqdm import tqdm
import numpy as np
import json

# Hyperparameters
batch_size = 16
num_epochs = 8
learning_rate = 1e-4
val_split = 0.15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load full dataset
full_dataset = AnimalSoundDataset()
val_size = int(len(full_dataset) * val_split)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

model = AnimalStyleEncoder(
    whisper_model_name='openai/whisper-base',
    n_animal_classes=len(full_dataset.animal_classes),
    style_dim=128,
    unfreeze_layers=2
).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

wandb.init(project="animal-style-encoder", 
           entity="jacobj17-imperial-college-london",
        config={
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        }
)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for whisper_mel, vocoder_mel, label, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} [train]"):
        whisper_mel = whisper_mel.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        _, logits = model(whisper_mel)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * whisper_mel.size(0)

        # Accuracy
        preds = logits.argmax(dim=1)
        correct += (preds == label).sum().item()
        total += label.size(0)

        wandb.log({"train_loss": loss.item(), "epoch": epoch+1})

    avg_train_loss = running_loss / len(train_dataset)
    train_acc = correct / total

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for whisper_mel, vocoder_mel, label, _ in tqdm(val_loader, desc=f"Epoch {epoch+1} [val]"):
            whisper_mel = whisper_mel.to(device)
            label = label.to(device)
            _, logits = model(whisper_mel)
            loss = criterion(logits, label)
            val_loss += loss.item() * whisper_mel.size(0)

            preds = logits.argmax(dim=1)
            val_correct += (preds == label).sum().item()
            val_total += label.size(0)

    avg_val_loss = val_loss / len(val_dataset)
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
    wandb.log({
        "avg_train_loss": avg_train_loss,
        "avg_val_loss": avg_val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "epoch": epoch+1
    })

torch.save(model.state_dict(), "animal_style_encoder.pt")
wandb.finish()

print("Training complete!")

model.eval()
style_dict = {cls: [] for cls in full_dataset.animal_classes}


with torch.no_grad():
    for whisper_mel, vocoder_mel, label, _ in tqdm(DataLoader(full_dataset, batch_size=1), desc="Extracting styles"):
        whisper_mel = whisper_mel.to(device)
        style_emb = model.extract_style(whisper_mel).cpu().squeeze(0).numpy()
        class_name = full_dataset.animal_classes[label.item()]
        style_dict[class_name].append(style_emb)

# Compute mean embedding for each class
style_means = {cls: np.mean(np.stack(embs), axis=0) for cls, embs in style_dict.items() if embs}

# Save to disk
np.save("animal_style_means.npy", style_means)
print("Saved style means to animal_style_means.npy")

# Save the animal_classes list (index to class name)
with open("animal_class_labels.json", "w") as f:
    json.dump(full_dataset.animal_classes, f)

# Optionally, save the class_to_idx mapping as well
with open("animal_class_to_idx.json", "w") as f:
    json.dump(full_dataset.class_to_idx, f)