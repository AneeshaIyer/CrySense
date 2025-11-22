import os
import torch
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# CONFIGURATION
DATASET_DIR = r"C:\Users\anees\Downloads\archive\Baby Crying Sounds"
SAMPLE_RATE = 16000
BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 0.001
N_CLASSES = 5   # Hungry, Sleepy, Needs Diaper, Pain, Discomfort
MAX_LEN = 500   # Max number of time frames for Mel spectrogram

# CLASS MAPPING
CLASS_NAMES = ["hungry", "sleepy", "diaper", "pain", "discomfort"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# HELPER FUNCTION: PAD/TRUNCATE
def pad_truncate(mel_spec, max_len=MAX_LEN):
    if mel_spec.shape[0] < max_len:
        pad_amount = max_len - mel_spec.shape[0]
        mel_spec = F.pad(mel_spec, (0, 0, 0, pad_amount))
    else:
        mel_spec = mel_spec[:max_len, :]
    return mel_spec

# DATASET CLASS
class CryDataset(Dataset):
    def __init__(self, root_dir):
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset directory NOT found: {root_dir}")

        self.file_paths = []
        self.labels = []

        for folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            if folder not in CLASS_TO_IDX:
                continue  # skip unknown folders/files
            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    self.file_paths.append(os.path.join(folder_path, file))
                    self.labels.append(CLASS_TO_IDX[folder])

        print(f"Loaded {len(self.file_paths)} audio files from {root_dir}")

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_mels=64
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]

        # Load waveform using soundfile
        waveform, sr = sf.read(audio_path)
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)  # [1, N]

        # Resample if needed
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

        # Convert to Mel spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec = mel_spec.squeeze().transpose(0, 1)
        mel_spec = pad_truncate(mel_spec, MAX_LEN)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel_spec, label

# MODEL ARCHITECTURE
class CryModel(nn.Module):
    def __init__(self):
        super(CryModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.fc = nn.Linear(256, N_CLASSES)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.mean(dim=2)
        return self.fc(x)

# TRAINING FUNCTION
def train():
    dataset = CryDataset(DATASET_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CryModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        running_loss = 0
        for mel_spec, labels in dataloader:
            mel_spec, labels = mel_spec.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(mel_spec)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {running_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "cry_model.pth")
    print("Training complete. Model saved as cry_model.pth")

# MAIN
if __name__ == "__main__":
    train()
