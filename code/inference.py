# inference.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import librosa

#CONFIG
SAMPLE_RATE = 16000
MAX_LEN = 5 * SAMPLE_RATE  # 5 seconds
N_CLASSES = 5
CLASS_NAMES = ["hungry", "sleepy", "diaper", "pain", "discomfort"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cry_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#WAV LOADER
def load_wav(path, target_sr=SAMPLE_RATE, max_len=MAX_LEN):
    waveform, sr = sf.read(path)
    waveform = waveform.astype('float32')
    if waveform.ndim == 1:
        waveform = waveform[None, :]  # shape (1, N)
    # Resample if needed
    if sr != target_sr:
        waveform = librosa.resample(y=waveform[0], orig_sr=sr, target_sr=target_sr)
        waveform = waveform[None, :]
    # Pad or truncate
    if waveform.shape[1] < max_len:
        pad_len = max_len - waveform.shape[1]
        waveform = np.pad(waveform, ((0,0),(0,pad_len)), mode='constant')
    else:
        waveform = waveform[:, :max_len]
    return torch.tensor(waveform)

# MODEL
class CryModel(nn.Module):
    def __init__(self):
        super(CryModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.fc = nn.Linear(256, N_CLASSES)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, channels, seq)
        x = self.cnn(x)
        x = x.mean(dim=2)
        return self.fc(x)

# MEL SPECTOGRAM
def waveform_to_mel(waveform):
    mel = librosa.feature.melspectrogram(
        y=waveform.numpy().squeeze(),
        sr=SAMPLE_RATE,
        n_mels=64
    )
    mel = torch.tensor(mel.T, dtype=torch.float32)  # time x mel
    return mel

#LOAD MODEL
model = CryModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

#PREDICTION FUNCTION
def predict_audio(file_path):
    waveform = load_wav(file_path)
    mel_spec = waveform_to_mel(waveform)
    mel_spec = mel_spec.unsqueeze(0).to(DEVICE)  # batch dim
    with torch.no_grad():
        logits = model(mel_spec)
        probs = F.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)
    pred_class = CLASS_NAMES[idx.item()]
    return pred_class, conf.item(), probs.cpu().numpy()
