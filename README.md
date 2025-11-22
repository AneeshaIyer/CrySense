# 🍼 CrySense AI — Baby Cry Analysis System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

*An AI-powered system that analyzes baby cries to identify their emotional needs*

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Model Architecture](#-model-architecture) • [Dataset](#-dataset)

</div>

---

## 📖 Overview

**CrySense AI** is a deep learning-based system designed to classify baby cries into five emotional categories, helping parents and caregivers understand what their baby needs. Using audio analysis and convolutional neural networks, CrySense provides real-time predictions with confidence scores and actionable recommendations.

### 🎯 Cry Categories

- **🍼 Hungry** - Baby needs feeding
- **😴 Sleepy** - Baby is tired and needs rest
- **👶 Diaper** - Diaper change required
- **⚠️ Pain** - Baby may be experiencing discomfort or pain
- **👕 Discomfort** - Environmental factors (temperature, clothing, etc.)

---

## ✨ Features

- **Deep Learning Model** - CNN-based architecture for audio classification
- **Real-time Analysis** - Process audio files or live recordings
- **Confidence Scoring** - Get probability scores for each prediction
- **PDF Reports** - Auto-generated analysis reports
- **Web Interface** - User-friendly Gradio UI for easy interaction
- **Batch Processing** - Analyze multiple audio files recursively
- **Visual Insights** - Mel spectrogram-based audio feature extraction

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/crysense-ai.git
cd crysense-ai
```

### Step 2: Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install soundfile librosa gradio reportlab numpy
```

### Step 3: Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"
```

---

## 📂 Project Structure

```
crysense-ai/
│
├── train.py                    # Model training script
├── inference.py                # Prediction engine
├── utils.py                    # Recommendation system
├── generate_report.py          # PDF report generator
├── test_inference.py           # Batch testing script
│
├── web/
│   ├── app.py                  # Gradio web interface
│   └── templates/
│       └── index.html          # HTML template
│
├── cry_model.pth               # Trained model weights (generated)
└── README.md                   # This file
```

---

## 🎓 Usage

### 1️⃣ Training the Model

Place your dataset in the following structure:

```
Baby Crying Sounds/
├── hungry/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── sleepy/
├── diaper/
├── pain/
└── discomfort/
```

Update the `DATASET_DIR` in `train.py` and run:

```bash
python train.py
```

**Training Configuration:**
- Sample Rate: 16,000 Hz
- Batch Size: 8
- Epochs: 15
- Learning Rate: 0.001

### 2️⃣ Single Audio Prediction

```python
from inference import predict_audio

prediction, confidence, probabilities = predict_audio("baby_cry.wav")
print(f"Prediction: {prediction} ({confidence*100:.1f}% confident)")
```

### 3️⃣ Batch Testing

Test multiple audio files recursively:

```bash
python test_inference.py
```

This will:
- Scan the dataset directory for all `.wav` files
- Run predictions on each file
- Generate PDF reports for each analysis
- Display results in the console

### 4️⃣ Web Interface

Launch the interactive web app:

```bash
cd web
python app.py
```

Then open your browser to `http://localhost:7860`

**Features:**
- Upload audio files (`.wav`, `.mp3`)
- Record directly from microphone
- View analysis results with recommendations
- Download PDF reports

---

## 🧠 Model Architecture

### Audio Processing Pipeline

1. **Audio Loading** - Load `.wav` files using `soundfile`
2. **Resampling** - Normalize to 16kHz sample rate
3. **Mel Spectrogram** - Convert waveform to 64-band mel spectrogram
4. **Padding/Truncation** - Standardize to 500 time frames
5. **Feature Extraction** - CNN processes spectral features
6. **Classification** - Fully connected layer outputs 5 class probabilities

### Network Architecture

```
Input: Mel Spectrogram (500 x 64)
    ↓
Conv1D (64→128, kernel=5, stride=2) + ReLU
    ↓
Conv1D (128→256, kernel=5, stride=2) + ReLU
    ↓
MaxPool1D (kernel=2)
    ↓
Global Average Pooling
    ↓
Fully Connected (256→5)
    ↓
Softmax → [hungry, sleepy, diaper, pain, discomfort]
```

**Model Highlights:**
- Lightweight CNN design for real-time inference
- Mel spectrogram feature representation
- Global average pooling reduces overfitting
- Adam optimizer with cross-entropy loss

---

## 📊 Dataset

The model is trained on the **Baby Crying Sounds** dataset containing labeled audio samples across five categories.

### Dataset Requirements

- **Format:** `.wav` files
- **Duration:** 2-6 seconds recommended
- **Sample Rate:** 16 kHz (auto-resampled if different)
- **Classes:** 5 balanced categories

### Data Organization

```
DATASET_DIR/
├── hungry/          # Feeding-related cries
├── sleepy/          # Tired/drowsy cries
├── diaper/          # Diaper change needed
├── pain/            # Pain/distress cries
└── discomfort/      # General discomfort
```

---

## 📈 Results & Analysis

### Confidence Levels

- **HIGH** (≥80%): Strong prediction, immediate action recommended
- **MEDIUM** (60-79%): Likely correct, monitor situation
- **LOW** (<60%): Uncertain, consider multiple factors

### Recommendations System

Each prediction comes with actionable guidance:

| Category | Recommendation |
|----------|----------------|
| **Hungry** | 🍼 Offer feeding. Watch for rooting/sucking cues. |
| **Pain** | ⚠️ Check for fever, rash, swelling. Consult pediatrician if persistent. |
| **Sleepy** | 😴 Dim lights, swaddle, reduce stimulation. |
| **Discomfort** | 👕 Check diaper, clothing, room temperature. |
| **Diaper** | 👶 Change diaper and ensure comfort. |

---

## 🔧 Configuration

### Key Parameters (editable in scripts)

```python
SAMPLE_RATE = 16000      # Audio sample rate
BATCH_SIZE = 8           # Training batch size
EPOCHS = 15              # Training epochs
LEARNING_RATE = 0.001    # Optimizer learning rate
N_CLASSES = 5            # Number of cry categories
MAX_LEN = 500            # Mel spectrogram time frames
```

---

## 📝 Report Generation

Automated PDF reports include:
- Predicted cry type with confidence score
- Timestamp of analysis
- Audio file information
- Recommendation guidelines
- System metadata

Example usage:

```python
from generate_report import create_report

pdf_path = create_report(
    prediction="hungry",
    confidence=0.87,
    file_in="baby_cry.wav"
)
```

---

## 🛠️ Development

### Adding New Cry Categories

1. Update `CLASS_NAMES` in `train.py`
2. Increment `N_CLASSES`
3. Add corresponding folder to dataset
4. Update `RECOMMENDATIONS` in `utils.py`
5. Retrain the model

### Improving Model Performance

- **Data Augmentation:** Add noise, time-stretching, pitch shifting
- **Deeper Architecture:** Add more convolutional layers
- **Transfer Learning:** Use pre-trained audio models
- **Ensemble Methods:** Combine multiple model predictions

---

## ⚠️ Important Notes

### Research Prototype

This system is designed for **research and educational purposes**. It should not replace professional medical advice or caregiver judgment.

### Limitations

- Performance depends on audio quality and recording conditions
- Model accuracy varies with background noise levels
- Cultural and individual differences in baby cries may affect results
- Always prioritize direct observation and professional guidance

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

## 🙏 Acknowledgments

- Dataset: Baby Crying Sounds Archive
- Frameworks: PyTorch, Gradio, librosa
- Audio Processing: torchaudio, soundfile

---

## 📧 Contact

For questions, suggestions, or collaboration:

- **GitHub Issues:** [Report bugs or request features](https://github.com/AneeshaIyer/CrySense/issues)
- **Email:** aneeshamiyer@gmail.com -- batulhsuratwala@gmail.com

---

<div align="center">

**Made with ❤️ for parents and caregivers everywhere**

⭐ Star this repo if you find it helpful!

</div>
