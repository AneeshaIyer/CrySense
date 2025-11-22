# test_inference_recursive.py
import os
from inference import predict_audio
from utils import interpret
from generate_report import create_report

TEST_FOLDER = r"C:\Users\anees\Downloads\archive\Baby Crying Sounds"

# Find all .wav files recursively
files = []
for root, _, filenames in os.walk(TEST_FOLDER):
    for f in filenames:
        if f.endswith(".wav"):
            files.append(os.path.join(root, f))

if not files:
    print("⚠️ No WAV files found in:", TEST_FOLDER)
else:
    for i, file_path in enumerate(files, 1):
        try:
            pred, conf, _ = predict_audio(file_path)
            md = interpret(pred, conf)
            pdf = create_report(pred, conf, file_in=os.path.basename(file_path))
            print(f"{i}/{len(files)}: {file_path} => {pred} ({conf*100:.1f}%), PDF: {pdf}")
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
