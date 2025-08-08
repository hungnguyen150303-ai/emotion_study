import pandas as pd
import tempfile
import os
import librosa
import torch
from transformers import pipeline

# Khởi tạo pipeline (chỉ định feature extractor đúng cách)
transcriber = pipeline(
    "automatic-speech-recognition",
    model="vinai/PhoWhisper-base",
    feature_extractor="vinai/PhoWhisper-base",
    tokenizer="vinai/PhoWhisper-base",
    device=0 if torch.cuda.is_available() else -1
)

# Đọc file .parquet
df = pd.read_parquet("train-00000-of-00001.parquet")
print(f"🗂️ Đã load {len(df)} dòng từ file .parquet")

transcripts = []

for idx, row in df.iterrows():
    path_info = row['path']
    tmp_file_path = None

    try:
        if isinstance(path_info, dict) and 'bytes' in path_info:
            audio_bytes = bytes(path_info['bytes'])

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file_path = tmp_file.name
        else:
            print(f"[{idx}] ❌ Dữ liệu audio không hợp lệ")
            transcripts.append("ERROR: invalid audio format")
            continue

        # Load audio thành numpy
        audio, sr = librosa.load(tmp_file_path, sr=16000)

        # 🎧 Dự đoán transcript (KHÔNG truyền sampling_rate)
        result = transcriber(audio)
        transcript = result['text']

        print(f"[{idx}] ✅ Transcript: {transcript[:50]}...")

    except Exception as e:
        transcript = f"ERROR: {e}"
        print(f"[{idx}] ❌ Lỗi xử lý audio: {e}")

    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

        transcripts.append(transcript)

# Gắn transcript vào DataFrame và lưu
df["transcript"] = transcripts
df.to_csv("pho_whisper_transcript.csv", index=False)
df.to_parquet("pho_whisper_transcript.parquet")

print("📁 Đã lưu transcript tiếng Việt vào 'pho_whisper_transcript.csv'")

