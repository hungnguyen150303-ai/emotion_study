import pandas as pd
import whisper
import tempfile
import os

# Load mô hình Whisper
model = whisper.load_model("base")

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

        # ✅ Chỉ định tiếng Việt ở đây
        result = model.transcribe(tmp_file_path, language="vi")
        transcript = result['text']
        print(f"[{idx}] 🇻🇳 Transcript: {transcript[:50]}...")

    except Exception as e:
        transcript = f"ERROR: {e}"
        print(f"[{idx}] ❌ Lỗi xử lý audio: {e}")

    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

        transcripts.append(transcript)

# Gắn transcript vào DataFrame và lưu file
df["transcript"] = transcripts
#df.to_csv("with_transcript_vi.csv", index=False)
df.to_parquet("with_transcript_vi.parquet")

print("📁 Đã lưu transcript tiếng Việt vào 'with_transcript_vi.csv'")

