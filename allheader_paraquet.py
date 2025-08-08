import pandas as pd

# Đọc file parquet
df = pd.read_parquet("pho_whisper_transcript.parquet")

# In tên cột
print("Tên các cột:", df.columns.tolist())

# In 5 dòng đầu để kiểm tra
print(df.head(5))

# Xuất toàn bộ ra file CSV
df.to_csv("output.csv", index=False, encoding="utf-8-sig")  # Giữ tiếng Việt rõ ràng
print("✅ Đã xuất dữ liệu ra file output_pho_whisper_transcript.csv")

