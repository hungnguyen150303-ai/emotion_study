import os
import io
import tempfile
import torch
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from implement import EmoTechModel, EmoTechTrainer


class LSVSCDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128, target_sample_rate=16000, max_audio_length=3.0):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_sample_rate = target_sample_rate
        self.max_audio_length = max_audio_length

        emotions = [item['emotion'] for item in data]
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(emotions)
        self.num_classes = len(self.label_encoder.classes_)

        print(f"Emotion classes: {self.label_encoder.classes_}")

    def __len__(self):
        return len(self.data)

    def load_audio(self, item):
        try:
            path_info = item.get("path")  # 'path' chứa dict có key 'bytes'

            if isinstance(path_info, dict) and 'bytes' in path_info:
                audio_bytes = bytes(path_info['bytes'])

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_file_path = tmp_file.name

                # Load bằng librosa
                y, sr = librosa.load(tmp_file_path, sr=self.target_sample_rate)
                return torch.tensor(y, dtype=torch.float32), sr
            else:
                raise ValueError("path không chứa audio hợp lệ (dict['bytes'])")

        except Exception as e:
            print(f"❌ Lỗi audio tại item: {e}")
            silence = torch.zeros(int(self.target_sample_rate * self.max_audio_length), dtype=torch.float32)
            return silence, self.target_sample_rate

    def preprocess_audio(self, item):
        waveform, sr = self.load_audio(item)
        if waveform.dim() > 1:
            waveform = torch.mean(waveform, dim=0)

        target_length = int(self.target_sample_rate * self.max_audio_length)
        if waveform.size(0) > target_length:
            waveform = waveform[:target_length]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.size(0)))

        return waveform

    def __getitem__(self, idx):
        item = self.data[idx]
        audio = self.preprocess_audio(item)
        text = str(item.get("transcript") or item.get("transcription") or "")
        encoded = self.tokenizer(text, padding='max_length', truncation=True,
                                 max_length=self.max_length, return_tensors='pt')
        label = self.label_encoder.transform([item['emotion']])[0]

        return {
            "audio": audio,
            "input_ids": encoded['input_ids'].squeeze(0),
            "attention_mask": encoded['attention_mask'].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


def create_dataloaders(parquet_path="pho_whisper_transcript.parquet", batch_size=4, max_length=128):
    print("📥 Đọc dữ liệu từ:", parquet_path)
    df = pd.read_parquet(parquet_path)

    if 'transcript' in df.columns:
        df.rename(columns={"transcript": "transcription"}, inplace=True)

    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['emotion'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['emotion'], random_state=42)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train_set = LSVSCDataset(train_df.to_dict("records"), tokenizer, max_length)
    val_set = LSVSCDataset(val_df.to_dict("records"), tokenizer, max_length)
    test_set = LSVSCDataset(test_df.to_dict("records"), tokenizer, max_length)

    val_set.label_encoder = train_set.label_encoder
    test_set.label_encoder = train_set.label_encoder

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader, test_loader, train_set.label_encoder


def train_model():
    num_epochs = 35 # số lần lặp để train
    learning_rate = 2e-5
    save_path = "emotech_model.pth"
    batch_size = 4

    train_loader, val_loader, test_loader, label_encoder = create_dataloaders(batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmoTechModel(
        audio_hidden_dim=128,
        text_model_name="bert-base-uncased",
        num_classes=len(label_encoder.classes_)
    )
    model.emotion_labels = label_encoder.classes_

    trainer = EmoTechTrainer(model, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\n🎯 Epoch {epoch + 1}/{num_epochs}")
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc="Training"):
            loss = trainer.train_step(
                batch['audio'], batch['input_ids'], batch['attention_mask'], batch['label'], optimizer
            )
            train_loss += loss

        avg_train_loss = train_loss / len(train_loader)
        val_loss, val_acc = trainer.evaluate(val_loader)
        scheduler.step(val_loss)

        print(f"📉 Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_encoder': label_encoder,
            }, save_path)
            print(f"✅ Saved best model at epoch {epoch + 1}")

    print("\n🚀 Đánh giá trên test set")
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"✅ Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    try:
        train_model()
        print("🎉 Huấn luyện hoàn tất!")
    except Exception as e:
        import traceback
        print("❌ Lỗi khi huấn luyện:", e)
        traceback.print_exc()

