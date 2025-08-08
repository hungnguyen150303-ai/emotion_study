import torch
from transformers import AutoTokenizer
import torchaudio
from implement import EmoTechModel
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import tempfile
import os

def load_trained_model(model_path='emotech_model.pth'):
    """Load trained model and label encoder"""
    try:
        checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)
    except Exception as e:
        print(f"Error loading with weights_only=False: {e}")
        torch.serialization.add_safe_globals([LabelEncoder])
        checkpoint = torch.load(model_path, map_location='cuda')

    label_encoder = checkpoint['label_encoder']
    num_classes = len(label_encoder.classes_)

    model = EmoTechModel(
        audio_hidden_dim=128,
        text_model_name='bert-base-uncased',
        num_classes=num_classes
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.emotion_labels = list(label_encoder.classes_)
    model.eval()

    return model, label_encoder

def predict_emotion(model, label_encoder, audio_path, text, device='cuda'):
    """Predict emotion from audio and text"""
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Error loading audio with torchaudio: {e}")
        import librosa
        y, sr = librosa.load(audio_path, sr=16000)
        waveform = torch.tensor(y).unsqueeze(0)
        sample_rate = sr

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    target_length = 16000 * 3
    if waveform.size(1) > target_length:
        waveform = waveform[:, :target_length]
    else:
        padding = target_length - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    encoded = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    model = model.to(device)
    audio = waveform.squeeze(0).to(device)
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(audio.unsqueeze(0), input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)

    emotion = label_encoder.inverse_transform([predicted_class.item()])[0]
    confidence = probabilities[0, predicted_class].item()
    emotion_probs = {
        label: probabilities[0, i].item()
        for i, label in enumerate(label_encoder.classes_)
    }

    return emotion, confidence, emotion_probs

def predict_emotion_from_arrays(model, label_encoder, audio_array, text, device='cuda', sample_rate=16000):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    if isinstance(audio_array, np.ndarray):
        waveform = torch.tensor(audio_array, dtype=torch.float32)
    else:
        waveform = audio_array

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    target_length = 16000 * 3
    if waveform.size(1) > target_length:
        waveform = waveform[:, :target_length]
    else:
        padding = target_length - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    encoded = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    model = model.to(device)
    audio = waveform.squeeze(0).to(device)
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(audio.unsqueeze(0), input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)

    emotion = label_encoder.inverse_transform([predicted_class.item()])[0]
    confidence = probabilities[0, predicted_class].item()
    emotion_probs = {
        label: probabilities[0, i].item()
        for i, label in enumerate(label_encoder.classes_)
    }

    return emotion, confidence, emotion_probs

if __name__ == "__main__":
    try:
        model, label_encoder = load_trained_model('emotech_model.pth')
        print("✅ Model loaded successfully!")
        print(f"🧠 Emotion classes: {label_encoder.classes_}")

        df = pd.read_parquet("pho_whisper_transcript.parquet")
        print(f"🗂️ Loaded {len(df)} rows")

        emotions, confidences, probs_list = [], [], []
        correct_count = 0  # Đếm đúng
        total_count = 0    # Tổng dòng hợp lệ

        for idx, row in df.iterrows():
            path_info = row.get("path")
            text = row.get("transcript", "")
            true_emotion = row.get("emotion", "").strip()
            tmp_file_path = None

            try:
                if not isinstance(path_info, dict) or "bytes" not in path_info:
                    raise ValueError("Invalid audio format in row")

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_file.write(bytes(path_info["bytes"]))
                    tmp_file_path = tmp_file.name

                emotion, confidence, emotion_probs = predict_emotion(
                    model, label_encoder, tmp_file_path, text
                )

                is_correct = emotion.strip().lower() == true_emotion.lower()
                result_str = "✅" if is_correct else "❌"

                print(f"[{idx}] {result_str} Pred: {emotion:10} | True: {true_emotion:10} | Conf: {confidence:.2f} | Text: {text[:30]}...")

                emotions.append(emotion)
                confidences.append(confidence)
                probs_list.append(emotion_probs)

                total_count += 1
                if is_correct:
                    correct_count += 1

            except Exception as e:
                print(f"[{idx}] ❌ Error: {e}")
                emotions.append("ERROR")
                confidences.append(0.0)
                probs_list.append({})

            finally:
                if tmp_file_path and os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)

        # Ghi kết quả
        df["predicted_emotion"] = emotions
        df["confidence"] = confidences
        df["emotion_probs"] = probs_list

        #df.to_csv("emotion_results.csv", index=False)
        df.to_parquet("emotion_results.parquet")

        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        print(f"\n📊 Tổng hợp kết quả:")
        print(f"✅ Đúng:   {correct_count}")
        print(f"❌ Sai:    {total_count - correct_count}")
        print(f"🎯 Accuracy: {accuracy:.2f}%")

    except Exception as e:
        print(f"❌ Failed to run inference: {e}")

