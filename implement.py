import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import torchaudio.transforms as T

class MFCCExtractor(nn.Module):
    def __init__(self, sample_rate=16000, n_mfcc=13):
        super(MFCCExtractor, self).__init__()
        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
        )
    
    def forward(self, audio):
        return self.mfcc_transform(audio)

class AudioBlock(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=128, num_layers=2):
        super(AudioBlock, self).__init__()
        
        # MFCC Extractor
        self.mfcc = MFCCExtractor()
        
        # Recurrent Network (LSTM)
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Conv2D Network
        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((1, 1))  # Pool to fixed size
        )
        
        # Flatten and Dense layers for conv branch
        self.conv_dense = nn.Linear(64, hidden_dim * 2)  # Match RNN output size
        self.final_dense = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, audio):
        # Extract MFCC features
        mfcc_features = self.mfcc(audio)  # [batch, n_mfcc, time]
        
        # RNN branch
        rnn_input = mfcc_features.transpose(1, 2)  # [batch, time, n_mfcc]
        rnn_out, _ = self.rnn(rnn_input)
        rnn_out = rnn_out[:, -1, :]  # Take last output [batch, hidden_dim * 2]
        
        # Conv2D branch
        conv_input = mfcc_features.unsqueeze(1)  # [batch, 1, n_mfcc, time]
        conv_out = self.conv2d(conv_input)  # [batch, 64, 1, 1]
        conv_out = conv_out.view(conv_out.size(0), -1)  # [batch, 64]
        conv_out = self.conv_dense(conv_out)  # [batch, hidden_dim * 2]
        
        # Combine and process
        combined = rnn_out + conv_out  # Both are [batch, hidden_dim * 2]
        output = self.final_dense(combined)  # [batch, hidden_dim]
        
        return output

class TextBlock(nn.Module):
    def __init__(self, model_name='bert-base-uncased', hidden_dim=128):
        super(TextBlock, self).__init__()
        
        # Text Embedding (using BERT)
        self.embedding = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Recurrent Network
        self.rnn = nn.LSTM(
            input_size=self.embedding.config.hidden_size,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Conv1D Network
        self.conv1d = nn.Sequential(
            nn.Conv1d(self.embedding.config.hidden_size, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Global Max Pooling
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, input_ids, attention_mask):
        # Text embedding
        embeddings = self.embedding(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = embeddings.last_hidden_state
        
        # RNN branch
        rnn_out, _ = self.rnn(sequence_output)
        rnn_pooled = self.global_max_pool(rnn_out.transpose(1, 2)).squeeze(-1)
        
        # Conv1D branch
        conv_input = sequence_output.transpose(1, 2)  # [batch, hidden, seq_len]
        conv_out = self.conv1d(conv_input).squeeze(-1)
        
        # Combine outputs
        combined = torch.cat([rnn_pooled, conv_out], dim=1)
        
        return combined

class ClassificationBlock(nn.Module):
    def __init__(self, input_dim, num_classes=5):
        super(ClassificationBlock, self).__init__()
        
        self.dense_layers = nn.Sequential(
            # Dense Layer 1 (256 units)
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Dense Layer 2 (128 units)
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Dense Layer 3 (64 units)
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Output Layer
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.dense_layers(x)

class EmoTechModel(nn.Module):
    def __init__(self, audio_hidden_dim=128, text_model_name='bert-base-uncased', num_classes=5):
        super(EmoTechModel, self).__init__()
        
        # Initialize blocks
        self.audio_block = AudioBlock(hidden_dim=audio_hidden_dim)
        self.text_block = TextBlock(model_name=text_model_name, hidden_dim=audio_hidden_dim)
        
        # Calculate concatenated dimension
        text_dim = 256 + 256  # RNN output + Conv1D output from text block
        concat_dim = audio_hidden_dim + text_dim
        
        self.classification_block = ClassificationBlock(concat_dim, num_classes)
        
        # Emotion labels
        self.emotion_labels = ['Anger', 'Sad', 'Happy', 'Excited', 'Neutral']
        
    def forward(self, audio, input_ids, attention_mask):
        # Process audio
        audio_features = self.audio_block(audio)
        
        # Process text
        text_features = self.text_block(input_ids, attention_mask)
        
        # Concatenate features
        combined_features = torch.cat([audio_features, text_features], dim=1)
        
        # Classification
        logits = self.classification_block(combined_features)
        
        return logits
    
    def predict_emotion(self, audio, input_ids, attention_mask):
        self.eval()
        with torch.no_grad():
            logits = self.forward(audio, input_ids, attention_mask)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            
        return self.emotion_labels[predicted_class.item()], probabilities.cpu().numpy()

# Training utilities
class EmoTechTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
    def train_step(self, audio, input_ids, attention_mask, labels, optimizer):
        self.model.train()
        
        audio = audio.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)
        
        optimizer.zero_grad()
        
        logits = self.model(audio, input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                audio = batch['audio'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(audio, input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return total_loss / len(dataloader), correct / total


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = EmoTechModel(
        audio_hidden_dim=128,
        text_model_name='bert-base-uncased',
        num_classes=5
    )
    
    # Example forward pass
    batch_size = 4
    audio_length = 16000 * 3  # 3 seconds of audio at 16kHz
    seq_length = 128
    
    # Dummy data
    audio = torch.randn(batch_size, audio_length)
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    # Forward pass
    logits = model(audio, input_ids, attention_mask)
    print(f"Output shape: {logits.shape}")
    print(f"Emotion predictions: {F.softmax(logits, dim=1)}")