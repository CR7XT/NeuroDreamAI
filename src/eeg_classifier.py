"""
EEG Emotion Classifier for NeuroDreamAI
Author: Prithviraj Chaudhari

This module implements a CNN-LSTM hybrid model for classifying emotions from EEG signals.
The model is designed to work with preprocessed EEG data and output emotion probabilities.

Note: For deployment, this version uses simulated models without PyTorch dependency.
"""

import numpy as np
import os
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using simulated emotion classification.")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class EEGDataset:
    """Custom dataset class for EEG data - fallback version without PyTorch"""
    
    def __init__(self, eeg_data, labels):
        if TORCH_AVAILABLE:
            self.eeg_data = torch.FloatTensor(eeg_data)
            self.labels = torch.LongTensor(labels)
        else:
            self.eeg_data = np.array(eeg_data)
            self.labels = np.array(labels)
    
    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        return self.eeg_data[idx], self.labels[idx]


class EEGEmotionClassifier:
    """
    CNN-LSTM hybrid model for EEG emotion classification
    Fallback version that works with or without PyTorch
    """
    
    def __init__(self, num_channels=14, sequence_length=384, num_emotions=7, 
                 conv_filters=64, lstm_hidden=128, dropout_rate=0.3):
        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.num_emotions = num_emotions
        
        if TORCH_AVAILABLE:
            # Initialize PyTorch model
            super(EEGEmotionClassifier, self).__init__()
            
            # Convolutional layers
            self.conv1 = nn.Conv1d(num_channels, conv_filters, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(conv_filters, conv_filters * 2, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(2)
            self.dropout_conv = nn.Dropout(dropout_rate)
            
            # LSTM layers
            self.lstm = nn.LSTM(conv_filters * 2, lstm_hidden, batch_first=True, 
                               num_layers=2, dropout=dropout_rate)
            
            # Fully connected layers
            self.fc1 = nn.Linear(lstm_hidden, 64)
            self.fc2 = nn.Linear(64, num_emotions)
            self.dropout_fc = nn.Dropout(dropout_rate)
        else:
            # Fallback: simple statistical model
            self.emotion_labels = ['happy', 'sad', 'fear', 'anger', 'surprise', 'disgust', 'neutral']
            print("Using fallback emotion classifier (no PyTorch)")
        
    def forward(self, x):
        if not TORCH_AVAILABLE:
            return self.predict_fallback(x)
            
        # Input shape: (batch_size, num_channels, sequence_length)
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        # Reshape for LSTM: (batch_size, sequence_length, features)
        x = x.transpose(1, 2)
        
        # LSTM layers
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last hidden state
        x = hidden[-1]  # Take the last layer's hidden state
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return F.softmax(x, dim=1)
    
    def predict_fallback(self, eeg_data):
        """Fallback prediction method using statistical features"""
        # Simple statistical approach for demo purposes
        if len(eeg_data.shape) == 3:
            # Batch processing
            batch_size = eeg_data.shape[0]
            predictions = []
            for i in range(batch_size):
                pred = self._single_prediction_fallback(eeg_data[i])
                predictions.append(pred)
            return np.array(predictions)
        else:
            # Single sample
            return self._single_prediction_fallback(eeg_data)
    
    def _single_prediction_fallback(self, single_eeg):
        """Statistical prediction for a single EEG sample"""
        # Calculate basic statistical features
        mean_power = np.mean(single_eeg, axis=1)
        std_power = np.std(single_eeg, axis=1)
        
        # Simple heuristic based on power distribution
        frontal_activity = np.mean(mean_power[:4])  # Frontal channels
        posterior_activity = np.mean(mean_power[-4:])  # Posterior channels
        
        # Emotion prediction based on activity patterns
        if frontal_activity > posterior_activity:
            if std_power.mean() > 0.5:
                emotion_idx = 3  # anger
            else:
                emotion_idx = 0  # happy
        else:
            if std_power.mean() > 0.3:
                emotion_idx = 2  # fear
            else:
                emotion_idx = 1  # sad
        
        # Create probability distribution
        probs = np.ones(7) * 0.1  # Base probability
        probs[emotion_idx] = 0.7  # High probability for predicted emotion
        probs = probs / np.sum(probs)  # Normalize
        
        return probs


class EEGPreprocessor:
    """Preprocessor for EEG data"""
    
    def __init__(self, sampling_rate=128, target_length=384):
        self.sampling_rate = sampling_rate
        self.target_length = target_length
        self.scaler = StandardScaler()
        
    def preprocess(self, eeg_data):
        """
        Preprocess EEG data
        
        Args:
            eeg_data: Raw EEG data (samples, channels, time_points)
            
        Returns:
            Preprocessed EEG data
        """
        # Normalize each channel
        processed_data = []
        
        for sample in eeg_data:
            # Ensure consistent length
            if sample.shape[1] > self.target_length:
                # Truncate
                sample = sample[:, :self.target_length]
            elif sample.shape[1] < self.target_length:
                # Pad with zeros
                padding = self.target_length - sample.shape[1]
                sample = np.pad(sample, ((0, 0), (0, padding)), mode='constant')
            
            # Normalize channels
            sample_normalized = self.scaler.fit_transform(sample.T).T
            processed_data.append(sample_normalized)
        
        return np.array(processed_data)


class EEGEmotionTrainer:
    """Trainer class for the EEG emotion classifier"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device if TORCH_AVAILABLE else 'cpu'
        self.emotion_labels = ['happy', 'sad', 'fear', 'anger', 'surprise', 'disgust', 'neutral']
        
        if TORCH_AVAILABLE:
            self.model = self.model.to(self.device)
    
    def predict_emotion(self, eeg_sample):
        """Predict emotion from a single EEG sample"""
        if not TORCH_AVAILABLE:
            # Use fallback prediction
            probabilities = self.model.predict_fallback(eeg_sample)
            predicted_class = np.argmax(probabilities)
            
            return {
                'emotion': self.emotion_labels[predicted_class],
                'confidence': probabilities[predicted_class],
                'all_probabilities': dict(zip(self.emotion_labels, probabilities))
            }
        
        # PyTorch prediction
        self.model.eval()
        
        with torch.no_grad():
            eeg_tensor = torch.FloatTensor(eeg_sample).unsqueeze(0).to(self.device)
            output = self.model(eeg_tensor)
            probabilities = output.cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
            
        return {
            'emotion': self.emotion_labels[predicted_class],
            'confidence': probabilities[predicted_class],
            'all_probabilities': dict(zip(self.emotion_labels, probabilities))
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        if TORCH_AVAILABLE:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'emotion_labels': self.emotion_labels
            }, filepath)
            print(f"Model saved to {filepath}")
        else:
            print("Model saving not available without PyTorch")
    
    def load_model(self, filepath):
        """Load a trained model"""
        if TORCH_AVAILABLE and os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.emotion_labels = checkpoint['emotion_labels']
            print(f"Model loaded from {filepath}")
        else:
            print("Model loading not available or file not found")


def generate_synthetic_data(num_samples=1000, num_channels=14, sequence_length=384):
    """
    Generate synthetic EEG data for testing purposes
    This simulates the structure of real EEG data
    """
    np.random.seed(42)
    
    # Generate synthetic EEG data
    eeg_data = []
    labels = []
    
    for i in range(num_samples):
        # Create synthetic EEG signal with different patterns for different emotions
        emotion = i % 7  # 7 emotions
        
        # Base signal
        signal = np.random.randn(num_channels, sequence_length) * 0.1
        
        # Add emotion-specific patterns
        if emotion == 0:  # happy - higher alpha waves
            signal += 0.3 * np.sin(2 * np.pi * 10 * np.linspace(0, 3, sequence_length))
        elif emotion == 1:  # sad - lower frequency components
            signal += 0.2 * np.sin(2 * np.pi * 5 * np.linspace(0, 3, sequence_length))
        elif emotion == 2:  # fear - high beta waves
            signal += 0.4 * np.sin(2 * np.pi * 25 * np.linspace(0, 3, sequence_length))
        elif emotion == 3:  # anger - high gamma activity
            signal += 0.3 * np.sin(2 * np.pi * 40 * np.linspace(0, 3, sequence_length))
        elif emotion == 4:  # surprise - sudden spikes
            spike_positions = np.random.choice(sequence_length, 5, replace=False)
            signal[:, spike_positions] += 0.5
        elif emotion == 5:  # disgust - irregular patterns
            signal += 0.2 * np.random.randn(num_channels, sequence_length)
        # neutral - keep base signal
        
        eeg_data.append(signal)
        labels.append(emotion)
    
    return np.array(eeg_data), np.array(labels)


if __name__ == "__main__":
    # Example usage
    print("Generating synthetic EEG data...")
    eeg_data, labels = generate_synthetic_data(num_samples=1000)
    
    print("Preprocessing data...")
    preprocessor = EEGPreprocessor()
    processed_data = preprocessor.preprocess(eeg_data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        processed_data, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Create datasets and dataloaders
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    test_dataset = EEGDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model and trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = EEGEmotionClassifier(num_channels=14, sequence_length=384, num_emotions=7)
    trainer = EEGEmotionTrainer(model, device)
    
    print("Training model...")
    training_history = trainer.train(train_loader, val_loader, num_epochs=20)
    
    print("Evaluating model...")
    accuracy, report, cm = trainer.evaluate(test_loader)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Save the model
    trainer.save_model("EEG_emotion_model.pth")
    
    # Test prediction on a single sample
    test_sample = X_test[0]
    prediction = trainer.predict_emotion(test_sample)
    print(f"\nSample prediction: {prediction}")

