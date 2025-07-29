# NeuroDreamAI API Reference

**Author:** Prithviraj Chaudhari  
**Email:** pruthviraj8811@gmail.com

## Overview

This document provides a comprehensive API reference for the NeuroDreamAI system components.

## Core Classes

### EEGEmotionClassifier

Main class for EEG-based emotion classification using CNN-LSTM architecture.

```python
from src.eeg_classifier import EEGEmotionClassifier

classifier = EEGEmotionClassifier(
    num_channels=14,
    sequence_length=384,
    num_emotions=7,
    conv_filters=64,
    lstm_hidden=128,
    dropout_rate=0.3
)
```

#### Parameters

- `num_channels` (int): Number of EEG channels (default: 14)
- `sequence_length` (int): Length of input sequences (default: 384)
- `num_emotions` (int): Number of emotion categories (default: 7)
- `conv_filters` (int): Number of convolutional filters (default: 64)
- `lstm_hidden` (int): LSTM hidden units (default: 128)
- `dropout_rate` (float): Dropout rate for regularization (default: 0.3)

#### Methods

##### `forward(x)`
Forward pass through the network.

**Parameters:**
- `x` (torch.Tensor): Input EEG data tensor

**Returns:**
- `torch.Tensor`: Emotion probability distribution

### EEGEmotionTrainer

Training and inference wrapper for the EEG emotion classifier.

```python
from src.eeg_classifier import EEGEmotionTrainer

trainer = EEGEmotionTrainer(model, device='cpu')
```

#### Parameters

- `model` (EEGEmotionClassifier): The classifier model
- `device` (str): Computing device ('cpu' or 'cuda')

#### Methods

##### `predict_emotion(eeg_sample)`
Predict emotion from EEG data.

**Parameters:**
- `eeg_sample` (numpy.ndarray): EEG data array (channels x samples)

**Returns:**
- `dict`: Prediction results containing:
  - `emotion` (str): Predicted emotion label
  - `confidence` (float): Prediction confidence
  - `all_probabilities` (dict): Probabilities for all emotions

**Example:**
```python
import numpy as np

# Generate sample EEG data
eeg_data = np.random.randn(14, 384)

# Predict emotion
result = trainer.predict_emotion(eeg_data)
print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']:.2%}")
```

##### `save_model(filepath)`
Save the trained model.

**Parameters:**
- `filepath` (str): Path to save the model

##### `load_model(filepath)`
Load a trained model.

**Parameters:**
- `filepath` (str): Path to the saved model

### DreamNarrativeGenerator

Generator for creating dream narratives from detected emotions.

```python
from src.dream_generator import DreamNarrativeGenerator

generator = DreamNarrativeGenerator()
```

#### Methods

##### `generate_dream_narrative(emotion, use_gpt=False)`
Generate a dream narrative for the given emotion.

**Parameters:**
- `emotion` (str): Detected emotion ('happy', 'sad', 'fear', etc.)
- `use_gpt` (bool): Whether to use GPT for generation (default: False)

**Returns:**
- `dict`: Generation results containing:
  - `narrative` (str): Generated dream narrative
  - `emotion` (str): Input emotion
  - `method` (str): Generation method used
  - `confidence` (float): Generation confidence
  - `timestamp` (float): Generation timestamp

**Example:**
```python
# Generate dream narrative
dream = generator.generate_dream_narrative('happy', use_gpt=True)
print(f"Dream: {dream['narrative']}")
print(f"Method: {dream['method']}")
```

### EEGPreprocessor

Preprocessing utilities for EEG signals.

```python
from src.eeg_classifier import EEGPreprocessor

preprocessor = EEGPreprocessor(
    sampling_rate=128,
    lowpass_freq=50,
    highpass_freq=1
)
```

#### Parameters

- `sampling_rate` (int): EEG sampling rate in Hz (default: 128)
- `lowpass_freq` (float): Low-pass filter frequency (default: 50)
- `highpass_freq` (float): High-pass filter frequency (default: 1)

#### Methods

##### `preprocess_eeg(eeg_data)`
Preprocess raw EEG data.

**Parameters:**
- `eeg_data` (numpy.ndarray): Raw EEG data

**Returns:**
- `numpy.ndarray`: Preprocessed EEG data

##### `extract_features(eeg_data)`
Extract frequency domain features.

**Parameters:**
- `eeg_data` (numpy.ndarray): Preprocessed EEG data

**Returns:**
- `numpy.ndarray`: Extracted features

### NeuroDreamAIApp

Main application class for the Gradio web interface.

```python
from src.app import NeuroDreamAIApp

app = NeuroDreamAIApp()
```

#### Methods

##### `create_gradio_interface()`
Create the Gradio web interface.

**Returns:**
- `gradio.Interface`: Configured Gradio interface

##### `process_eeg_data(emotion_selection, eeg_file)`
Process EEG data and detect emotions.

**Parameters:**
- `emotion_selection` (str): Selected emotion for simulation
- `eeg_file` (file): Uploaded EEG file (optional)

**Returns:**
- `tuple`: (emotion_plot, detected_emotion, confidence_text)

##### `generate_dream_narrative(detected_emotion, use_gpt)`
Generate dream narrative from detected emotion.

**Parameters:**
- `detected_emotion` (str): Detected emotion
- `use_gpt` (bool): Whether to use GPT

**Returns:**
- `tuple`: (narrative_text, dream_narrative)

## Utility Functions

### Data Loading

```python
import numpy as np

def load_eeg_csv(filepath):
    """Load EEG data from CSV file"""
    return np.loadtxt(filepath, delimiter=',')

def load_eeg_npy(filepath):
    """Load EEG data from NumPy file"""
    return np.load(filepath)
```

### Synthetic Data Generation

```python
def generate_synthetic_eeg(emotion='happy', duration=3.0, sampling_rate=128, num_channels=14):
    """Generate synthetic EEG data for testing"""
    # Implementation details...
    return eeg_data
```

## Configuration

### Environment Variables

Set these environment variables for full functionality:

```bash
# OpenAI API key for GPT-based generation
export OPENAI_API_KEY="your_key_here"

# EEG processing parameters
export EEG_SAMPLING_RATE=128
export EEG_CHANNELS=14
export EMOTION_CONFIDENCE_THRESHOLD=0.7
```

### Model Configuration

Configure models using JSON files in `models/model_configs/`:

```json
{
  "model_type": "CNN_LSTM",
  "num_channels": 14,
  "sequence_length": 384,
  "num_emotions": 7,
  "conv_filters": [64, 128],
  "lstm_hidden": 128,
  "dropout_rate": 0.3,
  "learning_rate": 0.001
}
```

## Error Handling

All API methods include comprehensive error handling:

```python
try:
    result = trainer.predict_emotion(eeg_data)
except Exception as e:
    print(f"Error in emotion prediction: {e}")
    # Fallback behavior
```

## Performance Considerations

- **Memory Usage**: ~2.3GB during processing
- **Processing Time**: 420-480ms per 3-second EEG window
- **GPU Acceleration**: Automatic when available
- **Batch Processing**: Supported for multiple files

## Examples

### Complete Workflow

```python
import numpy as np
from src.eeg_classifier import EEGEmotionClassifier, EEGEmotionTrainer
from src.dream_generator import DreamNarrativeGenerator

# Initialize components
classifier = EEGEmotionClassifier()
trainer = EEGEmotionTrainer(classifier)
generator = DreamNarrativeGenerator()

# Load EEG data
eeg_data = np.loadtxt('data/sample_eeg/happy_sample.csv', delimiter=',')

# Classify emotion
emotion_result = trainer.predict_emotion(eeg_data)
detected_emotion = emotion_result['emotion']

# Generate dream
dream_result = generator.generate_dream_narrative(detected_emotion)

print(f"Detected: {detected_emotion}")
print(f"Dream: {dream_result['narrative']}")
```

### Real-time Processing

```python
# Real-time EEG processing loop
for eeg_chunk in eeg_stream:
    emotion = trainer.predict_emotion(eeg_chunk)
    dream = generator.generate_dream_narrative(emotion['emotion'])
    # Process dream narrative...
```

## Support

For API support and questions:
- **Email**: pruthviraj8811@gmail.com
- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` directory

