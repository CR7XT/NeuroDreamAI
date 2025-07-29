# NeuroDreamAI User Guide

**Author:** Prithviraj Chaudhari  
**Email:** pruthviraj8811@gmail.com

## Table of Contents

1. [Getting Started](#getting-started)
2. [Web Interface Guide](#web-interface-guide)
3. [Command Line Usage](#command-line-usage)
4. [Working with EEG Data](#working-with-eeg-data)
5. [Dream Generation Options](#dream-generation-options)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

## Getting Started

### Quick Setup

1. **Download and extract** the NeuroDreamAI package
2. **Open terminal** in the project directory
3. **Run setup script**:
   ```bash
   python setup.py
   ```
4. **Activate virtual environment**:
   ```bash
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```
5. **Start the application**:
   ```bash
   python src/app.py
   ```
6. **Open browser** to: http://localhost:7860

### System Requirements

- **Python 3.8+** (Python 3.11 recommended)
- **8GB RAM** minimum (16GB recommended)
- **2GB disk space** for installation
- **Internet connection** for API-based features

## Web Interface Guide

### Main Interface

The NeuroDreamAI web interface provides an intuitive way to process EEG data and generate dream visualizations.

#### 1. EEG Data Input

**Option A: Select Emotion**
- Choose from 7 emotion categories
- System generates synthetic EEG data
- Good for testing and demonstration

**Option B: Upload EEG File**
- Support for CSV and NumPy formats
- Real EEG data processing
- Better accuracy for research use

#### 2. Processing Controls

**Emotion Detection Settings:**
- Confidence threshold adjustment
- Channel selection options
- Preprocessing parameters

**Dream Generation Options:**
- Template-based generation (fast, reliable)
- GPT-based generation (creative, requires API key)
- Narrative length control

#### 3. Results Display

**Emotion Analysis:**
- Detected emotion with confidence score
- Probability distribution across all emotions
- EEG signal visualization plots

**Dream Narrative:**
- Generated dream text
- Emotional consistency metrics
- Generation method information

**Video Planning:**
- Text-to-video generation setup
- Platform selection options
- Quality and style settings

### Step-by-Step Walkthrough

#### Basic Usage (5 minutes)

1. **Start the application**
   ```bash
   python src/app.py
   ```

2. **Open web browser** to http://localhost:7860

3. **Select an emotion** from the dropdown (e.g., "Happy")

4. **Click "Process EEG Data"**
   - Wait for emotion detection to complete
   - Review the detected emotion and confidence

5. **Click "Generate Dream Narrative"**
   - Choose template-based for faster results
   - Read the generated dream story

6. **Optional: Plan video generation**
   - Configure video settings
   - Note: Actual video generation requires API keys

#### Advanced Usage (15 minutes)

1. **Prepare EEG data file**
   - Format: CSV with channels as rows, samples as columns
   - Example: 14 channels × 384 samples for 3 seconds at 128Hz

2. **Upload EEG file**
   - Use the file upload component
   - System automatically detects format

3. **Configure processing**
   - Adjust confidence threshold if needed
   - Select specific channels if required

4. **Process and analyze**
   - Review detailed emotion probabilities
   - Examine EEG signal plots

5. **Generate multiple dreams**
   - Try both template and GPT generation
   - Compare results and quality

6. **Export results**
   - Download generated narratives
   - Save analysis plots

## Command Line Usage

### Basic Processing

Process a single EEG file:
```bash
python examples/basic_usage.py --eeg_file data/sample_eeg/happy_sample.csv
```

Generate synthetic data for testing:
```bash
python examples/basic_usage.py --emotion happy --use_gpt
```

### Batch Processing

Process multiple files:
```bash
python examples/batch_processing.py --input_dir data/sample_eeg/ --output_dir outputs/batch/
```

With GPT generation:
```bash
python examples/batch_processing.py --input_dir data/sample_eeg/ --use_gpt --max_files 10
```

### Command Line Options

**basic_usage.py options:**
- `--emotion`: Emotion to simulate (happy, sad, fear, anger, surprise, disgust, neutral)
- `--eeg_file`: Path to EEG file (CSV format)
- `--output_dir`: Output directory (default: ../outputs)
- `--use_gpt`: Enable GPT-based dream generation

**batch_processing.py options:**
- `--input_dir`: Directory containing EEG files
- `--output_dir`: Output directory
- `--file_pattern`: File pattern to match (default: *.csv)
- `--use_gpt`: Enable GPT generation
- `--max_files`: Maximum files to process

## Working with EEG Data

### Supported Formats

**CSV Format (Recommended):**
```
# 14 channels × 384 samples
channel1_sample1, channel1_sample2, ...
channel2_sample1, channel2_sample2, ...
...
```

**NumPy Format:**
```python
# Save EEG data
import numpy as np
eeg_data = np.random.randn(14, 384)  # channels × samples
np.save('eeg_data.npy', eeg_data)
```

**EDF Format (Future):**
- European Data Format support planned
- Requires MNE-Python library

### Data Requirements

**Channel Configuration:**
- **Minimum**: 8 channels
- **Recommended**: 14 channels (10-20 system)
- **Maximum**: 32 channels

**Sampling Rate:**
- **Minimum**: 64 Hz
- **Recommended**: 128 Hz
- **Maximum**: 512 Hz

**Duration:**
- **Minimum**: 1 second
- **Recommended**: 3-5 seconds
- **Maximum**: 30 seconds per segment

### Data Quality Guidelines

**Signal Quality:**
- Low noise levels (SNR > 10 dB)
- Minimal artifacts (eye blinks, muscle activity)
- Stable electrode impedance (< 5 kΩ)

**Preprocessing:**
- High-pass filter: 1 Hz
- Low-pass filter: 50 Hz
- Notch filter: 50/60 Hz (power line)

## Dream Generation Options

### Template-Based Generation

**Advantages:**
- Fast processing (< 1 second)
- Reliable output quality
- No API dependencies
- Consistent emotional themes

**Best for:**
- Quick testing and demos
- Offline usage
- Consistent results
- Educational purposes

**Configuration:**
```python
# Template generation settings
use_gpt = False
narrative_length = "medium"  # short, medium, long
emotional_intensity = "normal"  # subtle, normal, intense
```

### GPT-Based Generation

**Advantages:**
- Creative and varied output
- Natural language quality
- Contextual coherence
- Personalization potential

**Requirements:**
- OpenAI API key
- Internet connection
- API usage credits

**Setup:**
1. Get OpenAI API key from https://platform.openai.com/
2. Set environment variable:
   ```bash
   export OPENAI_API_KEY="your_key_here"
   ```
3. Enable GPT generation in interface

**Configuration:**
```python
# GPT generation settings
model = "gpt-4"  # or "gpt-3.5-turbo"
temperature = 0.8  # creativity level
max_tokens = 150  # narrative length
```

### Comparison

| Feature | Template-Based | GPT-Based |
|---------|----------------|-----------|
| Speed | Very Fast (< 1s) | Moderate (2-5s) |
| Quality | Good | Excellent |
| Creativity | Limited | High |
| Cost | Free | API costs |
| Offline | Yes | No |
| Consistency | High | Variable |

## Troubleshooting

### Common Issues

#### "No module named 'torch'"
**Problem**: PyTorch not installed
**Solution**:
```bash
pip install torch torchvision torchaudio
```

#### "CUDA out of memory"
**Problem**: GPU memory insufficient
**Solution**:
```bash
export CUDA_VISIBLE_DEVICES=""  # Use CPU only
python src/app.py
```

#### "OpenAI API key not found"
**Problem**: API key not configured
**Solution**:
```bash
export OPENAI_API_KEY="your_key_here"
# or create .env file with the key
```

#### "EEG file format not supported"
**Problem**: Unsupported file format
**Solution**:
```python
# Convert to CSV format
import numpy as np
data = np.load('file.npy')  # or other format
np.savetxt('file.csv', data, delimiter=',')
```

#### "Gradio interface not loading"
**Problem**: Port conflict or firewall
**Solution**:
```bash
# Try different port
python src/app.py --port 7861

# Check firewall settings
# Ensure localhost access is allowed
```

### Performance Issues

#### Slow Processing
**Causes:**
- Large EEG files
- CPU-only processing
- Insufficient RAM

**Solutions:**
- Reduce file size or duration
- Enable GPU acceleration
- Close other applications
- Use batch processing for multiple files

#### Memory Errors
**Causes:**
- Large datasets
- Memory leaks
- Insufficient RAM

**Solutions:**
- Process smaller chunks
- Restart application periodically
- Increase virtual memory
- Use 64-bit Python

### Getting Help

**Before asking for help:**
1. Check this troubleshooting section
2. Review error messages carefully
3. Try basic examples first
4. Check system requirements

**When reporting issues:**
- Include error messages
- Describe steps to reproduce
- Mention system specifications
- Attach sample data if possible

**Contact:**
- Email: pruthviraj8811@gmail.com
- Include "NeuroDreamAI" in subject line

## Advanced Usage

### Custom Model Training

Train your own emotion classifier:
```python
from src.eeg_classifier import EEGEmotionTrainer

# Load your dataset
trainer = EEGEmotionTrainer(model)
trainer.load_custom_dataset('path/to/data')
trainer.train(epochs=100)
trainer.save_model('custom_model.pth')
```

### API Integration

Use NeuroDreamAI in your own applications:
```python
from src import EEGEmotionClassifier, DreamNarrativeGenerator

# Initialize components
classifier = EEGEmotionClassifier()
generator = DreamNarrativeGenerator()

# Your application logic
def process_eeg_stream(eeg_data):
    emotion = classifier.predict(eeg_data)
    dream = generator.generate(emotion)
    return dream
```

### Real-time Processing

Set up real-time EEG processing:
```python
import time
from src.app import NeuroDreamAIApp

app = NeuroDreamAIApp()

# Real-time processing loop
while True:
    eeg_chunk = get_eeg_data()  # Your EEG acquisition
    result = app.process_eeg_data(None, eeg_chunk)
    time.sleep(0.1)  # 10 Hz processing rate
```

### Custom Dream Templates

Create your own dream templates:
```python
# Add to data/dream_templates/custom_templates.json
{
  "happy": [
    "I was {action} through a {environment} filled with {objects}...",
    "In my dream, I found myself {action} in a {environment}..."
  ]
}
```

### Video Generation Integration

Integrate with video generation APIs:
```python
from src.video_generator import VideoGenerator

generator = VideoGenerator(platform='openai_sora')
video_url = generator.generate_video(dream_narrative)
```

---

**Need more help?** Contact pruthviraj8811@gmail.com with your questions!

