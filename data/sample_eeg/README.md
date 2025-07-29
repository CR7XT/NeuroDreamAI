# Sample EEG Data

This directory contains sample EEG datasets for testing and demonstration purposes.

## File Formats Supported

- **CSV**: Comma-separated values (channels x samples)
- **NPY**: NumPy binary format
- **EDF**: European Data Format (requires MNE-Python)

## Sample Data Structure

EEG data should be organized as:
- **Rows**: EEG channels (typically 14-32 channels)
- **Columns**: Time samples (typically 128-512 Hz sampling rate)

## Example Files

- `happy_sample.csv` - Sample EEG data for happy emotion
- `sad_sample.csv` - Sample EEG data for sad emotion
- `fear_sample.csv` - Sample EEG data for fear emotion

## Generating Test Data

Use the `examples/basic_usage.py` script to generate synthetic EEG data:

```bash
python examples/basic_usage.py --emotion happy
```

