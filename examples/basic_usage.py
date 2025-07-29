#!/usr/bin/env python3
"""
NeuroDreamAI Basic Usage Example
Author: Prithviraj Chaudhari

This script demonstrates basic usage of the NeuroDreamAI system.
"""

import sys
import os
import argparse
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from eeg_classifier import EEGEmotionClassifier, EEGEmotionTrainer
from dream_generator import DreamNarrativeGenerator

def generate_sample_eeg(emotion='happy', duration=3.0, sampling_rate=128, num_channels=14):
    """Generate sample EEG data for testing"""
    
    num_samples = int(duration * sampling_rate)
    
    # Base frequencies for different emotions
    emotion_frequencies = {
        'happy': {'alpha': 10, 'beta': 20, 'gamma': 40},
        'sad': {'alpha': 8, 'theta': 6, 'delta': 3},
        'fear': {'beta': 25, 'gamma': 45, 'theta': 7},
        'anger': {'beta': 30, 'gamma': 50, 'alpha': 12},
        'surprise': {'gamma': 35, 'beta': 22, 'alpha': 11},
        'disgust': {'alpha': 9, 'beta': 18, 'theta': 5},
        'neutral': {'alpha': 10, 'beta': 15, 'theta': 6}
    }
    
    freqs = emotion_frequencies.get(emotion, emotion_frequencies['neutral'])
    
    # Generate synthetic EEG
    time = np.linspace(0, duration, num_samples)
    eeg_data = np.zeros((num_channels, num_samples))
    
    for channel in range(num_channels):
        signal = np.zeros(num_samples)
        
        # Add frequency components
        for freq_name, freq_val in freqs.items():
            amplitude = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2*np.pi)
            signal += amplitude * np.sin(2 * np.pi * freq_val * time + phase)
        
        # Add noise
        noise = np.random.normal(0, 0.1, num_samples)
        signal += noise
        
        eeg_data[channel] = signal
    
    return eeg_data

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='NeuroDreamAI Basic Usage Example')
    parser.add_argument('--emotion', default='happy', 
                       choices=['happy', 'sad', 'fear', 'anger', 'surprise', 'disgust', 'neutral'],
                       help='Emotion to simulate')
    parser.add_argument('--eeg_file', help='Path to EEG file (CSV format)')
    parser.add_argument('--output_dir', default='../outputs', help='Output directory')
    parser.add_argument('--use_gpt', action='store_true', help='Use GPT for dream generation')
    
    args = parser.parse_args()
    
    print("🧠 NeuroDreamAI Basic Usage Example")
    print("=" * 50)
    
    # Initialize components
    print("Initializing EEG classifier...")
    classifier = EEGEmotionClassifier()
    trainer = EEGEmotionTrainer(classifier)
    
    print("Initializing dream generator...")
    dream_generator = DreamNarrativeGenerator()
    
    # Load or generate EEG data
    if args.eeg_file:
        print(f"Loading EEG data from: {args.eeg_file}")
        # Load EEG data from file
        try:
            eeg_data = np.loadtxt(args.eeg_file, delimiter=',')
            if eeg_data.shape[0] > eeg_data.shape[1]:
                eeg_data = eeg_data.T  # Transpose if needed
        except Exception as e:
            print(f"Error loading EEG file: {e}")
            print("Generating sample data instead...")
            eeg_data = generate_sample_eeg(args.emotion)
    else:
        print(f"Generating sample EEG data for emotion: {args.emotion}")
        eeg_data = generate_sample_eeg(args.emotion)
    
    print(f"EEG data shape: {eeg_data.shape}")
    
    # Classify emotion
    print("\nClassifying emotion from EEG...")
    try:
        emotion_result = trainer.predict_emotion(eeg_data)
        detected_emotion = emotion_result['emotion']
        confidence = emotion_result['confidence']
        
        print(f"Detected emotion: {detected_emotion}")
        print(f"Confidence: {confidence:.2%}")
        print(f"All probabilities: {emotion_result['all_probabilities']}")
        
    except Exception as e:
        print(f"Error in emotion classification: {e}")
        detected_emotion = args.emotion
        confidence = 0.8
        print(f"Using fallback emotion: {detected_emotion}")
    
    # Generate dream narrative
    print(f"\nGenerating dream narrative for emotion: {detected_emotion}")
    try:
        dream_result = dream_generator.generate_dream_narrative(
            detected_emotion, 
            use_gpt=args.use_gpt
        )
        
        print(f"\nDream Narrative:")
        print("-" * 30)
        print(dream_result['narrative'])
        print("-" * 30)
        print(f"Generation method: {dream_result.get('method', 'unknown')}")
        print(f"Confidence: {dream_result.get('confidence', 0.8):.2%}")
        
    except Exception as e:
        print(f"Error in dream generation: {e}")
        dream_result = {
            'narrative': f"A {detected_emotion} dream filled with vivid imagery and emotions.",
            'emotion': detected_emotion,
            'method': 'fallback',
            'confidence': 0.8
        }
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'timestamp': timestamp,
        'input_emotion': args.emotion,
        'detected_emotion': detected_emotion,
        'confidence': confidence,
        'dream_narrative': dream_result['narrative'],
        'generation_method': dream_result.get('method', 'unknown'),
        'eeg_shape': list(eeg_data.shape)
    }
    
    output_file = os.path.join(args.output_dir, f"dream_result_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    print("\n✅ Basic usage example completed!")
    print("\nNext steps:")
    print("- Try different emotions with --emotion parameter")
    print("- Use real EEG data with --eeg_file parameter")
    print("- Enable GPT generation with --use_gpt flag")
    print("- Check outputs directory for results")

if __name__ == "__main__":
    main()

