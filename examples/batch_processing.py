#!/usr/bin/env python3
"""
NeuroDreamAI Batch Processing Example
Author: Prithviraj Chaudhari

This script demonstrates batch processing of multiple EEG files.
"""

import sys
import os
import argparse
import glob
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from eeg_classifier import EEGEmotionClassifier, EEGEmotionTrainer
from dream_generator import DreamNarrativeGenerator

def process_eeg_file(file_path, classifier, trainer, dream_generator, use_gpt=False):
    """Process a single EEG file"""
    
    try:
        # Load EEG data
        if file_path.endswith('.csv'):
            eeg_data = np.loadtxt(file_path, delimiter=',')
        elif file_path.endswith('.npy'):
            eeg_data = np.load(file_path)
        else:
            print(f"Unsupported file format: {file_path}")
            return None
        
        # Ensure correct shape (channels x samples)
        if eeg_data.shape[0] > eeg_data.shape[1]:
            eeg_data = eeg_data.T
        
        # Classify emotion
        emotion_result = trainer.predict_emotion(eeg_data)
        detected_emotion = emotion_result['emotion']
        confidence = emotion_result['confidence']
        
        # Generate dream narrative
        dream_result = dream_generator.generate_dream_narrative(
            detected_emotion, 
            use_gpt=use_gpt
        )
        
        return {
            'file_path': file_path,
            'eeg_shape': list(eeg_data.shape),
            'detected_emotion': detected_emotion,
            'confidence': confidence,
            'all_probabilities': emotion_result['all_probabilities'],
            'dream_narrative': dream_result['narrative'],
            'generation_method': dream_result.get('method', 'unknown'),
            'dream_confidence': dream_result.get('confidence', 0.8),
            'processing_time': None,  # Will be filled by caller
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'file_path': file_path,
            'error': str(e),
            'status': 'error'
        }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='NeuroDreamAI Batch Processing Example')
    parser.add_argument('--input_dir', required=True, help='Directory containing EEG files')
    parser.add_argument('--output_dir', default='../outputs/batch', help='Output directory')
    parser.add_argument('--file_pattern', default='*.csv', help='File pattern to match')
    parser.add_argument('--use_gpt', action='store_true', help='Use GPT for dream generation')
    parser.add_argument('--max_files', type=int, help='Maximum number of files to process')
    
    args = parser.parse_args()
    
    print("🧠 NeuroDreamAI Batch Processing Example")
    print("=" * 50)
    
    # Find EEG files
    input_pattern = os.path.join(args.input_dir, args.file_pattern)
    eeg_files = glob.glob(input_pattern)
    
    if not eeg_files:
        print(f"No files found matching pattern: {input_pattern}")
        return
    
    if args.max_files:
        eeg_files = eeg_files[:args.max_files]
    
    print(f"Found {len(eeg_files)} EEG files to process")
    
    # Initialize components
    print("Initializing NeuroDreamAI components...")
    classifier = EEGEmotionClassifier()
    trainer = EEGEmotionTrainer(classifier)
    dream_generator = DreamNarrativeGenerator()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process files
    results = []
    successful_count = 0
    error_count = 0
    
    print(f"\nProcessing {len(eeg_files)} files...")
    print("-" * 50)
    
    for i, file_path in enumerate(eeg_files, 1):
        print(f"[{i}/{len(eeg_files)}] Processing: {os.path.basename(file_path)}")
        
        start_time = datetime.now()
        
        result = process_eeg_file(
            file_path, 
            classifier, 
            trainer, 
            dream_generator, 
            use_gpt=args.use_gpt
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        if result:
            result['processing_time'] = processing_time
            results.append(result)
            
            if result['status'] == 'success':
                successful_count += 1
                print(f"  ✅ Emotion: {result['detected_emotion']} "
                      f"(confidence: {result['confidence']:.2%})")
                print(f"  📝 Dream: {result['dream_narrative'][:100]}...")
            else:
                error_count += 1
                print(f"  ❌ Error: {result['error']}")
        else:
            error_count += 1
            print(f"  ❌ Failed to process file")
        
        print(f"  ⏱️  Processing time: {processing_time:.2f}s")
        print()
    
    # Save batch results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    batch_summary = {
        'timestamp': timestamp,
        'input_directory': args.input_dir,
        'file_pattern': args.file_pattern,
        'total_files': len(eeg_files),
        'successful_files': successful_count,
        'error_files': error_count,
        'use_gpt': args.use_gpt,
        'results': results
    }
    
    # Save detailed results
    results_file = os.path.join(args.output_dir, f"batch_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(batch_summary, f, indent=2)
    
    # Save summary report
    summary_file = os.path.join(args.output_dir, f"batch_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write("NeuroDreamAI Batch Processing Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Input Directory: {args.input_dir}\n")
        f.write(f"File Pattern: {args.file_pattern}\n")
        f.write(f"Total Files: {len(eeg_files)}\n")
        f.write(f"Successful: {successful_count}\n")
        f.write(f"Errors: {error_count}\n")
        f.write(f"Success Rate: {successful_count/len(eeg_files)*100:.1f}%\n")
        f.write(f"GPT Generation: {'Enabled' if args.use_gpt else 'Disabled'}\n\n")
        
        # Emotion distribution
        emotions = [r['detected_emotion'] for r in results if r['status'] == 'success']
        if emotions:
            from collections import Counter
            emotion_counts = Counter(emotions)
            f.write("Emotion Distribution:\n")
            for emotion, count in emotion_counts.most_common():
                f.write(f"  {emotion}: {count} ({count/len(emotions)*100:.1f}%)\n")
        
        f.write(f"\nDetailed results saved to: {results_file}\n")
    
    # Print final summary
    print("=" * 50)
    print("📊 Batch Processing Summary")
    print("=" * 50)
    print(f"Total files processed: {len(eeg_files)}")
    print(f"Successful: {successful_count}")
    print(f"Errors: {error_count}")
    print(f"Success rate: {successful_count/len(eeg_files)*100:.1f}%")
    
    if successful_count > 0:
        avg_time = sum(r['processing_time'] for r in results if r['status'] == 'success') / successful_count
        print(f"Average processing time: {avg_time:.2f}s per file")
        
        # Show emotion distribution
        emotions = [r['detected_emotion'] for r in results if r['status'] == 'success']
        from collections import Counter
        emotion_counts = Counter(emotions)
        print(f"\nEmotion distribution:")
        for emotion, count in emotion_counts.most_common():
            print(f"  {emotion}: {count} files ({count/len(emotions)*100:.1f}%)")
    
    print(f"\nResults saved to:")
    print(f"  📄 Detailed: {results_file}")
    print(f"  📋 Summary: {summary_file}")
    
    print("\n✅ Batch processing completed!")

if __name__ == "__main__":
    main()

