#!/usr/bin/env python3
"""
Basic tests for NeuroDreamAI
Author: Prithviraj Chaudhari
"""

import sys
import os
import unittest
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from eeg_classifier import EEGEmotionClassifier, EEGEmotionTrainer
    from dream_generator import DreamNarrativeGenerator
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

class TestNeuroDreamAI(unittest.TestCase):
    """Basic tests for NeuroDreamAI components"""
    
    def setUp(self):
        """Set up test fixtures"""
        if MODULES_AVAILABLE:
            self.classifier = EEGEmotionClassifier()
            self.trainer = EEGEmotionTrainer(self.classifier)
            self.dream_generator = DreamNarrativeGenerator()
    
    @unittest.skipUnless(MODULES_AVAILABLE, "Modules not available")
    def test_eeg_classifier_initialization(self):
        """Test EEG classifier initialization"""
        self.assertIsNotNone(self.classifier)
        self.assertEqual(len(self.trainer.emotion_labels), 7)
    
    @unittest.skipUnless(MODULES_AVAILABLE, "Modules not available")
    def test_synthetic_eeg_generation(self):
        """Test synthetic EEG data generation"""
        # Generate sample EEG data
        eeg_data = np.random.randn(14, 384)  # 14 channels, 384 samples
        
        # Test emotion prediction
        result = self.trainer.predict_emotion(eeg_data)
        
        self.assertIn('emotion', result)
        self.assertIn('confidence', result)
        self.assertIn('all_probabilities', result)
        self.assertIn(result['emotion'], self.trainer.emotion_labels)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
    
    @unittest.skipUnless(MODULES_AVAILABLE, "Modules not available")
    def test_dream_generation(self):
        """Test dream narrative generation"""
        test_emotions = ['happy', 'sad', 'fear', 'anger']
        
        for emotion in test_emotions:
            with self.subTest(emotion=emotion):
                result = self.dream_generator.generate_dream_narrative(emotion)
                
                self.assertIn('narrative', result)
                self.assertIn('emotion', result)
                self.assertIsInstance(result['narrative'], str)
                self.assertGreater(len(result['narrative']), 10)
                self.assertEqual(result['emotion'], emotion)
    
    def test_package_structure(self):
        """Test that package structure is correct"""
        # Check that required directories exist
        base_dir = os.path.dirname(os.path.dirname(__file__))
        
        required_dirs = ['src', 'data', 'models', 'outputs', 'docs', 'examples']
        for dir_name in required_dirs:
            dir_path = os.path.join(base_dir, dir_name)
            self.assertTrue(os.path.exists(dir_path), f"Directory {dir_name} should exist")
        
        # Check that key files exist
        required_files = ['README.md', 'requirements.txt', 'setup.py', 'LICENSE']
        for file_name in required_files:
            file_path = os.path.join(base_dir, file_name)
            self.assertTrue(os.path.exists(file_path), f"File {file_name} should exist")

if __name__ == '__main__':
    unittest.main()

