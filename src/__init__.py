"""
NeuroDreamAI - EEG to Dream Visualization System
Author: Prithviraj Chaudhari

A comprehensive system for converting EEG signals into dream narratives and videos.
"""

__version__ = "1.0.0"
__author__ = "Prithviraj Chaudhari"
__email__ = "pruthviraj8811@gmail.com"

from .eeg_classifier import EEGEmotionClassifier, EEGEmotionTrainer, EEGPreprocessor
from .dream_generator import DreamNarrativeGenerator
from .app import NeuroDreamAIApp

__all__ = [
    "EEGEmotionClassifier",
    "EEGEmotionTrainer", 
    "EEGPreprocessor",
    "DreamNarrativeGenerator",
    "NeuroDreamAIApp"
]

