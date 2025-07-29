"""
Architecture Diagram Generator for NeuroDreamAI
Author: Prithviraj Chaudhari

This script generates a visual architecture diagram showing the NeuroDreamAI pipeline.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_architecture_diagram():
    """Create the NeuroDreamAI architecture diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Define colors
    colors = {
        'input': '#E8F4FD',
        'processing': '#B8E6B8', 
        'ai_model': '#FFE4B5',
        'output': '#FFB6C1',
        'arrow': '#4A90E2'
    }
    
    # Title
    ax.text(5, 7.5, 'NeuroDreamAI: Dream Visualization Pipeline', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Stage 1: EEG Input
    stage1_box = FancyBboxPatch((0.5, 6), 1.8, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['input'], 
                                edgecolor='black', linewidth=2)
    ax.add_patch(stage1_box)
    ax.text(1.4, 6.4, 'Stage 1:\nEEG Data Input', 
            fontsize=11, fontweight='bold', ha='center', va='center')
    
    # EEG details
    ax.text(1.4, 5.5, '• 14-channel EEG\n• 128 Hz sampling\n• 3-second windows', 
            fontsize=9, ha='center', va='top')
    
    # Stage 2: Emotion Classification
    stage2_box = FancyBboxPatch((3, 6), 1.8, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['ai_model'], 
                                edgecolor='black', linewidth=2)
    ax.add_patch(stage2_box)
    ax.text(3.9, 6.4, 'Stage 2:\nEmotion Classification', 
            fontsize=11, fontweight='bold', ha='center', va='center')
    
    # CNN-LSTM details
    ax.text(3.9, 5.5, '• CNN-LSTM Model\n• 7 Emotions\n• 89% Accuracy', 
            fontsize=9, ha='center', va='top')
    
    # Stage 3: Dream Generation
    stage3_box = FancyBboxPatch((5.5, 6), 1.8, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['ai_model'], 
                                edgecolor='black', linewidth=2)
    ax.add_patch(stage3_box)
    ax.text(6.4, 6.4, 'Stage 3:\nDream Generation', 
            fontsize=11, fontweight='bold', ha='center', va='center')
    
    # Dream generation details
    ax.text(6.4, 5.5, '• GPT-4 / Templates\n• Emotion-based\n• 2-3 sentences', 
            fontsize=9, ha='center', va='top')
    
    # Stage 4: Video Generation
    stage4_box = FancyBboxPatch((8, 6), 1.8, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['processing'], 
                                edgecolor='black', linewidth=2)
    ax.add_patch(stage4_box)
    ax.text(8.9, 6.4, 'Stage 4:\nVideo Generation', 
            fontsize=11, fontweight='bold', ha='center', va='center')
    
    # Video generation details
    ax.text(8.9, 5.5, '• Pika Labs / Sora\n• Text-to-Video\n• 30-60 seconds', 
            fontsize=9, ha='center', va='top')
    
    # Arrows between stages
    arrow1 = ConnectionPatch((2.3, 6.4), (3, 6.4), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, 
                            mutation_scale=20, fc=colors['arrow'], ec=colors['arrow'])
    ax.add_patch(arrow1)
    
    arrow2 = ConnectionPatch((4.8, 6.4), (5.5, 6.4), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, 
                            mutation_scale=20, fc=colors['arrow'], ec=colors['arrow'])
    ax.add_patch(arrow2)
    
    arrow3 = ConnectionPatch((7.3, 6.4), (8, 6.4), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, 
                            mutation_scale=20, fc=colors['arrow'], ec=colors['arrow'])
    ax.add_patch(arrow3)
    
    # Data flow labels
    ax.text(2.65, 6.7, 'EEG Signals', fontsize=9, ha='center', style='italic')
    ax.text(5.15, 6.7, 'Emotion Label', fontsize=9, ha='center', style='italic')
    ax.text(7.65, 6.7, 'Dream Text', fontsize=9, ha='center', style='italic')
    
    # Technical Stack Section
    ax.text(5, 4.5, 'Technical Implementation Stack', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Data Sources
    data_box = FancyBboxPatch((0.5, 3), 2, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['input'], 
                              edgecolor='black', linewidth=1)
    ax.add_patch(data_box)
    ax.text(1.5, 3.7, 'Data Sources', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.5, 3.3, '• DREAMER Dataset\n• DEAP Dataset\n• SEED Dataset\n• DreamBank', 
            fontsize=9, ha='center', va='center')
    
    # AI Models
    models_box = FancyBboxPatch((3, 3), 2, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['ai_model'], 
                                edgecolor='black', linewidth=1)
    ax.add_patch(models_box)
    ax.text(4, 3.7, 'AI Models', fontsize=12, fontweight='bold', ha='center')
    ax.text(4, 3.3, '• PyTorch CNN-LSTM\n• GPT-4 / GPT-3.5\n• Template Engine\n• MNE Preprocessing', 
            fontsize=9, ha='center', va='center')
    
    # Tools & Platforms
    tools_box = FancyBboxPatch((5.5, 3), 2, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['processing'], 
                               edgecolor='black', linewidth=1)
    ax.add_patch(tools_box)
    ax.text(6.5, 3.7, 'Tools & Platforms', fontsize=12, fontweight='bold', ha='center')
    ax.text(6.5, 3.3, '• Gradio Interface\n• Pika Labs\n• RunwayML Gen-2\n• OpenAI Sora', 
            fontsize=9, ha='center', va='center')
    
    # Output Formats
    output_box = FancyBboxPatch((8, 3), 1.5, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['output'], 
                                edgecolor='black', linewidth=1)
    ax.add_patch(output_box)
    ax.text(8.75, 3.7, 'Outputs', fontsize=12, fontweight='bold', ha='center')
    ax.text(8.75, 3.3, '• Dream Videos\n• Narratives\n• Visualizations\n• Reports', 
            fontsize=9, ha='center', va='center')
    
    # Performance Metrics
    ax.text(5, 2.2, 'Performance Metrics', 
            fontsize=14, fontweight='bold', ha='center')
    
    metrics_text = """
    • EEG Emotion Classification: 89% accuracy (target from paper)
    • Dream Narrative Generation: 75% BLEU score (target from paper)
    • Video Generation: 30-60 second clips
    • Processing Time: <5 minutes end-to-end
    • Supported Emotions: Happy, Sad, Fear, Anger, Surprise, Disgust, Neutral
    """
    
    ax.text(5, 1.5, metrics_text, fontsize=10, ha='center', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
    
    # Footer
    ax.text(5, 0.3, 'NeuroDreamAI Prototype - Prithviraj Chaudhari', 
            fontsize=10, ha='center', style='italic')
    
    plt.tight_layout()
    plt.savefig('outputs/architecture_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Architecture diagram saved to outputs/architecture_diagram.png")

def create_data_flow_diagram():
    """Create a detailed data flow diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(5, 7.5, 'NeuroDreamAI Data Flow Diagram', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Input layer
    ax.text(1, 6.8, 'Input Layer', fontsize=14, fontweight='bold', ha='center')
    
    # EEG input
    eeg_box = patches.Rectangle((0.2, 6), 1.6, 0.6, 
                               facecolor='lightblue', edgecolor='black')
    ax.add_patch(eeg_box)
    ax.text(1, 6.3, 'Raw EEG\n(14 channels)', fontsize=10, ha='center', va='center')
    
    # Processing layer
    ax.text(3, 6.8, 'Processing Layer', fontsize=14, fontweight='bold', ha='center')
    
    # Preprocessing
    prep_box = patches.Rectangle((2.2, 6), 1.6, 0.6, 
                                facecolor='lightgreen', edgecolor='black')
    ax.add_patch(prep_box)
    ax.text(3, 6.3, 'Preprocessing\n(Filter, Normalize)', fontsize=10, ha='center', va='center')
    
    # Feature extraction
    feat_box = patches.Rectangle((2.2, 5), 1.6, 0.6, 
                                facecolor='lightgreen', edgecolor='black')
    ax.add_patch(feat_box)
    ax.text(3, 5.3, 'Feature Extraction\n(CNN Layers)', fontsize=10, ha='center', va='center')
    
    # Temporal modeling
    temp_box = patches.Rectangle((2.2, 4), 1.6, 0.6, 
                                facecolor='lightgreen', edgecolor='black')
    ax.add_patch(temp_box)
    ax.text(3, 4.3, 'Temporal Modeling\n(LSTM Layers)', fontsize=10, ha='center', va='center')
    
    # AI Models layer
    ax.text(5.5, 6.8, 'AI Models Layer', fontsize=14, fontweight='bold', ha='center')
    
    # Emotion classifier
    emotion_box = patches.Rectangle((4.7, 5.5), 1.6, 1.2, 
                                   facecolor='orange', edgecolor='black')
    ax.add_patch(emotion_box)
    ax.text(5.5, 6.1, 'Emotion\nClassifier\n(CNN-LSTM)', fontsize=10, ha='center', va='center')
    
    # Dream generator
    dream_box = patches.Rectangle((4.7, 3.5), 1.6, 1.2, 
                                 facecolor='yellow', edgecolor='black')
    ax.add_patch(dream_box)
    ax.text(5.5, 4.1, 'Dream\nGenerator\n(GPT/Template)', fontsize=10, ha='center', va='center')
    
    # Output layer
    ax.text(8, 6.8, 'Output Layer', fontsize=14, fontweight='bold', ha='center')
    
    # Video generator
    video_box = patches.Rectangle((7.2, 5), 1.6, 0.6, 
                                 facecolor='pink', edgecolor='black')
    ax.add_patch(video_box)
    ax.text(8, 5.3, 'Video Generator\n(Pika/Sora)', fontsize=10, ha='center', va='center')
    
    # Final output
    output_box = patches.Rectangle((7.2, 4), 1.6, 0.6, 
                                  facecolor='lightcoral', edgecolor='black')
    ax.add_patch(output_box)
    ax.text(8, 4.3, 'Dream Video\n(30-60s)', fontsize=10, ha='center', va='center')
    
    # Arrows
    arrows = [
        ((1.8, 6.3), (2.2, 6.3)),  # EEG to preprocessing
        ((3, 5.8), (3, 5.6)),      # Preprocessing to features
        ((3, 4.8), (3, 4.6)),      # Features to temporal
        ((3.8, 4.3), (4.7, 5.5)),  # Temporal to emotion
        ((5.5, 5.3), (5.5, 4.7)),  # Emotion to dream
        ((6.3, 4.1), (7.2, 5.3)),  # Dream to video
        ((8, 4.8), (8, 4.6)),      # Video to output
    ]
    
    for start, end in arrows:
        arrow = patches.FancyArrowPatch(start, end, 
                                       arrowstyle='->', 
                                       mutation_scale=15, 
                                       color='blue')
        ax.add_patch(arrow)
    
    # Data labels
    data_labels = [
        (1.5, 6.5, 'Raw Signals'),
        (3.5, 5.8, 'Clean Signals'),
        (3.5, 4.8, 'Spatial Features'),
        (4.2, 4.8, 'Temporal Features'),
        (5.8, 4.8, 'Emotion Label'),
        (6.8, 4.8, 'Dream Text'),
        (8.3, 4.8, 'Video Clips'),
    ]
    
    for x, y, label in data_labels:
        ax.text(x, y, label, fontsize=8, ha='center', style='italic', 
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Technical specifications
    specs_text = """
    Technical Specifications:
    • EEG Channels: 14 (standard 10-20 system)
    • Sampling Rate: 128 Hz
    • Window Size: 3 seconds (384 samples)
    • CNN Filters: 64, 128
    • LSTM Hidden Units: 128
    • Emotions: 7 classes
    • Dream Length: 2-3 sentences
    • Video Duration: 30-60 seconds
    """
    
    ax.text(1, 2.5, specs_text, fontsize=9, va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('outputs/data_flow_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Data flow diagram saved to outputs/data_flow_diagram.png")

if __name__ == "__main__":
    import os
    os.makedirs('outputs', exist_ok=True)
    
    print("Generating NeuroDreamAI architecture diagrams...")
    create_architecture_diagram()
    create_data_flow_diagram()
    print("Diagrams generated successfully!")

