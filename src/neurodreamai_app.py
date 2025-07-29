"""
NeuroDreamAI Gradio Application
Author: Prithviraj Chaudhari

A web-based demo application for the NeuroDreamAI system that allows users
to upload EEG data or select emotions to generate dream narratives and videos.
"""

import gradio as gr
import numpy as np
import json
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Import our custom modules with fallback handling
try:
    from eeg_emotion_classifier import EEGEmotionClassifier, EEGEmotionTrainer, EEGPreprocessor
except ImportError as e:
    print(f"Warning: Could not import EEG classifier: {e}")
    EEGEmotionClassifier = None
    EEGEmotionTrainer = None
    EEGPreprocessor = None

try:
    from dream_generator import DreamNarrativeGenerator
except ImportError as e:
    print(f"Warning: Could not import dream generator: {e}")
    DreamNarrativeGenerator = None


class NeuroDreamAIApp:
    """Main application class for NeuroDreamAI demo"""
    
    def __init__(self):
        try:
            import torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        except ImportError:
            self.device = 'cpu'
            
        self.emotion_labels = ['happy', 'sad', 'fear', 'anger', 'surprise', 'disgust', 'neutral']
        
        # Initialize components with fallback
        if EEGEmotionClassifier and EEGEmotionTrainer and EEGPreprocessor:
            self.eeg_classifier = None
            self.eeg_trainer = None
            self.preprocessor = EEGPreprocessor()
            self._initialize_eeg_classifier()
        else:
            print("EEG components not available - using fallback")
            self.eeg_classifier = None
            self.eeg_trainer = None
            self.preprocessor = None
        
        if DreamNarrativeGenerator:
            self.dream_generator = DreamNarrativeGenerator()
        else:
            print("Dream generator not available - using fallback")
            self.dream_generator = None
        
        # Create output directories
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("outputs/dream_videos", exist_ok=True)
        os.makedirs("outputs/visualizations", exist_ok=True)
    
    def _initialize_eeg_classifier(self):
        """Initialize the EEG emotion classifier"""
        try:
            # Try to load pre-trained model
            self.eeg_classifier = EEGEmotionClassifier(
                num_channels=14, 
                sequence_length=384, 
                num_emotions=7
            )
            self.eeg_trainer = EEGEmotionTrainer(self.eeg_classifier, self.device)
            
            model_path = os.path.join(os.path.dirname(__file__), "EEG_emotion_model.pth")
            if os.path.exists(model_path):
                self.eeg_trainer.load_model(model_path)
                print("Pre-trained EEG model loaded successfully")
            else:
                print("No pre-trained model found. Using fallback model for demo.")
                
        except Exception as e:
            print(f"Error initializing EEG classifier: {e}")
            print("Using fallback emotion classification")
            self.eeg_classifier = EEGEmotionClassifier()
            self.eeg_trainer = EEGEmotionTrainer(self.eeg_classifier)
    
    def generate_synthetic_eeg(self, emotion_idx, duration=3.0, sampling_rate=128):
        """Generate synthetic EEG data for demo purposes"""
        num_channels = 14
        num_samples = int(duration * sampling_rate)
        
        # Create base signal
        signal = np.random.randn(num_channels, num_samples) * 0.1
        
        # Add emotion-specific patterns
        t = np.linspace(0, duration, num_samples)
        
        if emotion_idx == 0:  # happy - alpha waves (8-12 Hz)
            signal += 0.3 * np.sin(2 * np.pi * 10 * t)
        elif emotion_idx == 1:  # sad - theta waves (4-8 Hz)
            signal += 0.2 * np.sin(2 * np.pi * 6 * t)
        elif emotion_idx == 2:  # fear - beta waves (13-30 Hz)
            signal += 0.4 * np.sin(2 * np.pi * 20 * t)
        elif emotion_idx == 3:  # anger - gamma waves (30-100 Hz)
            signal += 0.3 * np.sin(2 * np.pi * 40 * t)
        elif emotion_idx == 4:  # surprise - mixed frequencies
            signal += 0.2 * (np.sin(2 * np.pi * 15 * t) + np.sin(2 * np.pi * 25 * t))
        elif emotion_idx == 5:  # disgust - irregular patterns
            signal += 0.2 * np.random.randn(num_channels, num_samples)
        # neutral - keep base signal
        
        return signal
    
    def visualize_eeg_data(self, eeg_data, title="EEG Signal"):
        """Create a visualization of EEG data"""
        fig, axes = plt.subplots(7, 2, figsize=(15, 20))
        fig.suptitle(title, fontsize=16)
        
        channel_names = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 
            'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4'
        ]
        
        for i in range(min(14, eeg_data.shape[0])):
            row = i // 2
            col = i % 2
            
            axes[row, col].plot(eeg_data[i, :1000])  # Plot first 1000 samples
            axes[row, col].set_title(f'Channel {channel_names[i]}')
            axes[row, col].set_xlabel('Time (samples)')
            axes[row, col].set_ylabel('Amplitude (μV)')
            axes[row, col].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(14, 14):
            row = i // 2
            col = i % 2
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"outputs/visualizations/eeg_visualization_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def process_eeg_to_emotion(self, emotion_selection=None, upload_file=None):
        """Process EEG data and predict emotion"""
        try:
            if upload_file is not None:
                # Handle uploaded file (simplified for demo)
                return "File upload processing not implemented in demo. Please use emotion selection.", None, None
            
            if emotion_selection is None:
                return "Please select an emotion or upload EEG data.", None, None
            
            # Map emotion name to index
            emotion_idx = self.emotion_labels.index(emotion_selection.lower())
            
            # Generate synthetic EEG data
            eeg_data = self.generate_synthetic_eeg(emotion_idx)
            
            # Visualize EEG data
            viz_path = self.visualize_eeg_data(eeg_data, f"Synthetic EEG - {emotion_selection}")
            
            # Predict emotion (if model is available)
            if self.eeg_trainer is not None:
                # Preprocess data
                processed_data = self.preprocessor.preprocess([eeg_data])
                prediction = self.eeg_trainer.predict_emotion(processed_data[0])
                
                emotion_result = f"""
                **Detected Emotion:** {prediction['emotion'].title()}
                **Confidence:** {prediction['confidence']:.2%}
                
                **All Probabilities:**
                """
                
                for emotion, prob in prediction['all_probabilities'].items():
                    emotion_result += f"\n- {emotion.title()}: {prob:.2%}"
                
                return emotion_result, viz_path, prediction['emotion']
            else:
                return f"**Selected Emotion:** {emotion_selection}\n(EEG classifier not available - using selected emotion)", viz_path, emotion_selection.lower()
                
        except Exception as e:
            return f"Error processing EEG data: {str(e)}", None, None
    
    def generate_dream_narrative(self, detected_emotion, use_gpt=False):
        """Generate dream narrative from detected emotion"""
        try:
            if detected_emotion is None:
                return "Please process EEG data first to detect emotion."
            
            if self.dream_generator:
                # Use actual dream generator
                dream_result = self.dream_generator.generate_dream_narrative(
                    detected_emotion, use_gpt=use_gpt
                )
            else:
                # Fallback dream generation
                dream_result = self._fallback_dream_generation(detected_emotion)
            
            narrative_text = f"""
            **Dream Narrative:**
            {dream_result['narrative']}
            
            **Emotion:** {dream_result['emotion'].title()}
            **Generation Method:** {dream_result.get('method', 'fallback').upper()}
            **Confidence:** {dream_result.get('confidence', 0.8):.2%}
            """
            
            # Save narrative
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"outputs/dream_narrative_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(dream_result, f, indent=2)
            
            return narrative_text, dream_result['narrative']
            
        except Exception as e:
            return f"Error generating dream narrative: {str(e)}", None
    
    def _fallback_dream_generation(self, emotion):
        """Fallback dream generation when main generator is not available"""
        emotion_dreams = {
            'happy': "I was flying through a bright, colorful sky filled with rainbow clouds. Gentle music played as I soared above a beautiful meadow where flowers danced in the warm breeze.",
            'sad': "I found myself walking alone on an empty beach under a grey sky. The waves whispered melancholy songs as I searched for something I had lost long ago.",
            'fear': "I was running through a dark forest where shadows moved between the trees. My heart pounded as I tried to find a way out of the endless maze of twisted branches.",
            'anger': "I stood on a mountaintop as lightning cracked across a stormy sky. The wind howled around me as I felt the power of the storm matching my inner fire.",
            'surprise': "I opened a door and found myself in a room where gravity worked sideways. Books floated in the air and the ceiling had become a window to another world.",
            'disgust': "I walked through a swamp where the air was thick and heavy. Strange, unpleasant odors filled my nostrils as I tried to find clean ground to stand on.",
            'neutral': "I was sitting in a familiar room, reading a book. Everything seemed normal and peaceful, with soft light filtering through the windows."
        }
        
        narrative = emotion_dreams.get(emotion, emotion_dreams['neutral'])
        
        return {
            'narrative': narrative,
            'emotion': emotion,
            'method': 'fallback',
            'confidence': 0.8,
            'timestamp': time.time()
        }
    
    def simulate_video_generation(self, dream_narrative):
        """Simulate video generation process (placeholder)"""
        try:
            if dream_narrative is None:
                return "Please generate a dream narrative first."
            
            # Break narrative into segments for video
            segments = self.dream_generator.enhance_narrative_for_video(dream_narrative)
            
            video_info = f"""
            **Video Generation Simulation**
            
            **Dream Narrative:** {dream_narrative}
            
            **Video Segments:**
            """
            
            for i, segment in enumerate(segments, 1):
                video_info += f"\n{i}. {segment}"
            
            video_info += f"""
            
            **Recommended Tools:**
            - Pika Labs (for surreal, cinematic style)
            - RunwayML Gen-2 (for controlled generation)
            - Sora (for highest quality, if available)
            
            **Estimated Video Length:** {len(segments) * 3}-{len(segments) * 5} seconds
            **Segments:** {len(segments)}
            """
            
            # Save video generation info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"outputs/dream_videos/video_plan_{timestamp}.txt"
            with open(filename, 'w') as f:
                f.write(video_info)
            
            return video_info
            
        except Exception as e:
            return f"Error in video generation simulation: {str(e)}"
    
    def create_gradio_interface(self):
        """Create the Gradio interface"""
        
        with gr.Blocks(title="NeuroDreamAI - Dream Visualization System", theme=gr.themes.Soft()) as interface:
            
            gr.Markdown("""
            # 🧠 NeuroDreamAI: Dream Visualization System
            
            **Transform brain signals into dream narratives and videos**
            
            This prototype demonstrates the NeuroDreamAI pipeline:
            1. **EEG Processing** → Emotion Detection
            2. **Emotion** → Dream Narrative Generation  
            3. **Dream Text** → Video Generation (Simulated)
            
            ---
            """)
            
            # State variables
            detected_emotion = gr.State(None)
            dream_narrative = gr.State(None)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🔬 Stage 1: EEG Input & Emotion Detection")
                    
                    emotion_selection = gr.Dropdown(
                        choices=self.emotion_labels,
                        label="Select Emotion (for demo)",
                        value="happy"
                    )
                    
                    eeg_file = gr.File(
                        label="Upload EEG Data (Optional)",
                        file_types=[".csv", ".txt", ".npy"]
                    )
                    
                    process_btn = gr.Button("🧠 Process EEG & Detect Emotion", variant="primary")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 EEG Visualization")
                    eeg_plot = gr.Image(label="EEG Signal Visualization")
            
            emotion_output = gr.Textbox(
                label="🎭 Emotion Detection Results",
                lines=8,
                interactive=False
            )
            
            gr.Markdown("---")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ✨ Stage 2: Dream Narrative Generation")
                    
                    use_gpt = gr.Checkbox(
                        label="Use GPT for generation (requires API key)",
                        value=False
                    )
                    
                    generate_dream_btn = gr.Button("✨ Generate Dream Narrative", variant="secondary")
                    
                    dream_output = gr.Textbox(
                        label="📖 Generated Dream Narrative",
                        lines=6,
                        interactive=False
                    )
            
            gr.Markdown("---")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🎬 Stage 3: Video Generation (Simulation)")
                    
                    generate_video_btn = gr.Button("🎬 Simulate Video Generation", variant="secondary")
                    
                    video_output = gr.Textbox(
                        label="🎥 Video Generation Plan",
                        lines=12,
                        interactive=False
                    )
            
            gr.Markdown("""
            ---
            ### 📋 Instructions:
            1. **Select an emotion** from the dropdown (simulates EEG emotion detection)
            2. **Click "Process EEG"** to generate synthetic EEG data and visualize it
            3. **Click "Generate Dream"** to create a narrative based on the detected emotion
            4. **Click "Simulate Video"** to see how the narrative would be converted to video
            
            ### 🔧 Technical Notes:
            - EEG data is synthetically generated for demo purposes
            - Real EEG processing would require specialized hardware and preprocessing
            - Video generation is simulated - actual implementation would use tools like Pika Labs, RunwayML, or Sora
            - The emotion classifier uses a CNN-LSTM architecture trained on EEG emotion datasets
            """)
            
            # Event handlers
            process_btn.click(
                fn=self.process_eeg_to_emotion,
                inputs=[emotion_selection, eeg_file],
                outputs=[emotion_output, eeg_plot, detected_emotion]
            )
            
            generate_dream_btn.click(
                fn=self.generate_dream_narrative,
                inputs=[detected_emotion, use_gpt],
                outputs=[dream_output, dream_narrative]
            )
            
            generate_video_btn.click(
                fn=self.simulate_video_generation,
                inputs=[dream_narrative],
                outputs=[video_output]
            )
        
        return interface


def main():
    """Main function to run the NeuroDreamAI application"""
    print("Initializing NeuroDreamAI Application...")
    
    # Create application instance
    app = NeuroDreamAIApp()
    
    # Create Gradio interface
    interface = app.create_gradio_interface()
    
    print("Starting NeuroDreamAI Demo...")
    print("Access the application at: http://localhost:7860")
    
    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    main()

