"""
Dream Narrative Generator for NeuroDreamAI
Author: Prithviraj Chaudhari

This module implements a GPT-based dream narrative generator that creates
vivid dream stories based on detected emotions from EEG signals.
"""

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI module not available. Using template-based generation only.")

import random
import json
from typing import Dict, List, Optional
import time


class DreamNarrativeGenerator:
    """
    A class for generating dream narratives based on emotions using GPT models
    """
    
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.8, max_tokens=150):
        """
        Initialize the dream generator
        
        Args:
            model_name: OpenAI model to use
            temperature: Creativity level (0.0 to 1.0)
            max_tokens: Maximum length of generated text
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Emotion-specific dream themes and elements
        self.emotion_themes = {
            'happy': {
                'settings': ['colorful meadows', 'sunny beaches', 'floating islands', 'magical gardens', 'crystal palaces'],
                'characters': ['friendly animals', 'laughing children', 'wise mentors', 'dancing spirits', 'glowing beings'],
                'actions': ['flying freely', 'dancing with joy', 'discovering treasures', 'singing melodies', 'embracing light'],
                'objects': ['golden butterflies', 'rainbow bridges', 'sparkling water', 'blooming flowers', 'shimmering stars']
            },
            'sad': {
                'settings': ['empty train stations', 'rainy streets', 'abandoned houses', 'misty forests', 'grey landscapes'],
                'characters': ['lost souls', 'crying figures', 'distant memories', 'fading shadows', 'lonely wanderers'],
                'actions': ['walking alone', 'searching endlessly', 'calling out silently', 'watching from afar', 'remembering the past'],
                'objects': ['wilted flowers', 'broken mirrors', 'old photographs', 'falling leaves', 'empty chairs']
            },
            'fear': {
                'settings': ['dark alleys', 'haunted forests', 'endless corridors', 'stormy nights', 'underground tunnels'],
                'characters': ['shadowy figures', 'pursuing monsters', 'faceless strangers', 'whispering voices', 'lurking presences'],
                'actions': ['running desperately', 'hiding frantically', 'screaming silently', 'falling endlessly', 'being chased'],
                'objects': ['creaking doors', 'flickering lights', 'moving shadows', 'cold winds', 'echoing footsteps']
            },
            'anger': {
                'settings': ['burning landscapes', 'stormy seas', 'crumbling cities', 'volcanic mountains', 'raging battlefields'],
                'characters': ['fierce warriors', 'roaring beasts', 'confronting enemies', 'rebellious spirits', 'powerful forces'],
                'actions': ['fighting fiercely', 'breaking barriers', 'shouting loudly', 'destroying obstacles', 'standing defiant'],
                'objects': ['blazing fires', 'crashing waves', 'thunderous sounds', 'sharp weapons', 'explosive energy']
            },
            'surprise': {
                'settings': ['shifting mazes', 'transforming rooms', 'impossible architectures', 'morphing landscapes', 'paradoxical spaces'],
                'characters': ['unexpected visitors', 'shape-shifters', 'mysterious guides', 'sudden appearances', 'changing faces'],
                'actions': ['discovering secrets', 'witnessing transformations', 'experiencing reversals', 'finding hidden doors', 'realizing truths'],
                'objects': ['opening portals', 'revealing mirrors', 'unfolding maps', 'appearing gifts', 'shifting patterns']
            },
            'disgust': {
                'settings': ['decaying swamps', 'polluted cities', 'rotting buildings', 'contaminated waters', 'toxic environments'],
                'characters': ['repulsive creatures', 'corrupted beings', 'diseased figures', 'grotesque forms', 'tainted souls'],
                'actions': ['recoiling in horror', 'trying to escape', 'covering the face', 'feeling nauseous', 'seeking purity'],
                'objects': ['foul odors', 'slimy textures', 'rotting matter', 'poisonous substances', 'contaminated food']
            },
            'neutral': {
                'settings': ['ordinary rooms', 'familiar streets', 'everyday places', 'common landscapes', 'regular environments'],
                'characters': ['known people', 'casual acquaintances', 'family members', 'colleagues', 'neighbors'],
                'actions': ['going about routines', 'having conversations', 'performing tasks', 'observing quietly', 'moving normally'],
                'objects': ['household items', 'work tools', 'common objects', 'regular furniture', 'everyday things']
            }
        }
        
        # Dream narrative templates
        self.narrative_templates = [
            "I found myself in {setting}. {character} appeared before me, and I began {action}. The {object} caught my attention, filling me with {emotion_description}.",
            "The dream started in {setting}. As I {action}, I encountered {character}. The presence of {object} made everything feel {emotion_description}.",
            "I was {action} through {setting} when {character} emerged. The {object} seemed to {action_verb}, creating a {emotion_description} atmosphere.",
            "In my dream, {setting} surrounded me. {character} guided me as I {action}. The {object} pulsed with {emotion_description} energy.",
            "The scene unfolded in {setting}. While {action}, I noticed {character} nearby. The {object} whispered secrets of {emotion_description}."
        ]
    
    def generate_dream_narrative(self, emotion: str, use_gpt: bool = True) -> Dict[str, str]:
        """
        Generate a dream narrative based on the given emotion
        
        Args:
            emotion: The detected emotion ('happy', 'sad', 'fear', etc.)
            use_gpt: Whether to use GPT for generation or template-based approach
            
        Returns:
            Dictionary containing the generated narrative and metadata
        """
        if use_gpt:
            return self._generate_with_gpt(emotion)
        else:
            return self._generate_with_template(emotion)
    
    def _generate_with_gpt(self, emotion: str) -> Dict[str, str]:
        """Generate dream narrative using GPT"""
        
        if not OPENAI_AVAILABLE:
            print("OpenAI not available, falling back to template generation")
            return self._generate_with_template(emotion)
        
        # Create emotion-specific prompt
        prompt = self._create_emotion_prompt(emotion)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a creative dream interpreter who generates vivid, surreal dream narratives. Keep responses to 2-3 sentences and make them emotionally evocative and dreamlike."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            narrative = response.choices[0].message.content.strip()
            
            return {
                'narrative': narrative,
                'emotion': emotion,
                'method': 'gpt',
                'confidence': 0.9,  # High confidence for GPT-generated content
                'timestamp': time.time()
            }
            
        except Exception as e:
            print(f"Error generating with GPT: {e}")
            # Fallback to template-based generation
            return self._generate_with_template(emotion)
    
    def _generate_with_template(self, emotion: str) -> Dict[str, str]:
        """Generate dream narrative using predefined templates"""
        
        if emotion not in self.emotion_themes:
            emotion = 'neutral'  # Default fallback
        
        theme = self.emotion_themes[emotion]
        template = random.choice(self.narrative_templates)
        
        # Select random elements from the emotion theme
        setting = random.choice(theme['settings'])
        character = random.choice(theme['characters'])
        action = random.choice(theme['actions'])
        obj = random.choice(theme['objects'])
        
        # Emotion descriptions
        emotion_descriptions = {
            'happy': 'joyful and uplifting',
            'sad': 'melancholic and heavy',
            'fear': 'terrifying and ominous',
            'anger': 'intense and fiery',
            'surprise': 'shocking and bewildering',
            'disgust': 'repulsive and disturbing',
            'neutral': 'calm and ordinary'
        }
        
        emotion_description = emotion_descriptions.get(emotion, 'mysterious')
        
        # Action verbs for objects
        action_verbs = ['glow', 'pulse', 'shimmer', 'whisper', 'dance', 'float', 'transform']
        action_verb = random.choice(action_verbs)
        
        # Fill in the template
        narrative = template.format(
            setting=setting,
            character=character,
            action=action,
            object=obj,
            emotion_description=emotion_description,
            action_verb=action_verb
        )
        
        return {
            'narrative': narrative,
            'emotion': emotion,
            'method': 'template',
            'confidence': 0.7,  # Lower confidence for template-based
            'timestamp': time.time(),
            'elements': {
                'setting': setting,
                'character': character,
                'action': action,
                'object': obj
            }
        }
    
    def _create_emotion_prompt(self, emotion: str) -> str:
        """Create a GPT prompt based on the emotion"""
        
        emotion_prompts = {
            'happy': "Write a short, vivid dream about joy and happiness. Include flying, bright colors, or magical elements. Make it feel uplifting and wonderful.",
            'sad': "Write a short, vivid dream about sadness and melancholy. Include rain, empty spaces, or lost connections. Make it feel poignant and touching.",
            'fear': "Write a short, vivid dream about fear and anxiety. Include darkness, being chased, or unknown threats. Make it feel tense and frightening.",
            'anger': "Write a short, vivid dream about anger and frustration. Include fire, storms, or confrontations. Make it feel intense and powerful.",
            'surprise': "Write a short, vivid dream about surprise and wonder. Include unexpected transformations, sudden revelations, or impossible events. Make it feel astonishing.",
            'disgust': "Write a short, vivid dream about disgust and revulsion. Include decay, contamination, or repulsive elements. Make it feel unsettling.",
            'neutral': "Write a short, vivid dream about everyday life with a surreal twist. Include familiar places or people in unusual situations. Make it feel dreamlike but calm."
        }
        
        return emotion_prompts.get(emotion, emotion_prompts['neutral'])
    
    def generate_multiple_narratives(self, emotion: str, count: int = 3) -> List[Dict[str, str]]:
        """Generate multiple dream narratives for the same emotion"""
        narratives = []
        
        for i in range(count):
            # Alternate between GPT and template methods
            use_gpt = (i % 2 == 0)
            narrative = self.generate_dream_narrative(emotion, use_gpt)
            narrative['variant'] = i + 1
            narratives.append(narrative)
            
            # Small delay to avoid rate limiting
            if use_gpt:
                time.sleep(0.5)
        
        return narratives
    
    def enhance_narrative_for_video(self, narrative: str) -> List[str]:
        """
        Break down a narrative into video-friendly segments
        
        Args:
            narrative: The dream narrative text
            
        Returns:
            List of sentences/phrases suitable for video generation
        """
        # Split into sentences
        sentences = narrative.split('. ')
        
        # Clean up sentences
        video_segments = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not sentence.endswith('.'):
                sentence += '.'
            if sentence:
                video_segments.append(sentence)
        
        # If only one long sentence, try to split on other punctuation
        if len(video_segments) == 1:
            text = video_segments[0]
            # Split on commas or other natural breaks
            parts = text.replace(',', '.|').replace(';', '.|').split('.|')
            video_segments = [part.strip() + '.' for part in parts if part.strip()]
        
        return video_segments
    
    def save_narratives(self, narratives: List[Dict], filename: str):
        """Save generated narratives to a JSON file"""
        with open(filename, 'w') as f:
            json.dump(narratives, f, indent=2)
        print(f"Narratives saved to {filename}")
    
    def load_narratives(self, filename: str) -> List[Dict]:
        """Load narratives from a JSON file"""
        with open(filename, 'r') as f:
            narratives = json.load(f)
        print(f"Narratives loaded from {filename}")
        return narratives


def demo_dream_generation():
    """Demonstrate the dream narrative generator"""
    
    generator = DreamNarrativeGenerator()
    emotions = ['happy', 'sad', 'fear', 'anger', 'surprise', 'disgust', 'neutral']
    
    print("=== NeuroDreamAI Dream Narrative Generator Demo ===\n")
    
    all_narratives = []
    
    for emotion in emotions:
        print(f"Generating dreams for emotion: {emotion.upper()}")
        print("-" * 50)
        
        # Generate with template method (always available)
        template_dream = generator.generate_dream_narrative(emotion, use_gpt=False)
        print(f"Template-based: {template_dream['narrative']}")
        
        # Try to generate with GPT (may fail if API not available)
        try:
            gpt_dream = generator.generate_dream_narrative(emotion, use_gpt=True)
            print(f"GPT-based: {gpt_dream['narrative']}")
            all_narratives.extend([template_dream, gpt_dream])
        except:
            print("GPT-based: [API not available, using template only]")
            all_narratives.append(template_dream)
        
        print()
    
    # Save all narratives
    generator.save_narratives(all_narratives, "sample_dream_narratives.json")
    
    # Demonstrate video segmentation
    print("=== Video Segmentation Example ===")
    sample_narrative = all_narratives[0]['narrative']
    video_segments = generator.enhance_narrative_for_video(sample_narrative)
    print(f"Original: {sample_narrative}")
    print("Video segments:")
    for i, segment in enumerate(video_segments, 1):
        print(f"  {i}. {segment}")


if __name__ == "__main__":
    demo_dream_generation()

