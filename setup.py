#!/usr/bin/env python3
"""
NeuroDreamAI Setup Script
Author: Prithviraj Chaudhari

Automated setup script for the NeuroDreamAI development environment.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("=" * 60)
    print("🧠 NeuroDreamAI - Automated Setup")
    print("Created by: Prithviraj Chaudhari")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8+ is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")

def create_virtual_environment():
    """Create and activate virtual environment"""
    print("\n📦 Creating virtual environment...")
    
    if os.path.exists("venv"):
        print("Virtual environment already exists")
        return
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to create virtual environment")
        sys.exit(1)

def get_pip_command():
    """Get the correct pip command for the platform"""
    if platform.system() == "Windows":
        return os.path.join("venv", "Scripts", "pip")
    else:
        return os.path.join("venv", "bin", "pip")

def install_dependencies():
    """Install Python dependencies"""
    print("\n📚 Installing dependencies...")
    
    pip_cmd = get_pip_command()
    
    try:
        # Upgrade pip first
        subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
        
        # Install package in development mode
        subprocess.run([pip_cmd, "install", "-e", "."], check=True)
        
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("Try running manually:")
        print(f"  {pip_cmd} install -r requirements.txt")
        sys.exit(1)

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating project directories...")
    
    directories = [
        "data/sample_eeg",
        "data/dream_templates", 
        "data/emotion_mappings",
        "models/model_configs",
        "outputs/dreams",
        "outputs/videos",
        "outputs/visualizations",
        "assets/sample_videos",
        "assets/screenshots",
        "assets/logos"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Project directories created")

def create_env_file():
    """Create example .env file"""
    print("\n⚙️  Creating environment configuration...")
    
    env_content = """# NeuroDreamAI Environment Configuration
# Copy this file to .env and update with your API keys

# OpenAI API (for GPT-based dream generation)
OPENAI_API_KEY=your_openai_api_key_here

# Video generation platforms (optional)
PIKA_LABS_API_KEY=your_pika_labs_key
RUNWAY_API_KEY=your_runway_key

# EEG processing settings
EEG_SAMPLING_RATE=128
EEG_CHANNELS=14
EMOTION_CONFIDENCE_THRESHOLD=0.7

# Output settings
SAVE_INTERMEDIATE_RESULTS=true
OUTPUT_VIDEO_QUALITY=high
DEBUG_MODE=false
"""
    
    with open(".env.example", "w") as f:
        f.write(env_content)
    
    print("✅ Environment configuration created (.env.example)")

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# NeuroDreamAI .gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Environment variables
.env

# Data files
data/private/
*.eeg
*.bdf
*.edf

# Model files
models/*.pth
models/*.pkl
models/*.h5

# Output files
outputs/
temp/
cache/

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# API Keys
api_keys.txt
secrets.json
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print("✅ .gitignore created")

def copy_source_files():
    """Copy source files from the prototype"""
    print("\n📋 Copying source files...")
    
    # Copy from the prototype directory
    prototype_dir = Path("../NeuroDreamAI_Prototype")
    
    if prototype_dir.exists():
        import shutil
        
        # Copy app files to src
        if (prototype_dir / "app").exists():
            for file in (prototype_dir / "app").glob("*.py"):
                shutil.copy2(file, "src/")
        
        # Copy documentation
        if (prototype_dir / "docs").exists():
            for file in (prototype_dir / "docs").glob("*"):
                if file.is_file():
                    shutil.copy2(file, "docs/")
        
        # Copy sample data
        if (prototype_dir / "sample_dream_narratives.json").exists():
            shutil.copy2(prototype_dir / "sample_dream_narratives.json", "data/")
        
        print("✅ Source files copied")
    else:
        print("⚠️  Prototype directory not found, creating placeholder files")
        create_placeholder_files()

def create_placeholder_files():
    """Create placeholder source files"""
    
    # Create __init__.py files
    init_files = [
        "src/__init__.py",
        "tests/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
    
    # Create main app file
    app_content = '''"""
NeuroDreamAI Main Application
Author: Prithviraj Chaudhari
"""

import gradio as gr

def main():
    """Main application entry point"""
    print("🧠 NeuroDreamAI Starting...")
    print("Visit: http://localhost:7860")
    
    # Create simple demo interface
    interface = gr.Interface(
        fn=lambda x: f"Hello from NeuroDreamAI! Input: {x}",
        inputs="text",
        outputs="text",
        title="🧠 NeuroDreamAI Demo",
        description="EEG-to-Dream Visualization System"
    )
    
    interface.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
'''
    
    with open("src/app.py", "w") as f:
        f.write(app_content)

def print_completion_message():
    """Print setup completion message"""
    print("\n" + "=" * 60)
    print("🎉 NeuroDreamAI Setup Complete!")
    print("=" * 60)
    
    if platform.system() == "Windows":
        activate_cmd = "venv\\Scripts\\activate"
        python_cmd = "python"
    else:
        activate_cmd = "source venv/bin/activate"
        python_cmd = "python"
    
    print("\n🚀 Next Steps:")
    print(f"1. Activate virtual environment: {activate_cmd}")
    print(f"2. Run the demo: {python_cmd} src/app.py")
    print("3. Open browser to: http://localhost:7860")
    print("4. Open project in VS Code: code .")
    
    print("\n📚 Documentation:")
    print("- README.md - Complete setup guide")
    print("- docs/ - Technical documentation")
    print("- examples/ - Usage examples")
    
    print("\n⚙️  Configuration:")
    print("- Copy .env.example to .env")
    print("- Add your API keys")
    print("- Customize settings as needed")
    
    print("\n💡 Need help? Email: pruthviraj8811@gmail.com")

def main():
    """Main setup function"""
    print_banner()
    check_python_version()
    create_virtual_environment()
    create_directories()
    create_env_file()
    create_gitignore()
    copy_source_files()
    install_dependencies()
    print_completion_message()

if __name__ == "__main__":
    main()

