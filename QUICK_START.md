# 🚀 NeuroDreamAI - Quick Start Guide

**Get up and running in 5 minutes!**

## 📋 Prerequisites

- Python 3.8+ installed
- VS Code (recommended)
- 8GB+ RAM
- Internet connection (for setup)

## ⚡ Super Quick Setup

### 1. Download & Extract
Download the NeuroDreamAI_Complete folder to your computer.

### 2. Open in VS Code
```bash
code NeuroDreamAI_Complete/
```

### 3. Run Setup (One Command!)
```bash
python setup.py
```
*This will create virtual environment and install all dependencies automatically.*

### 4. Activate Environment
**Linux/Mac:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 5. Start the App
```bash
python src/app.py
```

### 6. Open Browser
Go to: **http://localhost:7860**

## 🎯 First Test

1. **Select emotion**: Choose "Happy" from dropdown
2. **Click "Process EEG Data"**: Wait for emotion detection
3. **Click "Generate Dream Narrative"**: Read your AI-generated dream!

## 🔧 Optional: Add API Keys

For GPT-powered dreams:
1. Copy `.env.example` to `.env`
2. Add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_key_here
   ```
3. Restart the app

## 📚 Next Steps

- **Read**: `README.md` for complete guide
- **Try**: `examples/basic_usage.py` for command line
- **Learn**: `docs/USER_GUIDE.md` for advanced features
- **Test**: Sample EEG files in `data/sample_eeg/`

## 🆘 Need Help?

**Common Issues:**
- **"No module named..."**: Run `python setup.py` again
- **Port 7860 busy**: Try `python src/app.py --port 7861`
- **Slow performance**: Close other applications

**Contact:** pruthviraj8811@gmail.com

---

**🎉 You're ready to explore AI-powered dream visualization!**

