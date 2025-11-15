#!/bin/bash

# Quick deployment script for SuhasMartha/Parkinson-Detector

echo "🚀 Ultimate Parkinson's Detector - Deployment Script"
echo "=================================================="

# Check Python
echo "✓ Checking Python..."
python --version

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "✓ Creating virtual environment..."
    python -m venv venv
fi

# Activate venv
echo "✓ Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "✓ Installing dependencies..."
pip install -r requirements_suhas.txt

# Verify models
echo "✓ Verifying model files..."
if [ -f "models/mri_model.h5" ]; then echo "  ✅ mri_model.h5"; else echo "  ❌ mri_model.h5 MISSING"; fi
if [ -f "models/drawing_model.h5" ]; then echo "  ✅ drawing_model.h5"; else echo "  ❌ drawing_model.h5 MISSING"; fi
if [ -f "models/speech_model.pkl" ]; then echo "  ✅ speech_model.pkl"; else echo "  ❌ speech_model.pkl MISSING"; fi
if [ -f "models/gait_model.pkl" ]; then echo "  ✅ gait_model.pkl"; else echo "  ❌ gait_model.pkl MISSING"; fi

echo ""
echo "=================================================="
echo "✅ Setup complete!"
echo ""
echo "To run the app:"
echo "  streamlit run app.py"
echo ""
echo "App will open at: http://localhost:8501"
echo "=================================================="
