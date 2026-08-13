#!/usr/bin/env python3
"""
Setup script for Agentic Chatbot application
"""
import subprocess
import sys
import os

def install_requirements():
    """Install all required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        sys.exit(1)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} is compatible")

def setup_environment():
    """Setup environment variables and directories"""
    # Create necessary directories
    dirs = ["logs", "data", "models"]
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)

    print("✅ Environment directories created")

def main():
    print("🚀 Setting up Agentic Chatbot...")

    check_python_version()
    setup_environment()
    install_requirements()

    print("\n🎉 Setup complete! You can now run the application:")
    print("   python main.py")

if __name__ == "__main__":
    main()