# Installation Guide

## System Requirements

### Hardware
- **CPU**: Modern multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended for faster processing)
- **Storage**: 5GB free space for installation + space for videos and outputs

### Software
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python**: 3.10 or higher
- **CUDA**: 11.7+ (if using GPU acceleration)

---

## Step-by-Step Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/Motion-OS-Claude.git
cd Motion-OS-Claude
```

### 2. Create Virtual Environment

**Using venv (recommended):**
```bash
python -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n motion-os python=3.10
conda activate motion-os
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### 4. Install Motion OS

```bash
# Development installation (editable)
pip install -e .

# Or standard installation
pip install .
```

### 5. Verify Installation

```bash
# Check installation
python -c "import motion_os; print(motion_os.__version__)"

# Run help command
motion-os --help
```

---

## GPU Support (Optional)

For faster processing with NVIDIA GPUs:

### Install CUDA Toolkit

1. Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
2. Install following NVIDIA's instructions
3. Verify installation:
   ```bash
   nvcc --version
   ```

### Install PyTorch with CUDA

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU support:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

---

## Installing WHAM Model (Optional)

For full 3D pose estimation capabilities:

### 1. Clone WHAM Repository

```bash
cd ..
git clone https://github.com/yohanshin/WHAM.git
cd WHAM
```

### 2. Download Checkpoints

```bash
# Download from releases or model repository
# Place checkpoint in Motion-OS-Claude/models/wham/
```

### 3. Update Configuration

Edit `motion_os/config/config.yaml`:
```yaml
wham:
  checkpoint_path: "models/wham/checkpoint.pth"
```

---

## Installing SAM 3D Body (Optional)

For enhanced body segmentation:

### 1. Clone SAM 3D Repository

```bash
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
```

### 2. Download Model Weights

```bash
# Download SAM checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P models/sam3d/
```

---

## Troubleshooting

### Common Issues

#### 1. ImportError: No module named 'cv2'

```bash
pip install opencv-python
```

#### 2. MediaPipe Installation Fails

Try installing with specific version:
```bash
pip install mediapipe==0.10.0
```

#### 3. PyTorch CUDA Not Working

Reinstall PyTorch with correct CUDA version:
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 4. Memory Error During Processing

Reduce batch size or video resolution in config:
```yaml
video:
  resolution: [1280, 720]  # Reduce from [1920, 1080]
```

#### 5. ffmpeg Not Found

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

---

## Platform-Specific Notes

### Windows

1. Install Microsoft Visual C++ Redistributable
2. Use PowerShell or Command Prompt as administrator
3. May need to install Windows Build Tools:
   ```bash
   npm install --global windows-build-tools
   ```

### macOS

1. Install Xcode Command Line Tools:
   ```bash
   xcode-select --install
   ```

2. For Apple Silicon (M1/M2), use conda for better compatibility:
   ```bash
   conda install pytorch torchvision -c pytorch
   ```

### Linux

1. Install system dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-dev build-essential libgl1-mesa-glx
   ```

2. For headless servers, install virtual display:
   ```bash
   sudo apt-get install xvfb
   ```

---

## Verification Test

Create a test script `test_installation.py`:

```python
#!/usr/bin/env python3

import sys
print("Testing Motion OS installation...\n")

# Test imports
try:
    import motion_os
    print("✓ Motion OS imported successfully")
except ImportError as e:
    print(f"✗ Failed to import Motion OS: {e}")
    sys.exit(1)

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("✗ PyTorch not installed")

try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__}")
except ImportError:
    print("✗ OpenCV not installed")

try:
    import mediapipe
    print(f"✓ MediaPipe {mediapipe.__version__}")
except ImportError:
    print("⚠ MediaPipe not installed (fallback pose estimation unavailable)")

print("\n✓ Installation verification complete!")
```

Run test:
```bash
python test_installation.py
```

---

## Next Steps

After successful installation:

1. Read [USAGE.md](USAGE.md) for usage instructions
2. Review [API.md](API.md) for API documentation
3. Check [EXAMPLES.md](EXAMPLES.md) for example workflows
4. Try processing a sample video

---

## Getting Help

If you encounter issues:

1. Check [FAQ.md](FAQ.md) for common questions
2. Search [GitHub Issues](https://github.com/yourusername/Motion-OS-Claude/issues)
3. Create a new issue with:
   - OS and Python version
   - Error message and full traceback
   - Steps to reproduce
