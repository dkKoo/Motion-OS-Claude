# Motion OS Phase 1 🚀

## Physics-Informed 3D Gait Analysis System

Motion OS is a cutting-edge biomechanical analysis system that transforms video recordings into precise, physics-validated gait analysis reports. Unlike traditional AI-based pose estimation that produces "guesses," Motion OS applies **physics constraints** to ensure biomechanically-accurate results.

---

## 🎯 Key Innovation: Physics-Informed Refinement

The core breakthrough of Motion OS is applying **physical laws** to filter AI predictions:

### Three Critical Constraints

1. **Bone Length Constraint** 🦴
   - All bone segments maintain constant length across frames
   - Eliminates unrealistic bone deformations
   - L2 loss optimization ensures anatomical accuracy

2. **Zero Velocity Update (ZUPT)** 👣
   - Feet have zero velocity when in contact with ground
   - Prevents foot sliding artifacts
   - Uses foot-contact detection from WHAM

3. **Signal Smoothing** 📊
   - Savitzky-Golay filtering removes jittering
   - Preserves motion dynamics while eliminating noise
   - Applied pre- and post-optimization

---

## 🏗️ System Architecture

```
[Input Video] → [Module A] → [Module B] → [Module C] → [Module D] → [Outputs]
                    ↓            ↓            ↓            ↓
                 3D Pose    Coordinate   Physics      Gait
                Estimation  Transform   Refinement   Analysis
```

### Pipeline Stages

**Stage 1: Video Processing & 3D Pose Estimation**
- Uses WHAM (World-grounded Humans with Accurate Motion)
- Extracts SMPL/SMPL-X mesh data
- Detects foot-ground contact
- Fallback to MediaPipe if WHAM unavailable

**Stage 2: Coordinate Transformation**
- Converts normalized coordinates to metric (meters)
- Camera calibration and perspective correction
- Height-based scaling using reference measurements

**Stage 3: Physics-Informed Refinement** ⭐ *Critical Stage*
- Applies bone length constraints
- Implements Zero Velocity Update (ZUPT)
- Smooths trajectories
- Optimizes with PyTorch for GPU acceleration

**Stage 4: Biomechanical Analysis**
- Gait cycle detection (Heel Strike, Toe Off)
- Spatio-temporal parameters (velocity, stride length, cadence)
- Joint kinematics (ROM in sagittal, coronal, transverse planes)

---

## 📊 Outputs

### Data Files
- **CSV**: Joint positions, gait parameters
- **JSON**: Complete analysis results
- **NPZ**: Compressed numpy arrays

### Visualizations
- **Joint angle graphs**: ROM plots for hip, knee, ankle
- **Gait parameter plots**: Stride length, velocity, cadence
- **3D trajectory plots**: Interactive HTML visualizations
- **Comparison plots**: Before/after physics refinement

### Annotated Video
- 3D skeleton overlay on original video
- Real-time gait metrics display
- Color-coded left/right limbs

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Motion-OS-Claude.git
cd Motion-OS-Claude

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Motion OS
pip install -e .
```

### Basic Usage

```bash
# Process a video
python -m motion_os.core.pipeline path/to/video.mp4 --output results/

# Or use the installed command
motion-os path/to/video.mp4 --output results/
```

### Python API

```python
from motion_os import MotionOSPipeline

# Initialize pipeline
pipeline = MotionOSPipeline()

# Process video
results = pipeline.process_video('video.mp4', output_dir='results/')

# Access results
joints_3d = results['joints_3d_refined']
gait_analysis = results['gait_analysis']
```

---

## 📋 Requirements

### Core Dependencies
- Python 3.10+
- PyTorch 2.0+
- OpenCV
- NumPy, SciPy
- MediaPipe (fallback pose estimator)

### Optional (for full functionality)
- WHAM model checkpoint
- SAM 3D Body model checkpoint
- CUDA-capable GPU (recommended for faster processing)

---

## 📁 Project Structure

```
Motion-OS-Claude/
├── motion_os/
│   ├── core/
│   │   ├── pipeline.py          # Main orchestrator
│   │   └── output_generator.py  # Output generation
│   ├── modules/
│   │   ├── video_processor/     # Module A: Video & Pose
│   │   ├── coordinate_transformer/  # Module B: Coord Transform
│   │   ├── biomechanical_filter/    # Module C: Physics Refinement ⭐
│   │   └── gait_analysis/       # Module D: Gait Analysis
│   ├── utils/
│   │   ├── smpl_utils.py        # SMPL utilities
│   │   └── visualization.py     # Visualization tools
│   └── config/
│       └── config.yaml          # Configuration file
├── outputs/                     # Generated outputs
│   ├── data/                    # CSV, JSON, NPZ files
│   ├── graphs/                  # Visualization plots
│   └── videos/                  # Annotated videos
├── requirements.txt
├── setup.py
└── README.md
```

---

## ⚙️ Configuration

Edit `motion_os/config/config.yaml` to customize:

```yaml
# Physics constraints
physics:
  bone_constraint:
    enabled: true
    weight: 10.0      # Bone length constraint weight

  zupt:
    enabled: true
    weight: 5.0       # Zero velocity update weight

  smoothing:
    enabled: true
    method: "savgol"  # Savitzky-Golay filter
    window_length: 11
    polyorder: 3

# Reference height for scaling
coordinate_transform:
  reference_height_m: 1.70  # Average human height
```

---

## 🔬 Scientific Background

### Gait Analysis Metrics

**Spatio-temporal Parameters:**
- **Velocity**: Forward walking speed (m/s)
- **Stride Length**: Distance between successive heel strikes (m)
- **Cadence**: Steps per minute
- **Cycle Duration**: Time for one gait cycle (s)

**Joint Kinematics:**
- **Sagittal Plane**: Flexion/Extension
- **Coronal Plane**: Abduction/Adduction
- **Transverse Plane**: Internal/External Rotation

**Gait Cycle Phases:**
- **Heel Strike**: Initial foot contact with ground
- **Stance Phase**: Foot in contact with ground
- **Toe Off**: Foot leaving ground
- **Swing Phase**: Foot in air

---

## 🎯 Use Cases

- **Clinical Gait Analysis**: Assess walking patterns for rehabilitation
- **Sports Biomechanics**: Optimize athletic performance
- **Ergonomics Research**: Study workplace movement patterns
- **Animation & VFX**: Capture realistic human motion
- **Wearable Validation**: Validate sensor-based gait metrics

---

## 🤝 Contributing

We welcome contributions! Areas for enhancement:

1. **Model Integration**: Add support for more 3D pose models
2. **Real-time Processing**: Optimize for live video streams
3. **Additional Metrics**: Implement more biomechanical parameters
4. **Multi-person Support**: Track multiple subjects simultaneously
5. **UI/UX**: Build web interface for easier usage

---

## 📚 References

### Key Papers

1. **WHAM**: *World-grounded Humans with Accurate Motion*
   - Global coordinate estimation
   - Foot-contact detection

2. **SMPL/SMPL-X**: Body models for 3D human shape
   - Parametric body representation
   - Joint locations and skinning

3. **Zero Velocity Update (ZUPT)**:
   - Inertial navigation technique
   - Foot-mounted IMU systems

### Physics-Informed Machine Learning
- Combining data-driven predictions with physical constraints
- Ensures outputs respect fundamental laws (constant bone length, etc.)

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- WHAM team for 3D pose estimation model
- SMPL/SMPL-X developers for body models
- MediaPipe team for fallback pose estimation
- Open-source community for foundational libraries

---

## 📧 Contact

For questions, issues, or collaborations:
- **Issues**: [GitHub Issues](https://github.com/yourusername/Motion-OS-Claude/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/Motion-OS-Claude/discussions)

---

## 🎓 Citation

If you use Motion OS in your research, please cite:

```bibtex
@software{motion_os_2024,
  title={Motion OS: Physics-Informed 3D Gait Analysis System},
  author={Motion OS Team},
  year={2024},
  url={https://github.com/yourusername/Motion-OS-Claude}
}
```

---

**Built with ❤️ for accurate, physics-validated biomechanical analysis**
