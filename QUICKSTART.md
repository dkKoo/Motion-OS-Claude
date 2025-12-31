# Quick Start Guide

Get Motion OS running in 5 minutes!

## Installation (2 minutes)

```bash
# Clone and enter directory
git clone https://github.com/yourusername/Motion-OS-Claude.git
cd Motion-OS-Claude

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt
pip install -e .
```

## Process Your First Video (3 minutes)

### Method 1: Command Line

```bash
# Place your video in data/sample_videos/
cp /path/to/your/walking_video.mp4 data/sample_videos/

# Process it
motion-os data/sample_videos/walking_video.mp4
```

### Method 2: Python Script

Create `test.py`:

```python
from motion_os import MotionOSPipeline

pipeline = MotionOSPipeline()
results = pipeline.process_video('data/sample_videos/walking_video.mp4')

# Print results
gait = results['gait_analysis']['spatio_temporal']
print(f"Velocity: {gait['velocity_mean']:.2f} m/s")
print(f"Cadence: {gait['cadence']:.1f} steps/min")
```

Run:
```bash
python test.py
```

## View Results

Results are saved in `outputs/`:

```
outputs/
├── data/
│   ├── walking_video_joints.csv          # Joint positions
│   ├── walking_video_gait_parameters.csv # Gait metrics
│   └── walking_video_analysis.json       # Complete results
├── graphs/
│   ├── walking_video_hip_angles.png      # Joint angle plots
│   ├── walking_video_knee_angles.png
│   ├── walking_video_ankle_angles.png
│   ├── walking_video_gait_parameters.png # Gait metrics plot
│   └── walking_video_3d_trajectory.html  # Interactive 3D plot
└── videos/
    └── walking_video_analysis.mp4        # Annotated video
```

## Understanding the Output

### Key Metrics

**Velocity** (m/s): Walking speed
- Normal walking: 1.2-1.4 m/s
- Slow: < 1.0 m/s
- Fast: > 1.5 m/s

**Cadence** (steps/min): Step frequency
- Normal: 100-120 steps/min
- Slow: < 90 steps/min
- Fast: > 130 steps/min

**Stride Length** (m): Distance per step
- Normal: 1.3-1.5 m
- Short: < 1.2 m
- Long: > 1.6 m

**Joint ROM** (degrees): Range of motion
- Hip: 40-60° (sagittal)
- Knee: 60-70° (sagittal)
- Ankle: 25-35° (sagittal)

## Customization

Edit `motion_os/config/config.yaml`:

```yaml
# Adjust physics constraints
physics:
  bone_constraint:
    weight: 10.0  # Increase for stricter bone length
  zupt:
    weight: 5.0   # Increase for stricter foot velocity

# Change reference height
coordinate_transform:
  reference_height_m: 1.70  # Your subject's height
```

## Troubleshooting

**"No module named motion_os"**
```bash
pip install -e .
```

**"CUDA out of memory"**
Edit config: `system.device: "cpu"`

**"Video not found"**
Check path: `ls -la data/sample_videos/`

**Poor results**
1. Ensure good video quality (1080p+)
2. Subject fully visible
3. Adjust reference height in config
4. Try different lighting conditions

## Next Steps

1. Read full [README.md](README.md)
2. Check [examples/](examples/) for advanced usage
3. Review [docs/INSTALLATION.md](docs/INSTALLATION.md) for detailed setup
4. Explore configuration options in `config.yaml`

## Need Help?

- [GitHub Issues](https://github.com/yourusername/Motion-OS-Claude/issues)
- [Documentation](docs/)
- [Examples](examples/)

---

**You're all set! Happy analyzing! 🚀**
