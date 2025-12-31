#!/usr/bin/env python3
"""
Motion OS - Advanced Usage Example

This example demonstrates advanced features:
- Custom configuration
- Batch processing
- Custom analysis
"""

import sys
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from motion_os import MotionOSPipeline


def custom_configuration_example():
    """Example: Using custom configuration"""

    print("\n" + "="*60)
    print("EXAMPLE 1: Custom Configuration")
    print("="*60 + "\n")

    # Load default config
    config_path = "motion_os/config/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Modify configuration
    config['physics']['bone_constraint']['weight'] = 15.0  # Stricter bone constraint
    config['physics']['zupt']['weight'] = 8.0  # Stronger ZUPT
    config['coordinate_transform']['reference_height_m'] = 1.75  # Taller reference

    # Save custom config
    custom_config_path = "examples/custom_config.yaml"
    with open(custom_config_path, 'w') as f:
        yaml.dump(config, f)

    # Initialize with custom config
    pipeline = MotionOSPipeline(config_path=custom_config_path)

    print("Pipeline initialized with custom configuration:")
    print(f"  - Bone constraint weight: {config['physics']['bone_constraint']['weight']}")
    print(f"  - ZUPT weight: {config['physics']['zupt']['weight']}")
    print(f"  - Reference height: {config['coordinate_transform']['reference_height_m']} m")


def batch_processing_example():
    """Example: Batch processing multiple videos"""

    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Processing")
    print("="*60 + "\n")

    pipeline = MotionOSPipeline()

    # List of videos to process
    video_paths = [
        "data/sample_videos/walking_1.mp4",
        "data/sample_videos/walking_2.mp4",
        "data/sample_videos/running.mp4"
    ]

    # Filter existing videos
    existing_videos = [v for v in video_paths if os.path.exists(v)]

    if not existing_videos:
        print("No sample videos found. Skipping batch processing example.")
        return

    # Process all videos
    results = pipeline.batch_process(
        video_paths=existing_videos,
        output_dir="outputs/batch_example"
    )

    # Compare results
    print("\nComparison across videos:")
    print(f"{'Video':<25} {'Velocity (m/s)':<15} {'Cadence (steps/min)':<20}")
    print("-" * 60)

    for i, (video_path, result) in enumerate(zip(existing_videos, results)):
        video_name = os.path.basename(video_path)
        st_params = result['gait_analysis']['spatio_temporal']
        print(f"{video_name:<25} {st_params['velocity_mean']:<15.3f} {st_params['cadence']:<20.1f}")


def custom_analysis_example():
    """Example: Custom analysis on results"""

    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Analysis")
    print("="*60 + "\n")

    pipeline = MotionOSPipeline()

    video_path = "data/sample_videos/walking.mp4"

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    # Process video
    results = pipeline.process_video(video_path, output_dir="outputs/custom_analysis")

    # Custom analysis: Analyze step width
    joints_3d = results['joints_3d_refined']
    gait_cycles = results['gait_analysis']['gait_cycles']

    # Get left and right ankle positions
    left_ankle_idx = 7   # SMPL joint index
    right_ankle_idx = 8

    step_widths = []

    # Compute step width at each left heel strike
    for cycle in gait_cycles['left']:
        frame = cycle['start_frame']
        left_pos = joints_3d[frame, left_ankle_idx]
        right_pos = joints_3d[frame, right_ankle_idx]

        # Step width = lateral distance
        step_width = abs(left_pos[0] - right_pos[0])
        step_widths.append(step_width)

    if step_widths:
        avg_step_width = np.mean(step_widths)
        std_step_width = np.std(step_widths)

        print(f"\nCustom Analysis: Step Width")
        print(f"  - Average: {avg_step_width:.3f} m")
        print(f"  - Std Dev: {std_step_width:.3f} m")
        print(f"  - Min: {np.min(step_widths):.3f} m")
        print(f"  - Max: {np.max(step_widths):.3f} m")

        # Plot step width variation
        plt.figure(figsize=(10, 6))
        plt.plot(step_widths, marker='o')
        plt.axhline(avg_step_width, color='r', linestyle='--', label=f'Mean: {avg_step_width:.3f} m')
        plt.xlabel('Gait Cycle')
        plt.ylabel('Step Width (m)')
        plt.title('Step Width Variation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('outputs/custom_analysis/step_width.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n  Plot saved to: outputs/custom_analysis/step_width.png")


def main():
    """Run all examples"""

    print("\n" + "="*60)
    print("MOTION OS - ADVANCED USAGE EXAMPLES")
    print("="*60)

    # Example 1: Custom configuration
    custom_configuration_example()

    # Example 2: Batch processing
    batch_processing_example()

    # Example 3: Custom analysis
    custom_analysis_example()

    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
