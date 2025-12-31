#!/usr/bin/env python3
"""
Motion OS - Basic Usage Example

This example demonstrates how to use Motion OS to analyze a gait video.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from motion_os import MotionOSPipeline


def main():
    """Basic usage example"""

    # Initialize the pipeline
    print("Initializing Motion OS Pipeline...")
    pipeline = MotionOSPipeline()

    # Path to your video file
    video_path = "data/sample_videos/walking.mp4"

    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        print("Please place a sample video in data/sample_videos/")
        return

    # Process the video
    print(f"\nProcessing video: {video_path}")

    results = pipeline.process_video(
        video_path=video_path,
        output_dir="outputs/basic_example"
    )

    # Access results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    # Refined 3D joints
    joints_3d = results['joints_3d_refined']
    print(f"\nRefined 3D Joints Shape: {joints_3d.shape}")
    print(f"  - {joints_3d.shape[0]} frames")
    print(f"  - {joints_3d.shape[1]} joints")
    print(f"  - {joints_3d.shape[2]} dimensions (X, Y, Z)")

    # Gait analysis
    gait_analysis = results['gait_analysis']

    print("\nGait Cycles Detected:")
    print(f"  - Left foot: {len(gait_analysis['gait_cycles']['left'])} cycles")
    print(f"  - Right foot: {len(gait_analysis['gait_cycles']['right'])} cycles")

    print("\nSpatio-temporal Parameters:")
    st_params = gait_analysis['spatio_temporal']
    print(f"  - Average velocity: {st_params['velocity_mean']:.3f} m/s")
    print(f"  - Cadence: {st_params['cadence']:.1f} steps/min")
    print(f"  - Stride length (left): {st_params['stride_length_left_mean']:.3f} m")
    print(f"  - Stride length (right): {st_params['stride_length_right_mean']:.3f} m")

    print("\nJoint Range of Motion (ROM):")
    for joint_name, joint_data in gait_analysis['joint_angles'].items():
        rom = joint_data['rom']['sagittal']
        print(f"  - {joint_name.capitalize()}: {rom['range']:.1f}° "
              f"(min: {rom['min']:.1f}°, max: {rom['max']:.1f}°)")

    # Refinement metrics
    refinement_metrics = results['refinement_metrics']
    print("\nPhysics Refinement Quality:")
    print(f"  - Bone length variance reduced: "
          f"{refinement_metrics.get('bone_length_variance_original', 0):.6f} → "
          f"{refinement_metrics.get('bone_length_variance_refined', 0):.6f}")

    print("\n" + "="*60)
    print("Analysis complete! Check outputs/basic_example/ for detailed results.")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
