"""
Motion OS Pipeline
Main orchestrator that integrates all modules
"""

import os
import yaml
import time
from typing import Dict, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from modules.video_processor import VideoProcessor
from modules.coordinate_transformer import CoordinateTransformer
from modules.biomechanical_filter import BiomechanicalFilter
from modules.gait_analysis import GaitAnalyzer
from core.output_generator import OutputGenerator


class MotionOSPipeline:
    """
    Motion OS Phase 1 Pipeline

    Orchestrates the 4-stage process:
    1. Video Processing (3D Pose Estimation)
    2. Coordinate Transformation (Metric Conversion)
    3. Physics-Informed Refinement (Biomechanical Filtering)
    4. Gait Analysis (Biomechanical Analytics)
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Motion OS Pipeline

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'config',
                'config.yaml'
            )

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print("\n" + "="*60)
        print("MOTION OS PHASE 1")
        print("Physics-Informed 3D Gait Analysis System")
        print("="*60 + "\n")

        # Initialize modules
        print("Initializing modules...")

        self.video_processor = VideoProcessor(self.config)
        self.coordinate_transformer = CoordinateTransformer(self.config)
        self.biomechanical_filter = BiomechanicalFilter(self.config)
        self.gait_analyzer = GaitAnalyzer(self.config)

        print("\nAll modules initialized successfully!\n")

    def process_video(self, video_path: str, output_dir: str = 'outputs') -> Dict:
        """
        Process video through complete pipeline

        Args:
            video_path: Path to input video
            output_dir: Output directory for results

        Returns:
            Dictionary containing all analysis results
        """
        start_time = time.time()

        print("\n" + "="*60)
        print(f"PROCESSING: {os.path.basename(video_path)}")
        print("="*60 + "\n")

        # Stage 1: Video Processing & 3D Pose Estimation
        print("\n" + "-"*60)
        print("STAGE 1: VIDEO PROCESSING & 3D POSE ESTIMATION")
        print("-"*60)

        video_data = self.video_processor.process_video(video_path)

        joints_3d_raw = video_data['joints_3d']
        foot_contact = video_data['foot_contact']
        frames = video_data['frames']
        metadata = video_data['metadata']

        fps = metadata['fps']

        print(f"\n✓ Stage 1 complete: Extracted {len(joints_3d_raw)} poses")

        # Stage 2: Coordinate Transformation
        print("\n" + "-"*60)
        print("STAGE 2: COORDINATE TRANSFORMATION")
        print("-"*60)

        joints_3d_metric = self.coordinate_transformer.transform_coordinates(
            joints_3d_raw,
            metadata
        )

        print(f"\n✓ Stage 2 complete: Coordinates transformed to metric space")

        # Stage 3: Physics-Informed Refinement
        print("\n" + "-"*60)
        print("STAGE 3: PHYSICS-INFORMED REFINEMENT")
        print("-"*60)

        refinement_result = self.biomechanical_filter.refine_poses(
            joints_3d_metric,
            foot_contact,
            fps
        )

        joints_3d_refined = refinement_result['refined_joints']
        refinement_metrics = refinement_result['metrics']

        print(f"\n✓ Stage 3 complete: Poses refined with physics constraints")

        # Stage 4: Gait Analysis
        print("\n" + "-"*60)
        print("STAGE 4: BIOMECHANICAL GAIT ANALYSIS")
        print("-"*60)

        gait_analysis = self.gait_analyzer.analyze_gait(
            joints_3d_refined,
            fps
        )

        print(f"\n✓ Stage 4 complete: Gait analysis performed")

        # Generate outputs
        output_generator = OutputGenerator(self.config, output_dir)

        output_generator.generate_all_outputs(
            video_name=os.path.basename(video_path),
            frames=frames,
            joints_3d_original=joints_3d_metric,
            joints_3d_refined=joints_3d_refined,
            gait_analysis=gait_analysis,
            refinement_metrics=refinement_metrics,
            video_metadata=metadata,
            fps=fps
        )

        # Summary
        elapsed_time = time.time() - start_time

        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"\nTotal processing time: {elapsed_time:.2f} seconds")
        print(f"Output directory: {output_dir}")

        # Print key results
        print("\n" + "-"*60)
        print("KEY RESULTS")
        print("-"*60)

        print("\nGait Parameters:")
        st_params = gait_analysis['spatio_temporal']
        print(f"  • Average velocity: {st_params['velocity_mean']:.3f} m/s")
        print(f"  • Cadence: {st_params['cadence']:.1f} steps/min")
        print(f"  • Stride length (L): {st_params['stride_length_left_mean']:.3f} m")
        print(f"  • Stride length (R): {st_params['stride_length_right_mean']:.3f} m")
        print(f"  • Gait cycles: {st_params['num_cycles_left']} left, {st_params['num_cycles_right']} right")

        print("\nJoint ROM (Range of Motion):")
        for joint_name, joint_data in gait_analysis['joint_angles'].items():
            rom = joint_data['rom']['sagittal']
            print(f"  • {joint_name.capitalize()}: {rom['range']:.1f}° "
                  f"[{rom['min']:.1f}° to {rom['max']:.1f}°]")

        print("\nPhysics Refinement Quality:")
        print(f"  • Bone length variance: "
              f"{refinement_metrics.get('bone_length_variance_original', 0):.6f} → "
              f"{refinement_metrics.get('bone_length_variance_refined', 0):.6f}")

        print("\n" + "="*60 + "\n")

        return {
            'joints_3d_original': joints_3d_metric,
            'joints_3d_refined': joints_3d_refined,
            'gait_analysis': gait_analysis,
            'refinement_metrics': refinement_metrics,
            'metadata': metadata
        }

    def batch_process(self, video_paths: list, output_dir: str = 'outputs') -> list:
        """
        Process multiple videos

        Args:
            video_paths: List of video paths
            output_dir: Output directory

        Returns:
            List of results for each video
        """
        results = []

        for i, video_path in enumerate(video_paths):
            print(f"\n{'='*60}")
            print(f"BATCH PROCESSING: Video {i+1}/{len(video_paths)}")
            print(f"{'='*60}\n")

            result = self.process_video(video_path, output_dir)
            results.append(result)

        return results


def main():
    """Main entry point for CLI usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Motion OS Phase 1: Physics-Informed 3D Gait Analysis'
    )

    parser.add_argument(
        'video',
        type=str,
        help='Path to input video file'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (optional)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Output directory (default: outputs)'
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = MotionOSPipeline(config_path=args.config)

    # Process video
    pipeline.process_video(args.video, args.output)


if __name__ == '__main__':
    main()
