"""
Output Generator
Handles all output generation: CSV, JSON, graphs, and videos
"""

import os
import cv2
import json
import numpy as np
import pandas as pd
from typing import Dict
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.visualization import (
    SkeletonVisualizer,
    plot_joint_angles,
    plot_gait_parameters,
    plot_3d_trajectory,
    create_comparison_plot
)
from utils.smpl_utils import SMPL_JOINT_NAMES


class OutputGenerator:
    """
    Generates all output artifacts:
    - CSV and JSON data files
    - Visualization graphs
    - Annotated video with skeleton overlay
    """

    def __init__(self, config: Dict, output_dir: str):
        """
        Initialize Output Generator

        Args:
            config: Configuration dictionary
            output_dir: Output directory path
        """
        self.config = config
        self.output_dir = output_dir

        # Create output subdirectories
        self.data_dir = os.path.join(output_dir, 'data')
        self.graphs_dir = os.path.join(output_dir, 'graphs')
        self.videos_dir = os.path.join(output_dir, 'videos')

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.graphs_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)

        # Initialize visualizer
        self.visualizer = SkeletonVisualizer(SMPL_JOINT_NAMES)

        print(f"[OutputGenerator] Initialized. Output directory: {output_dir}")

    def generate_all_outputs(
        self,
        video_name: str,
        frames: np.ndarray,
        joints_3d_original: np.ndarray,
        joints_3d_refined: np.ndarray,
        gait_analysis: Dict,
        refinement_metrics: Dict,
        video_metadata: Dict,
        fps: float = 30.0
    ):
        """
        Generate all output artifacts

        Args:
            video_name: Name of the input video
            frames: Video frames
            joints_3d_original: Original joint positions
            joints_3d_refined: Refined joint positions
            gait_analysis: Gait analysis results
            refinement_metrics: Physics refinement metrics
            video_metadata: Video metadata
            fps: Frame rate
        """
        print("\n" + "="*60)
        print("GENERATING OUTPUTS")
        print("="*60)

        base_name = os.path.splitext(video_name)[0]

        # 1. Save data files
        if self.config['output']['save_csv']:
            self._save_csv(base_name, joints_3d_refined, gait_analysis, fps)

        if self.config['output']['save_json']:
            self._save_json(base_name, joints_3d_refined, gait_analysis, refinement_metrics, video_metadata)

        if self.config['output']['save_npz']:
            self._save_npz(base_name, joints_3d_original, joints_3d_refined, gait_analysis)

        # 2. Generate graphs
        if self.config['output']['generate_graphs']:
            self._generate_graphs(base_name, joints_3d_original, joints_3d_refined, gait_analysis)

        # 3. Generate annotated video
        if self.config['output']['generate_video']:
            self._generate_video(base_name, frames, joints_3d_refined, gait_analysis, fps)

        print("\n" + "="*60)
        print("OUTPUT GENERATION COMPLETE")
        print("="*60 + "\n")

    def _save_csv(
        self,
        base_name: str,
        joints_3d: np.ndarray,
        gait_analysis: Dict,
        fps: float
    ):
        """Save joint positions and gait parameters to CSV"""
        print("[OutputGenerator] Saving CSV files...")

        # Joint positions CSV
        T, J, _ = joints_3d.shape
        rows = []

        for t in range(T):
            row = {'frame': t, 'time': t / fps}

            for j in range(J):
                joint_name = SMPL_JOINT_NAMES[j]
                row[f'{joint_name}_x'] = joints_3d[t, j, 0]
                row[f'{joint_name}_y'] = joints_3d[t, j, 1]
                row[f'{joint_name}_z'] = joints_3d[t, j, 2]

            rows.append(row)

        df_joints = pd.DataFrame(rows)
        joints_csv_path = os.path.join(self.data_dir, f'{base_name}_joints.csv')
        df_joints.to_csv(joints_csv_path, index=False)
        print(f"  Saved: {joints_csv_path}")

        # Gait parameters CSV
        gait_params = gait_analysis['spatio_temporal']
        gait_dict = {
            'Parameter': [],
            'Value': [],
            'Unit': []
        }

        gait_dict['Parameter'].append('Average Velocity')
        gait_dict['Value'].append(gait_params['velocity_mean'])
        gait_dict['Unit'].append('m/s')

        gait_dict['Parameter'].append('Cadence')
        gait_dict['Value'].append(gait_params['cadence'])
        gait_dict['Unit'].append('steps/min')

        gait_dict['Parameter'].append('Stride Length (Left)')
        gait_dict['Value'].append(gait_params['stride_length_left_mean'])
        gait_dict['Unit'].append('m')

        gait_dict['Parameter'].append('Stride Length (Right)')
        gait_dict['Value'].append(gait_params['stride_length_right_mean'])
        gait_dict['Unit'].append('m')

        df_gait = pd.DataFrame(gait_dict)
        gait_csv_path = os.path.join(self.data_dir, f'{base_name}_gait_parameters.csv')
        df_gait.to_csv(gait_csv_path, index=False)
        print(f"  Saved: {gait_csv_path}")

    def _save_json(
        self,
        base_name: str,
        joints_3d: np.ndarray,
        gait_analysis: Dict,
        refinement_metrics: Dict,
        video_metadata: Dict
    ):
        """Save complete analysis results to JSON"""
        print("[OutputGenerator] Saving JSON file...")

        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        output_data = {
            'metadata': video_metadata,
            'joints_3d': convert_to_serializable(joints_3d),
            'gait_analysis': convert_to_serializable(gait_analysis),
            'refinement_metrics': convert_to_serializable(refinement_metrics)
        }

        json_path = os.path.join(self.data_dir, f'{base_name}_analysis.json')

        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"  Saved: {json_path}")

    def _save_npz(
        self,
        base_name: str,
        joints_3d_original: np.ndarray,
        joints_3d_refined: np.ndarray,
        gait_analysis: Dict
    ):
        """Save numpy arrays in compressed format"""
        print("[OutputGenerator] Saving NPZ file...")

        npz_path = os.path.join(self.data_dir, f'{base_name}_data.npz')

        np.savez_compressed(
            npz_path,
            joints_3d_original=joints_3d_original,
            joints_3d_refined=joints_3d_refined,
            **{k: v for k, v in gait_analysis.items() if isinstance(v, np.ndarray)}
        )

        print(f"  Saved: {npz_path}")

    def _generate_graphs(
        self,
        base_name: str,
        joints_3d_original: np.ndarray,
        joints_3d_refined: np.ndarray,
        gait_analysis: Dict
    ):
        """Generate all visualization graphs"""
        print("[OutputGenerator] Generating graphs...")

        # 1. Joint angle plots for each joint
        joint_angles = gait_analysis['joint_angles']

        for joint_name, joint_data in joint_angles.items():
            graph_path = os.path.join(self.graphs_dir, f'{base_name}_{joint_name}_angles.png')
            plot_joint_angles(joint_data['angles'], joint_name, graph_path)
            print(f"  Generated: {graph_path}")

        # 2. Gait parameters plot
        gait_params_path = os.path.join(self.graphs_dir, f'{base_name}_gait_parameters.png')
        plot_gait_parameters(gait_analysis['spatio_temporal'], gait_params_path)
        print(f"  Generated: {gait_params_path}")

        # 3. 3D trajectory plot (interactive)
        trajectory_path = os.path.join(self.graphs_dir, f'{base_name}_3d_trajectory.html')
        foot_indices = [10, 11]  # left_foot, right_foot
        plot_3d_trajectory(joints_3d_refined, foot_indices, trajectory_path)
        print(f"  Generated: {trajectory_path}")

        # 4. Comparison plots (original vs refined)
        # Example: Compare pelvis height
        pelvis_idx = 0
        pelvis_height_original = joints_3d_original[:, pelvis_idx, 1]
        pelvis_height_refined = joints_3d_refined[:, pelvis_idx, 1]

        comparison_path = os.path.join(self.graphs_dir, f'{base_name}_refinement_comparison.png')
        create_comparison_plot(
            pelvis_height_original,
            pelvis_height_refined,
            'Pelvis Height (m)',
            comparison_path
        )
        print(f"  Generated: {comparison_path}")

    def _generate_video(
        self,
        base_name: str,
        frames: np.ndarray,
        joints_3d: np.ndarray,
        gait_analysis: Dict,
        fps: float
    ):
        """Generate annotated video with skeleton overlay"""
        print("[OutputGenerator] Generating annotated video...")

        output_path = os.path.join(self.videos_dir, f'{base_name}_analysis.mp4')

        # Video writer
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self.config['output']['video_codec'])
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Get camera matrix (simplified - assuming centered perspective)
        camera_matrix = np.array([
            [width, 0, width/2],
            [0, width, height/2],
            [0, 0, 1]
        ])

        # Spatio-temporal parameters for overlay
        gait_params = gait_analysis['spatio_temporal']

        for frame_idx, frame in enumerate(frames):
            # Get joints for this frame
            joints_frame = joints_3d[frame_idx]

            # Project 3D to 2D (simplified - assuming joints are already in camera space)
            # For a more accurate projection, use proper camera parameters
            joints_2d = self._simple_projection(joints_frame, width, height)

            # Draw skeleton
            if self.config['output']['overlay_skeleton']:
                frame = self.visualizer.draw_skeleton_on_frame(frame, joints_2d)

            # Add metrics overlay
            if self.config['output']['show_metrics']:
                metrics = {
                    'Velocity': f"{gait_params['velocity_mean']:.2f} m/s",
                    'Cadence': f"{gait_params['cadence']:.1f} steps/min"
                }
                frame = self.visualizer.add_metrics_overlay(frame, metrics, frame_idx)

            out.write(frame)

        out.release()
        print(f"  Generated: {output_path}")

    def _simple_projection(self, joints_3d: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Simple orthographic projection for visualization

        Args:
            joints_3d: (J, 3) 3D joint positions
            width: Frame width
            height: Frame height

        Returns:
            (J, 2) 2D joint positions
        """
        # Normalize to frame dimensions
        # Assume joints are in metric space, scale to pixels

        # Find bounding box
        x_min, x_max = joints_3d[:, 0].min(), joints_3d[:, 0].max()
        y_min, y_max = joints_3d[:, 1].min(), joints_3d[:, 1].max()

        # Scale to fit in frame (with padding)
        padding = 0.1
        scale_x = width * (1 - 2*padding) / (x_max - x_min + 1e-8)
        scale_y = height * (1 - 2*padding) / (y_max - y_min + 1e-8)
        scale = min(scale_x, scale_y)

        # Project
        joints_2d = np.zeros((len(joints_3d), 2))
        joints_2d[:, 0] = (joints_3d[:, 0] - x_min) * scale + width * padding
        joints_2d[:, 1] = height - ((joints_3d[:, 1] - y_min) * scale + height * padding)  # Flip Y

        return joints_2d
