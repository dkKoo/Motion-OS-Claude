"""
Visualization utilities for 3D skeleton and analysis results
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation
import seaborn as sns


# Skeleton connections for visualization
SKELETON_CONNECTIONS = [
    # Spine
    ('pelvis', 'spine1'),
    ('spine1', 'spine2'),
    ('spine2', 'spine3'),
    ('spine3', 'neck'),
    ('neck', 'head'),

    # Left leg
    ('pelvis', 'left_hip'),
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
    ('left_ankle', 'left_foot'),

    # Right leg
    ('pelvis', 'right_hip'),
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),
    ('right_ankle', 'right_foot'),

    # Left arm
    ('spine3', 'left_collar'),
    ('left_collar', 'left_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    ('left_wrist', 'left_hand'),

    # Right arm
    ('spine3', 'right_collar'),
    ('right_collar', 'right_shoulder'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    ('right_wrist', 'right_hand'),
]


class SkeletonVisualizer:
    """Visualize 3D skeleton on video frames"""

    def __init__(self, joint_names):
        self.joint_names = joint_names
        self.colors = {
            'left': (0, 255, 0),   # Green for left side
            'right': (255, 0, 0),  # Blue for right side
            'center': (0, 0, 255)  # Red for center
        }

    def get_joint_index(self, joint_name):
        """Get index of joint by name"""
        return self.joint_names.index(joint_name)

    def project_3d_to_2d(self, joints_3d, camera_matrix):
        """
        Project 3D joints to 2D image coordinates

        Args:
            joints_3d: (J, 3) 3D joint positions
            camera_matrix: (3, 3) camera intrinsic matrix

        Returns:
            (J, 2) 2D joint positions
        """
        # Simple perspective projection
        # Assume camera is at origin looking down +Z
        joints_2d = joints_3d[:, :2] / (joints_3d[:, 2:3] + 1e-8)

        # Apply camera matrix
        joints_2d_homogeneous = np.hstack([joints_2d, np.ones((len(joints_2d), 1))])
        joints_2d_projected = (camera_matrix @ joints_2d_homogeneous.T).T

        return joints_2d_projected[:, :2]

    def draw_skeleton_on_frame(self, frame, joints_2d, confidence=None):
        """
        Draw skeleton on video frame

        Args:
            frame: Video frame (H, W, 3)
            joints_2d: (J, 2) 2D joint positions
            confidence: Optional (J,) confidence scores

        Returns:
            Frame with skeleton overlay
        """
        frame_copy = frame.copy()

        # Draw connections
        for joint_start, joint_end in SKELETON_CONNECTIONS:
            idx_start = self.get_joint_index(joint_start)
            idx_end = self.get_joint_index(joint_end)

            pt1 = tuple(joints_2d[idx_start].astype(int))
            pt2 = tuple(joints_2d[idx_end].astype(int))

            # Determine color based on side
            if 'left' in joint_start or 'left' in joint_end:
                color = self.colors['left']
            elif 'right' in joint_start or 'right' in joint_end:
                color = self.colors['right']
            else:
                color = self.colors['center']

            cv2.line(frame_copy, pt1, pt2, color, 2)

        # Draw joints
        for i, joint in enumerate(joints_2d):
            pt = tuple(joint.astype(int))

            # Color based on confidence
            if confidence is not None:
                color_intensity = int(255 * confidence[i])
                color = (color_intensity, color_intensity, 0)
            else:
                color = (255, 255, 0)  # Yellow

            cv2.circle(frame_copy, pt, 4, color, -1)

        return frame_copy

    def add_metrics_overlay(self, frame, metrics, frame_idx):
        """
        Add text overlay with metrics

        Args:
            frame: Video frame
            metrics: Dictionary of metrics to display
            frame_idx: Current frame index

        Returns:
            Frame with metrics overlay
        """
        frame_copy = frame.copy()

        # Add semi-transparent background
        overlay = frame_copy.copy()
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame_copy, 0.4, 0, frame_copy)

        # Add text
        y_offset = 40
        cv2.putText(frame_copy, f"Frame: {frame_idx}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        y_offset += 30
        for key, value in metrics.items():
            text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
            cv2.putText(frame_copy, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25

        return frame_copy


def plot_joint_angles(angles_data, joint_name, output_path):
    """
    Plot joint angles over time for all planes

    Args:
        angles_data: Dictionary with 'sagittal', 'coronal', 'transverse' keys
        joint_name: Name of the joint
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    planes = ['sagittal', 'coronal', 'transverse']
    titles = [
        f'{joint_name} - Sagittal Plane (Flexion/Extension)',
        f'{joint_name} - Coronal Plane (Abduction/Adduction)',
        f'{joint_name} - Transverse Plane (Rotation)'
    ]

    for ax, plane, title in zip(axes, planes, titles):
        if plane in angles_data:
            data = angles_data[plane]
            frames = np.arange(len(data))

            ax.plot(frames, data, linewidth=2, label=plane.capitalize())
            ax.set_xlabel('Frame')
            ax.set_ylabel('Angle (degrees)')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_gait_parameters(gait_data, output_path):
    """
    Plot gait cycle parameters

    Args:
        gait_data: Dictionary with gait cycle information
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Stride length
    if 'stride_lengths' in gait_data:
        axes[0, 0].plot(gait_data['stride_lengths'], marker='o')
        axes[0, 0].set_title('Stride Length')
        axes[0, 0].set_xlabel('Cycle')
        axes[0, 0].set_ylabel('Length (m)')
        axes[0, 0].grid(True, alpha=0.3)

    # Velocity
    if 'velocities' in gait_data:
        axes[0, 1].plot(gait_data['velocities'], marker='o', color='orange')
        axes[0, 1].set_title('Gait Velocity')
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Velocity (m/s)')
        axes[0, 1].grid(True, alpha=0.3)

    # Cadence
    if 'cadence' in gait_data:
        axes[1, 0].bar(['Cadence'], [gait_data['cadence']], color='green')
        axes[1, 0].set_title('Cadence')
        axes[1, 0].set_ylabel('Steps/min')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Cycle duration
    if 'cycle_durations' in gait_data:
        axes[1, 1].plot(gait_data['cycle_durations'], marker='o', color='red')
        axes[1, 1].set_title('Gait Cycle Duration')
        axes[1, 1].set_xlabel('Cycle')
        axes[1, 1].set_ylabel('Duration (s)')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_3d_trajectory(joints_3d, joint_indices, output_path):
    """
    Create interactive 3D trajectory plot

    Args:
        joints_3d: (T, J, 3) joint positions
        joint_indices: List of joint indices to plot
        output_path: Path to save the HTML plot
    """
    fig = go.Figure()

    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for i, joint_idx in enumerate(joint_indices):
        trajectory = joints_3d[:, joint_idx, :]

        fig.add_trace(go.Scatter3d(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            z=trajectory[:, 2],
            mode='lines+markers',
            name=f'Joint {joint_idx}',
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=3)
        ))

    fig.update_layout(
        title='3D Joint Trajectories',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        ),
        width=1000,
        height=800
    )

    fig.write_html(output_path)


def create_comparison_plot(original_data, refined_data, metric_name, output_path):
    """
    Create comparison plot between original and refined data

    Args:
        original_data: Original data array
        refined_data: Refined data array
        metric_name: Name of the metric
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))

    frames = np.arange(len(original_data))

    plt.plot(frames, original_data, label='Original', alpha=0.7, linewidth=2)
    plt.plot(frames, refined_data, label='Refined (Physics-Informed)',
            alpha=0.7, linewidth=2)

    plt.xlabel('Frame')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name}: Original vs Physics-Informed Refinement')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
