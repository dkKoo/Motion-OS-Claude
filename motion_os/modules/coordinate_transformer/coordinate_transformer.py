"""
Coordinate Transformer Module
Transforms pixel/normalized coordinates to real-world metric coordinates
"""

import numpy as np
from typing import Dict, Optional, Tuple


class CoordinateTransformer:
    """
    Module B: Coordinate Transformer
    - Converts normalized/pixel coordinates to metric (meters)
    - Handles camera calibration
    - Scales based on reference height
    """

    def __init__(self, config: Dict):
        """
        Initialize Coordinate Transformer

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.reference_height_m = config['coordinate_transform']['reference_height_m']
        self.focal_length = config['coordinate_transform'].get('focal_length', None)
        self.principal_point = config['coordinate_transform'].get('principal_point', None)

        self.scale_factor = None
        self.camera_matrix = None

    def transform_coordinates(
        self,
        joints_3d: np.ndarray,
        video_metadata: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Transform coordinates from normalized/pixel space to metric space

        Args:
            joints_3d: (T, J, 3) joint positions in normalized coordinates
            video_metadata: Optional video metadata for camera calibration

        Returns:
            (T, J, 3) joint positions in metric coordinates (meters)
        """
        print("[CoordinateTransformer] Transforming coordinates to metric space...")

        # Step 1: Estimate scale factor from reference height
        scale_factor = self._estimate_scale_factor(joints_3d)

        # Step 2: Apply scaling
        joints_metric = joints_3d * scale_factor

        # Step 3: Apply camera calibration if available
        if video_metadata is not None:
            joints_metric = self._apply_camera_calibration(joints_metric, video_metadata)

        # Step 4: Set ground plane (lowest foot point = 0)
        joints_metric = self._align_to_ground(joints_metric)

        print(f"[CoordinateTransformer] Scale factor: {scale_factor:.4f}")
        print(f"[CoordinateTransformer] Coordinate range: X=[{joints_metric[:,:,0].min():.2f}, {joints_metric[:,:,0].max():.2f}], "
              f"Y=[{joints_metric[:,:,1].min():.2f}, {joints_metric[:,:,1].max():.2f}], "
              f"Z=[{joints_metric[:,:,2].min():.2f}, {joints_metric[:,:,2].max():.2f}]")

        self.scale_factor = scale_factor

        return joints_metric

    def _estimate_scale_factor(self, joints_3d: np.ndarray) -> float:
        """
        Estimate scale factor based on reference height

        Args:
            joints_3d: (T, J, 3) joint positions

        Returns:
            Scale factor to convert to meters
        """
        # SMPL joint indices
        HEAD_IDX = 15
        LEFT_FOOT_IDX = 10
        RIGHT_FOOT_IDX = 11

        # Compute height for each frame
        heights = []
        for frame_joints in joints_3d:
            head_y = frame_joints[HEAD_IDX, 1]
            left_foot_y = frame_joints[LEFT_FOOT_IDX, 1]
            right_foot_y = frame_joints[RIGHT_FOOT_IDX, 1]

            # Use lowest foot point
            foot_y = min(left_foot_y, right_foot_y)

            height = abs(head_y - foot_y)
            heights.append(height)

        # Use median height for robustness
        median_height = np.median(heights)

        if median_height < 1e-6:
            print("[CoordinateTransformer] Warning: Invalid height detected, using default scale")
            return 1.0

        # Scale factor to match reference height
        scale_factor = self.reference_height_m / median_height

        return scale_factor

    def _apply_camera_calibration(
        self,
        joints_3d: np.ndarray,
        video_metadata: Dict
    ) -> np.ndarray:
        """
        Apply camera calibration to correct perspective distortion

        Args:
            joints_3d: (T, J, 3) joint positions
            video_metadata: Video metadata

        Returns:
            Calibrated joint positions
        """
        # If camera parameters not provided, estimate from video
        if self.focal_length is None:
            width = video_metadata.get('width', 1920)
            height = video_metadata.get('height', 1080)

            # Estimate focal length (rough approximation)
            # Assuming 60-degree horizontal FOV
            self.focal_length = width / (2 * np.tan(np.radians(30)))

            # Principal point at center
            self.principal_point = [width / 2, height / 2]

        # Create camera intrinsic matrix
        fx = fy = self.focal_length
        cx, cy = self.principal_point

        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        # For now, return as-is
        # In full implementation, would apply lens distortion correction
        return joints_3d

    def _align_to_ground(self, joints_3d: np.ndarray) -> np.ndarray:
        """
        Align coordinates so ground plane is at Y=0

        Args:
            joints_3d: (T, J, 3) joint positions

        Returns:
            Ground-aligned joint positions
        """
        # Find lowest Y coordinate across all frames (ground level)
        LEFT_FOOT_IDX = 10
        RIGHT_FOOT_IDX = 11

        foot_indices = [LEFT_FOOT_IDX, RIGHT_FOOT_IDX]

        min_y = np.min(joints_3d[:, foot_indices, 1])

        # Shift all Y coordinates so minimum is 0
        joints_aligned = joints_3d.copy()
        joints_aligned[:, :, 1] -= min_y

        return joints_aligned

    def pixel_to_metric(
        self,
        pixel_coords: np.ndarray,
        depth: np.ndarray
    ) -> np.ndarray:
        """
        Convert 2D pixel coordinates to 3D metric coordinates

        Args:
            pixel_coords: (N, 2) pixel coordinates [x, y]
            depth: (N,) depth values in meters

        Returns:
            (N, 3) 3D coordinates in meters
        """
        if self.camera_matrix is None:
            raise ValueError("Camera matrix not initialized. Run transform_coordinates first.")

        # Unproject using camera matrix
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        x_3d = (pixel_coords[:, 0] - cx) * depth / fx
        y_3d = (pixel_coords[:, 1] - cy) * depth / fy
        z_3d = depth

        coords_3d = np.stack([x_3d, y_3d, z_3d], axis=1)

        return coords_3d

    def metric_to_pixel(
        self,
        coords_3d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D metric coordinates to 2D pixel coordinates

        Args:
            coords_3d: (N, 3) 3D coordinates in meters

        Returns:
            pixel_coords: (N, 2) pixel coordinates
            depth: (N,) depth values
        """
        if self.camera_matrix is None:
            raise ValueError("Camera matrix not initialized. Run transform_coordinates first.")

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # Perspective projection
        x_pixel = fx * coords_3d[:, 0] / (coords_3d[:, 2] + 1e-8) + cx
        y_pixel = fy * coords_3d[:, 1] / (coords_3d[:, 2] + 1e-8) + cy

        pixel_coords = np.stack([x_pixel, y_pixel], axis=1)
        depth = coords_3d[:, 2]

        return pixel_coords, depth

    def get_scale_factor(self) -> float:
        """Get the computed scale factor"""
        return self.scale_factor if self.scale_factor is not None else 1.0

    def get_camera_matrix(self) -> Optional[np.ndarray]:
        """Get the camera intrinsic matrix"""
        return self.camera_matrix
