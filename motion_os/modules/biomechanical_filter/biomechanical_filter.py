"""
Biomechanical Filter Module - Physics-Informed Refinement

This is the CRITICAL module that transforms AI predictions into
physically-accurate biomechanical data using physics constraints.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import savgol_filter
from typing import Dict, Tuple
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.smpl_utils import (
    compute_bone_lengths,
    get_foot_joints_indices,
    compute_joint_velocities,
    BODY_SEGMENTS,
    get_joint_index
)


class BiomechanicalFilter:
    """
    Module C: Physics-Informed Biomechanical Filter

    Applies three critical constraints:
    1. Bone Length Constraint: All bone segments maintain constant length
    2. Zero Velocity Update (ZUPT): Feet have zero velocity when in contact with ground
    3. Smoothing: Remove jittering while preserving motion dynamics
    """

    def __init__(self, config: Dict):
        """
        Initialize Biomechanical Filter

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.physics_config = config['physics']

        # Extract configuration
        self.bone_constraint_enabled = self.physics_config['bone_constraint']['enabled']
        self.bone_constraint_weight = self.physics_config['bone_constraint']['weight']
        self.bone_constraint_tolerance = self.physics_config['bone_constraint']['tolerance']

        self.zupt_enabled = self.physics_config['zupt']['enabled']
        self.zupt_velocity_threshold = self.physics_config['zupt']['velocity_threshold']
        self.zupt_contact_threshold = self.physics_config['zupt']['contact_threshold']
        self.zupt_weight = self.physics_config['zupt']['weight']

        self.smoothing_enabled = self.physics_config['smoothing']['enabled']
        self.smoothing_method = self.physics_config['smoothing']['method']
        self.smoothing_window = self.physics_config['smoothing']['window_length']
        self.smoothing_polyorder = self.physics_config['smoothing']['polyorder']

        self.optimization_config = self.physics_config['optimization']

        # Reference bone lengths (computed from input data)
        self.reference_bone_lengths = None

        print("[BiomechanicalFilter] Initialized with physics constraints:")
        print(f"  - Bone Length Constraint: {self.bone_constraint_enabled} (weight={self.bone_constraint_weight})")
        print(f"  - Zero Velocity Update: {self.zupt_enabled} (weight={self.zupt_weight})")
        print(f"  - Smoothing: {self.smoothing_enabled} (method={self.smoothing_method})")

    def refine_poses(
        self,
        joints_3d: np.ndarray,
        foot_contact: np.ndarray,
        fps: float = 30.0
    ) -> Dict:
        """
        Apply physics-informed refinement to raw pose estimates

        Args:
            joints_3d: (T, J, 3) raw 3D joint positions
            foot_contact: (T, 2) foot contact flags [left, right]
            fps: Frame rate

        Returns:
            Dictionary containing:
                - refined_joints: (T, J, 3) refined joint positions
                - metrics: Refinement metrics
        """
        print("\n" + "="*60)
        print("PHYSICS-INFORMED REFINEMENT")
        print("="*60)

        # Compute reference bone lengths from input
        self.reference_bone_lengths = compute_bone_lengths(joints_3d)
        print(f"\n[BiomechanicalFilter] Reference bone lengths computed:")
        for segment, length in list(self.reference_bone_lengths.items())[:5]:
            print(f"  {segment}: {length:.4f} m")

        # Step 1: Apply smoothing (pre-optimization)
        if self.smoothing_enabled:
            joints_smoothed = self._apply_smoothing(joints_3d)
        else:
            joints_smoothed = joints_3d.copy()

        # Step 2: Optimization with physics constraints
        refined_joints = self._optimize_with_constraints(
            joints_smoothed,
            foot_contact,
            fps
        )

        # Step 3: Final smoothing pass (post-optimization)
        if self.smoothing_enabled:
            refined_joints = self._apply_smoothing(refined_joints)

        # Compute refinement metrics
        metrics = self._compute_refinement_metrics(
            joints_3d,
            refined_joints,
            foot_contact,
            fps
        )

        print("\n" + "="*60)
        print("REFINEMENT COMPLETE")
        print("="*60 + "\n")

        return {
            'refined_joints': refined_joints,
            'metrics': metrics,
            'reference_bone_lengths': self.reference_bone_lengths
        }

    def _apply_smoothing(self, joints_3d: np.ndarray) -> np.ndarray:
        """
        Apply Savitzky-Golay smoothing to remove jitter

        Args:
            joints_3d: (T, J, 3) joint positions

        Returns:
            Smoothed joint positions
        """
        print(f"[BiomechanicalFilter] Applying {self.smoothing_method} smoothing...")

        T, J, _ = joints_3d.shape
        joints_smoothed = np.zeros_like(joints_3d)

        # Apply filter to each joint and dimension
        for j in range(J):
            for d in range(3):
                signal = joints_3d[:, j, d]

                if self.smoothing_method == 'savgol':
                    # Ensure window length is odd and less than signal length
                    window = min(self.smoothing_window, T - 1)
                    if window % 2 == 0:
                        window -= 1
                    window = max(window, 3)

                    smoothed = savgol_filter(
                        signal,
                        window_length=window,
                        polyorder=min(self.smoothing_polyorder, window - 1)
                    )
                else:
                    smoothed = signal

                joints_smoothed[:, j, d] = smoothed

        return joints_smoothed

    def _optimize_with_constraints(
        self,
        joints_3d: np.ndarray,
        foot_contact: np.ndarray,
        fps: float
    ) -> np.ndarray:
        """
        Optimize joint positions with physics constraints

        Args:
            joints_3d: (T, J, 3) initial joint positions
            foot_contact: (T, 2) foot contact flags
            fps: Frame rate

        Returns:
            Optimized joint positions
        """
        print("\n[BiomechanicalFilter] Running physics-constrained optimization...")

        # Convert to PyTorch tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        joints_tensor = torch.tensor(joints_3d, dtype=torch.float32, device=device, requires_grad=True)
        foot_contact_tensor = torch.tensor(foot_contact, dtype=torch.float32, device=device)

        # Optimizer
        optimizer = optim.Adam([joints_tensor], lr=self.optimization_config['learning_rate'])

        max_iterations = self.optimization_config['max_iterations']
        convergence_threshold = self.optimization_config['convergence_threshold']

        prev_loss = float('inf')

        pbar = tqdm(range(max_iterations), desc="Optimization")

        for iteration in pbar:
            optimizer.zero_grad()

            # Compute total loss
            total_loss = 0.0
            losses_dict = {}

            # 1. Bone Length Constraint Loss
            if self.bone_constraint_enabled:
                bone_loss = self._compute_bone_length_loss(joints_tensor)
                total_loss += self.bone_constraint_weight * bone_loss
                losses_dict['bone'] = bone_loss.item()

            # 2. Zero Velocity Update (ZUPT) Loss
            if self.zupt_enabled:
                zupt_loss = self._compute_zupt_loss(joints_tensor, foot_contact_tensor, fps)
                total_loss += self.zupt_weight * zupt_loss
                losses_dict['zupt'] = zupt_loss.item()

            # 3. Data fidelity loss (regularization - stay close to original)
            original_tensor = torch.tensor(joints_3d, dtype=torch.float32, device=device)
            data_loss = torch.mean((joints_tensor - original_tensor) ** 2)
            total_loss += 1.0 * data_loss
            losses_dict['data'] = data_loss.item()

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Update progress bar
            pbar.set_postfix(losses_dict)

            # Check convergence
            if abs(prev_loss - total_loss.item()) < convergence_threshold:
                print(f"\n[BiomechanicalFilter] Converged at iteration {iteration}")
                break

            prev_loss = total_loss.item()

        pbar.close()

        # Convert back to numpy
        refined_joints = joints_tensor.detach().cpu().numpy()

        print(f"[BiomechanicalFilter] Optimization complete. Final loss: {total_loss.item():.6f}")

        return refined_joints

    def _compute_bone_length_loss(self, joints: torch.Tensor) -> torch.Tensor:
        """
        Compute bone length constraint loss (L2)

        All bone segments should maintain constant length across frames

        Args:
            joints: (T, J, 3) joint positions tensor

        Returns:
            Bone length loss
        """
        total_loss = 0.0
        num_segments = 0

        for segment_name, (joint_start, joint_end) in BODY_SEGMENTS.items():
            idx_start = get_joint_index(joint_start)
            idx_end = get_joint_index(joint_end)

            # Reference length
            ref_length = self.reference_bone_lengths[segment_name]

            # Compute current lengths for all frames
            segment_vectors = joints[:, idx_end] - joints[:, idx_start]
            current_lengths = torch.norm(segment_vectors, dim=1)

            # L2 loss: (current_length - reference_length)^2
            length_errors = (current_lengths - ref_length) ** 2
            total_loss += torch.mean(length_errors)
            num_segments += 1

        return total_loss / num_segments

    def _compute_zupt_loss(
        self,
        joints: torch.Tensor,
        foot_contact: torch.Tensor,
        fps: float
    ) -> torch.Tensor:
        """
        Compute Zero Velocity Update (ZUPT) loss

        When foot is in contact with ground, velocity should be zero

        Args:
            joints: (T, J, 3) joint positions
            foot_contact: (T, 2) foot contact flags
            fps: Frame rate

        Returns:
            ZUPT loss
        """
        dt = 1.0 / fps

        # Get foot joint indices
        foot_indices = get_foot_joints_indices()

        total_loss = 0.0
        num_contacts = 0

        # Left foot
        for joint_idx in foot_indices['left']:
            # Compute velocities
            velocities = (joints[1:, joint_idx] - joints[:-1, joint_idx]) / dt

            # Apply ZUPT only when foot is in contact
            contact_mask = foot_contact[:-1, 0]  # Left foot contact

            # Loss: velocity should be zero when in contact
            velocity_magnitudes = torch.norm(velocities, dim=1)
            zupt_errors = contact_mask * (velocity_magnitudes ** 2)

            total_loss += torch.sum(zupt_errors)
            num_contacts += torch.sum(contact_mask)

        # Right foot
        for joint_idx in foot_indices['right']:
            velocities = (joints[1:, joint_idx] - joints[:-1, joint_idx]) / dt
            contact_mask = foot_contact[:-1, 1]  # Right foot contact

            velocity_magnitudes = torch.norm(velocities, dim=1)
            zupt_errors = contact_mask * (velocity_magnitudes ** 2)

            total_loss += torch.sum(zupt_errors)
            num_contacts += torch.sum(contact_mask)

        # Normalize by number of contact frames
        if num_contacts > 0:
            return total_loss / (num_contacts + 1e-8)
        else:
            return torch.tensor(0.0, device=joints.device)

    def _compute_refinement_metrics(
        self,
        original_joints: np.ndarray,
        refined_joints: np.ndarray,
        foot_contact: np.ndarray,
        fps: float
    ) -> Dict:
        """
        Compute metrics to evaluate refinement quality

        Args:
            original_joints: Original joint positions
            refined_joints: Refined joint positions
            foot_contact: Foot contact flags
            fps: Frame rate

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # 1. Bone length consistency
        bone_lengths_original = self._compute_all_bone_lengths(original_joints)
        bone_lengths_refined = self._compute_all_bone_lengths(refined_joints)

        # Variance of bone lengths (lower is better)
        metrics['bone_length_variance_original'] = np.mean([np.var(lengths) for lengths in bone_lengths_original.values()])
        metrics['bone_length_variance_refined'] = np.mean([np.var(lengths) for lengths in bone_lengths_refined.values()])

        # 2. Foot velocity during contact (should be near zero)
        foot_velocities_original = self._compute_foot_velocities(original_joints, fps)
        foot_velocities_refined = self._compute_foot_velocities(refined_joints, fps)

        # Average velocity during contact
        left_contact_mask = foot_contact[:, 0] > 0.5
        right_contact_mask = foot_contact[:, 1] > 0.5

        if np.any(left_contact_mask):
            metrics['left_foot_velocity_original'] = np.mean(foot_velocities_original['left'][left_contact_mask[:-1]])
            metrics['left_foot_velocity_refined'] = np.mean(foot_velocities_refined['left'][left_contact_mask[:-1]])

        if np.any(right_contact_mask):
            metrics['right_foot_velocity_original'] = np.mean(foot_velocities_original['right'][right_contact_mask[:-1]])
            metrics['right_foot_velocity_refined'] = np.mean(foot_velocities_refined['right'][right_contact_mask[:-1]])

        # 3. Overall position change
        position_change = np.mean(np.linalg.norm(refined_joints - original_joints, axis=2))
        metrics['average_position_change'] = position_change

        # Print summary
        print("\n[BiomechanicalFilter] Refinement Metrics:")
        print(f"  Bone length variance: {metrics['bone_length_variance_original']:.6f} → {metrics['bone_length_variance_refined']:.6f}")
        if 'left_foot_velocity_original' in metrics:
            print(f"  Left foot velocity (contact): {metrics['left_foot_velocity_original']:.4f} → {metrics['left_foot_velocity_refined']:.4f} m/s")
        if 'right_foot_velocity_original' in metrics:
            print(f"  Right foot velocity (contact): {metrics['right_foot_velocity_original']:.4f} → {metrics['right_foot_velocity_refined']:.4f} m/s")
        print(f"  Average position change: {position_change:.4f} m")

        return metrics

    def _compute_all_bone_lengths(self, joints: np.ndarray) -> Dict:
        """Compute bone lengths for all frames"""
        bone_lengths = {segment: [] for segment in BODY_SEGMENTS.keys()}

        for segment_name, (joint_start, joint_end) in BODY_SEGMENTS.items():
            idx_start = get_joint_index(joint_start)
            idx_end = get_joint_index(joint_end)

            lengths = np.linalg.norm(joints[:, idx_end] - joints[:, idx_start], axis=1)
            bone_lengths[segment_name] = lengths

        return bone_lengths

    def _compute_foot_velocities(self, joints: np.ndarray, fps: float) -> Dict:
        """Compute foot velocities"""
        dt = 1.0 / fps
        foot_indices = get_foot_joints_indices()

        velocities = {}

        # Left foot (average of ankle and foot)
        left_vels = []
        for joint_idx in foot_indices['left']:
            vel = np.linalg.norm(np.diff(joints[:, joint_idx], axis=0), axis=1) / dt
            left_vels.append(vel)
        velocities['left'] = np.mean(left_vels, axis=0)

        # Right foot
        right_vels = []
        for joint_idx in foot_indices['right']:
            vel = np.linalg.norm(np.diff(joints[:, joint_idx], axis=0), axis=1) / dt
            right_vels.append(vel)
        velocities['right'] = np.mean(right_vels, axis=0)

        return velocities
