"""
Gait Analysis Module
Comprehensive biomechanical analysis of walking patterns
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.signal import find_peaks

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.smpl_utils import (
    get_joint_index,
    compute_joint_angle,
    compute_joint_velocities
)


class GaitAnalyzer:
    """
    Module D: Gait Analysis
    - Detects gait cycles (Heel Strike, Toe Off)
    - Computes spatio-temporal parameters
    - Calculates joint kinematics (ROM in 3 planes)
    """

    def __init__(self, config: Dict):
        """
        Initialize Gait Analyzer

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.biomech_config = config['biomechanics']

        # Gait cycle detection parameters
        self.heel_strike_threshold = self.biomech_config['gait_cycle']['heel_strike_threshold']
        self.toe_off_threshold = self.biomech_config['gait_cycle']['toe_off_threshold']
        self.min_cycle_duration = self.biomech_config['gait_cycle']['min_cycle_duration']

        # Joint configuration
        self.joints_config = self.biomech_config['joints']
        self.planes = self.biomech_config['planes']

        print("[GaitAnalyzer] Initialized")
        print(f"  Analyzing joints: {[j['name'] for j in self.joints_config]}")
        print(f"  Planes: {self.planes}")

    def analyze_gait(
        self,
        joints_3d: np.ndarray,
        fps: float = 30.0
    ) -> Dict:
        """
        Perform comprehensive gait analysis

        Args:
            joints_3d: (T, J, 3) refined 3D joint positions
            fps: Frame rate

        Returns:
            Dictionary containing:
                - gait_cycles: Detected gait cycles
                - spatio_temporal: Spatio-temporal parameters
                - joint_angles: Joint kinematics
        """
        print("\n" + "="*60)
        print("GAIT ANALYSIS")
        print("="*60)

        # Step 1: Detect gait cycles
        gait_cycles = self._detect_gait_cycles(joints_3d, fps)

        # Step 2: Compute spatio-temporal parameters
        spatio_temporal = self._compute_spatio_temporal_parameters(
            joints_3d,
            gait_cycles,
            fps
        )

        # Step 3: Calculate joint kinematics
        joint_angles = self._compute_joint_kinematics(joints_3d, fps)

        print("\n" + "="*60)
        print("GAIT ANALYSIS COMPLETE")
        print("="*60 + "\n")

        return {
            'gait_cycles': gait_cycles,
            'spatio_temporal': spatio_temporal,
            'joint_angles': joint_angles
        }

    def _detect_gait_cycles(
        self,
        joints_3d: np.ndarray,
        fps: float
    ) -> Dict:
        """
        Detect gait cycles using heel strike and toe off events

        Args:
            joints_3d: (T, J, 3) joint positions
            fps: Frame rate

        Returns:
            Dictionary with detected cycles for left and right feet
        """
        print("\n[GaitAnalyzer] Detecting gait cycles...")

        # Get foot joint indices
        left_heel_idx = get_joint_index('left_ankle')
        right_heel_idx = get_joint_index('right_ankle')
        left_toe_idx = get_joint_index('left_foot')
        right_toe_idx = get_joint_index('right_foot')

        # Detect events for left foot
        left_heel_strikes, left_toe_offs = self._detect_foot_events(
            joints_3d[:, left_heel_idx, 1],
            joints_3d[:, left_toe_idx, 1],
            fps
        )

        # Detect events for right foot
        right_heel_strikes, right_toe_offs = self._detect_foot_events(
            joints_3d[:, right_heel_idx, 1],
            joints_3d[:, right_toe_idx, 1],
            fps
        )

        # Build gait cycles
        left_cycles = self._build_gait_cycles(left_heel_strikes, left_toe_offs, fps)
        right_cycles = self._build_gait_cycles(right_heel_strikes, right_toe_offs, fps)

        print(f"[GaitAnalyzer] Detected {len(left_cycles)} left foot cycles")
        print(f"[GaitAnalyzer] Detected {len(right_cycles)} right foot cycles")

        return {
            'left': left_cycles,
            'right': right_cycles,
            'left_heel_strikes': left_heel_strikes,
            'left_toe_offs': left_toe_offs,
            'right_heel_strikes': right_heel_strikes,
            'right_toe_offs': right_toe_offs
        }

    def _detect_foot_events(
        self,
        heel_y: np.ndarray,
        toe_y: np.ndarray,
        fps: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect heel strike and toe off events

        Args:
            heel_y: (T,) Y coordinates of heel
            toe_y: (T,) Y coordinates of toe
            fps: Frame rate

        Returns:
            heel_strikes: Frame indices of heel strikes
            toe_offs: Frame indices of toe offs
        """
        # Compute vertical velocity
        dt = 1.0 / fps
        heel_velocity = np.gradient(heel_y, dt)

        # Heel Strike: when heel is low and velocity changes from negative to positive
        # (foot moving downward then stopping)
        heel_is_low = heel_y < self.heel_strike_threshold

        # Find local minima in heel height
        heel_strikes, _ = find_peaks(-heel_y, distance=int(fps * self.min_cycle_duration))

        # Filter based on threshold
        heel_strikes = heel_strikes[heel_y[heel_strikes] < self.heel_strike_threshold]

        # Toe Off: when toe lifts off ground (toe Y increases above threshold)
        toe_is_high = toe_y > self.toe_off_threshold

        # Find when toe transitions from low to high
        toe_off_transitions = np.diff(toe_is_high.astype(int))
        toe_offs = np.where(toe_off_transitions > 0)[0] + 1

        return heel_strikes, toe_offs

    def _build_gait_cycles(
        self,
        heel_strikes: np.ndarray,
        toe_offs: np.ndarray,
        fps: float
    ) -> List[Dict]:
        """
        Build gait cycles from heel strike and toe off events

        Args:
            heel_strikes: Frame indices of heel strikes
            toe_offs: Frame indices of toe offs
            fps: Frame rate

        Returns:
            List of gait cycle dictionaries
        """
        cycles = []

        for i in range(len(heel_strikes) - 1):
            cycle_start = heel_strikes[i]
            cycle_end = heel_strikes[i + 1]

            # Find toe offs within this cycle
            cycle_toe_offs = toe_offs[(toe_offs > cycle_start) & (toe_offs < cycle_end)]

            if len(cycle_toe_offs) > 0:
                toe_off = cycle_toe_offs[0]
            else:
                toe_off = None

            duration = (cycle_end - cycle_start) / fps

            # Only include cycles that meet minimum duration
            if duration >= self.min_cycle_duration:
                cycles.append({
                    'start_frame': cycle_start,
                    'end_frame': cycle_end,
                    'toe_off_frame': toe_off,
                    'duration': duration,
                    'stance_duration': (toe_off - cycle_start) / fps if toe_off else None,
                    'swing_duration': (cycle_end - toe_off) / fps if toe_off else None
                })

        return cycles

    def _compute_spatio_temporal_parameters(
        self,
        joints_3d: np.ndarray,
        gait_cycles: Dict,
        fps: float
    ) -> Dict:
        """
        Compute spatio-temporal gait parameters

        Args:
            joints_3d: (T, J, 3) joint positions
            gait_cycles: Detected gait cycles
            fps: Frame rate

        Returns:
            Dictionary of spatio-temporal parameters
        """
        print("\n[GaitAnalyzer] Computing spatio-temporal parameters...")

        # Get pelvis trajectory (for velocity)
        pelvis_idx = get_joint_index('pelvis')
        pelvis_pos = joints_3d[:, pelvis_idx, :]

        # Compute velocity (forward direction, assuming Z-axis)
        dt = 1.0 / fps
        velocity = np.linalg.norm(np.diff(pelvis_pos, axis=0), axis=1) / dt

        # Average velocity
        avg_velocity = np.mean(velocity)

        # Compute stride lengths for each foot
        stride_lengths_left = []
        stride_lengths_right = []

        left_foot_idx = get_joint_index('left_foot')
        right_foot_idx = get_joint_index('right_foot')

        for cycle in gait_cycles['left']:
            start_pos = joints_3d[cycle['start_frame'], left_foot_idx]
            end_pos = joints_3d[cycle['end_frame'], left_foot_idx]
            stride_length = np.linalg.norm(end_pos - start_pos)
            stride_lengths_left.append(stride_length)

        for cycle in gait_cycles['right']:
            start_pos = joints_3d[cycle['start_frame'], right_foot_idx]
            end_pos = joints_3d[cycle['end_frame'], right_foot_idx]
            stride_length = np.linalg.norm(end_pos - start_pos)
            stride_lengths_right.append(stride_length)

        # Cadence (steps per minute)
        total_steps = len(gait_cycles['left']) + len(gait_cycles['right'])
        total_duration = len(joints_3d) / fps
        cadence = (total_steps / total_duration) * 60 if total_duration > 0 else 0

        # Cycle durations
        cycle_durations_left = [c['duration'] for c in gait_cycles['left']]
        cycle_durations_right = [c['duration'] for c in gait_cycles['right']]

        params = {
            'velocity_mean': avg_velocity,
            'velocity_std': np.std(velocity),
            'velocities': velocity,

            'stride_length_left_mean': np.mean(stride_lengths_left) if stride_lengths_left else 0,
            'stride_length_right_mean': np.mean(stride_lengths_right) if stride_lengths_right else 0,
            'stride_lengths_left': stride_lengths_left,
            'stride_lengths_right': stride_lengths_right,

            'cadence': cadence,

            'cycle_duration_left_mean': np.mean(cycle_durations_left) if cycle_durations_left else 0,
            'cycle_duration_right_mean': np.mean(cycle_durations_right) if cycle_durations_right else 0,
            'cycle_durations': cycle_durations_left + cycle_durations_right,

            'num_cycles_left': len(gait_cycles['left']),
            'num_cycles_right': len(gait_cycles['right'])
        }

        print(f"  Average velocity: {params['velocity_mean']:.3f} m/s")
        print(f"  Average stride length (left): {params['stride_length_left_mean']:.3f} m")
        print(f"  Average stride length (right): {params['stride_length_right_mean']:.3f} m")
        print(f"  Cadence: {params['cadence']:.1f} steps/min")

        return params

    def _compute_joint_kinematics(
        self,
        joints_3d: np.ndarray,
        fps: float
    ) -> Dict:
        """
        Compute joint angles (ROM) in all planes

        Args:
            joints_3d: (T, J, 3) joint positions
            fps: Frame rate

        Returns:
            Dictionary of joint angles for each joint and plane
        """
        print("\n[GaitAnalyzer] Computing joint kinematics (ROM)...")

        joint_angles = {}

        for joint_config in self.joints_config:
            joint_name = joint_config['name']
            parent_name = joint_config['parent']
            child_name = joint_config['child']

            print(f"  Analyzing {joint_name}...")

            # Get joint indices
            parent_idx = get_joint_index(parent_name)
            joint_idx = get_joint_index(child_name)  # The joint itself is the child in the hierarchy

            # For computing angles, we need three points: parent -> joint -> child
            # For joints like hip, knee, ankle, we use:
            # - Hip: pelvis -> hip -> knee
            # - Knee: hip -> knee -> ankle
            # - Ankle: knee -> ankle -> foot

            if joint_name == 'hip':
                point_parent_idx = get_joint_index('pelvis')
                point_joint_idx = get_joint_index('left_hip')  # or right_hip
                point_child_idx = get_joint_index('left_knee')
            elif joint_name == 'knee':
                point_parent_idx = get_joint_index('left_hip')
                point_joint_idx = get_joint_index('left_knee')
                point_child_idx = get_joint_index('left_ankle')
            elif joint_name == 'ankle':
                point_parent_idx = get_joint_index('left_knee')
                point_joint_idx = get_joint_index('left_ankle')
                point_child_idx = get_joint_index('left_foot')
            else:
                # Default mapping
                point_parent_idx = parent_idx
                point_joint_idx = joint_idx
                point_child_idx = joint_idx

            angles = {plane: [] for plane in self.planes}

            # Compute angles for each frame
            for frame in joints_3d:
                parent_pos = frame[point_parent_idx]
                joint_pos = frame[point_joint_idx]
                child_pos = frame[point_child_idx]

                for plane in self.planes:
                    angle = compute_joint_angle(parent_pos, joint_pos, child_pos, plane)
                    angles[plane].append(angle)

            # Convert to numpy arrays
            for plane in self.planes:
                angles[plane] = np.array(angles[plane])

            # Compute ROM (Range of Motion)
            rom = {}
            for plane in self.planes:
                rom[plane] = {
                    'min': np.min(angles[plane]),
                    'max': np.max(angles[plane]),
                    'range': np.max(angles[plane]) - np.min(angles[plane]),
                    'mean': np.mean(angles[plane])
                }

            print(f"    Sagittal ROM: {rom['sagittal']['range']:.1f}° "
                  f"[{rom['sagittal']['min']:.1f}° to {rom['sagittal']['max']:.1f}°]")

            joint_angles[joint_name] = {
                'angles': angles,
                'rom': rom
            }

        return joint_angles
