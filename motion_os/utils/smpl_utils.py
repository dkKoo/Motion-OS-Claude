"""
SMPL/SMPL-X utility functions for body model handling
"""

import numpy as np
import torch


# SMPL Joint Names (24 joints)
SMPL_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
]

# Body segments for bone length constraints
BODY_SEGMENTS = {
    'left_femur': ('left_hip', 'left_knee'),
    'right_femur': ('right_hip', 'right_knee'),
    'left_tibia': ('left_knee', 'left_ankle'),
    'right_tibia': ('right_knee', 'right_ankle'),
    'left_foot': ('left_ankle', 'left_foot'),
    'right_foot': ('right_ankle', 'right_foot'),
    'left_humerus': ('left_shoulder', 'left_elbow'),
    'right_humerus': ('right_shoulder', 'right_elbow'),
    'left_radius': ('left_elbow', 'left_wrist'),
    'right_radius': ('right_elbow', 'right_wrist'),
    'spine_lower': ('pelvis', 'spine1'),
    'spine_middle': ('spine1', 'spine2'),
    'spine_upper': ('spine2', 'spine3'),
    'neck_segment': ('spine3', 'neck'),
}

# Foot contact joints (for ZUPT)
FOOT_CONTACT_JOINTS = {
    'left': ['left_ankle', 'left_foot'],
    'right': ['right_ankle', 'right_foot']
}


def get_joint_index(joint_name):
    """Get index of joint by name"""
    if joint_name in SMPL_JOINT_NAMES:
        return SMPL_JOINT_NAMES.index(joint_name)
    raise ValueError(f"Unknown joint name: {joint_name}")


def compute_bone_lengths(joints_3d):
    """
    Compute bone lengths for all segments

    Args:
        joints_3d: (T, J, 3) array of 3D joint positions

    Returns:
        Dictionary of bone lengths for each segment
    """
    bone_lengths = {}

    for segment_name, (joint_start, joint_end) in BODY_SEGMENTS.items():
        idx_start = get_joint_index(joint_start)
        idx_end = get_joint_index(joint_end)

        # Compute length across all frames
        lengths = np.linalg.norm(
            joints_3d[:, idx_end] - joints_3d[:, idx_start],
            axis=1
        )

        # Use median as reference length
        bone_lengths[segment_name] = np.median(lengths)

    return bone_lengths


def get_foot_joints_indices():
    """Get indices of foot joints for contact detection"""
    left_indices = [get_joint_index(j) for j in FOOT_CONTACT_JOINTS['left']]
    right_indices = [get_joint_index(j) for j in FOOT_CONTACT_JOINTS['right']]

    return {
        'left': left_indices,
        'right': right_indices
    }


def compute_joint_velocities(joints_3d, fps=30):
    """
    Compute joint velocities from positions

    Args:
        joints_3d: (T, J, 3) array of 3D joint positions
        fps: Frame rate

    Returns:
        (T, J, 3) array of velocities
    """
    dt = 1.0 / fps

    # Compute velocities using central difference
    velocities = np.zeros_like(joints_3d)
    velocities[1:-1] = (joints_3d[2:] - joints_3d[:-2]) / (2 * dt)

    # Forward/backward difference for boundaries
    velocities[0] = (joints_3d[1] - joints_3d[0]) / dt
    velocities[-1] = (joints_3d[-1] - joints_3d[-2]) / dt

    return velocities


def rotation_matrix_to_euler(R, order='XYZ'):
    """
    Convert rotation matrix to Euler angles

    Args:
        R: 3x3 rotation matrix
        order: Rotation order (default: 'XYZ')

    Returns:
        Euler angles in radians (3,)
    """
    if order == 'XYZ':
        # Sagittal (flexion/extension)
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)

        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])
    else:
        raise NotImplementedError(f"Rotation order {order} not implemented")


def compute_joint_angle(parent_pos, joint_pos, child_pos, plane='sagittal'):
    """
    Compute joint angle in specific plane

    Args:
        parent_pos: (3,) position of parent joint
        joint_pos: (3,) position of joint
        child_pos: (3,) position of child joint
        plane: 'sagittal', 'coronal', or 'transverse'

    Returns:
        Angle in degrees
    """
    # Vectors from joint to parent and child
    v1 = parent_pos - joint_pos
    v2 = child_pos - joint_pos

    # Project onto plane
    if plane == 'sagittal':  # YZ plane (flexion/extension)
        v1 = v1[[1, 2]]
        v2 = v2[[1, 2]]
    elif plane == 'coronal':  # XZ plane (abduction/adduction)
        v1 = v1[[0, 2]]
        v2 = v2[[0, 2]]
    elif plane == 'transverse':  # XY plane (rotation)
        v1 = v1[[0, 1]]
        v2 = v2[[0, 1]]

    # Compute angle
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    return np.degrees(angle)


def scale_smpl_to_height(smpl_joints, target_height_m):
    """
    Scale SMPL joints to match target height

    Args:
        smpl_joints: (T, J, 3) joint positions
        target_height_m: Target height in meters

    Returns:
        Scaled joint positions
    """
    # Compute current height (head to foot)
    head_idx = get_joint_index('head')
    foot_indices = [get_joint_index('left_foot'), get_joint_index('right_foot')]

    # Average height across frames
    heights = []
    for frame in smpl_joints:
        head_y = frame[head_idx, 1]
        foot_y = min(frame[foot_indices[0], 1], frame[foot_indices[1], 1])
        heights.append(head_y - foot_y)

    current_height = np.median(heights)

    # Scale factor
    scale = target_height_m / (current_height + 1e-8)

    return smpl_joints * scale
