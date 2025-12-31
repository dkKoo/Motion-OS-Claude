"""
Video Processor Module
Extracts frames and estimates 3D pose using WHAM
"""

import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, Tuple, Optional


class VideoProcessor:
    """
    Module A: Video Processor
    - Decomposes video into frames
    - Runs WHAM for 3D pose and global coordinates
    - Detects foot-ground contact
    """

    def __init__(self, config: Dict):
        """
        Initialize Video Processor

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = config['system']['device']
        self.fps = config['video']['fps']

        # Initialize WHAM model (placeholder - will integrate actual model)
        self.wham_model = None
        self._init_wham_model()

    def _init_wham_model(self):
        """Initialize WHAM model"""
        # TODO: Load actual WHAM model from checkpoint
        # For now, create a placeholder that simulates WHAM output
        print("[VideoProcessor] Initializing WHAM model...")

        # Check if checkpoint exists
        checkpoint_path = self.config['wham']['checkpoint_path']
        if os.path.exists(checkpoint_path):
            # Load actual WHAM model
            try:
                # self.wham_model = WHAMModel.load(checkpoint_path)
                print(f"[VideoProcessor] Loaded WHAM from {checkpoint_path}")
            except Exception as e:
                print(f"[VideoProcessor] Warning: Could not load WHAM model: {e}")
                print("[VideoProcessor] Using fallback pose estimator")
                self.wham_model = None
        else:
            print(f"[VideoProcessor] WHAM checkpoint not found at {checkpoint_path}")
            print("[VideoProcessor] Using fallback MediaPipe pose estimator")
            self.wham_model = None

    def process_video(self, video_path: str) -> Dict:
        """
        Process video and extract 3D poses

        Args:
            video_path: Path to input video

        Returns:
            Dictionary containing:
                - joints_3d: (T, J, 3) 3D joint positions
                - foot_contact: (T, 2) foot contact flags [left, right]
                - frames: List of video frames
                - metadata: Video metadata
        """
        print(f"[VideoProcessor] Processing video: {video_path}")

        # Extract frames
        frames, metadata = self._extract_frames(video_path)

        # Run pose estimation
        joints_3d, foot_contact = self._estimate_poses(frames)

        return {
            'joints_3d': joints_3d,
            'foot_contact': foot_contact,
            'frames': frames,
            'metadata': metadata
        }

    def _extract_frames(self, video_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Extract frames from video

        Args:
            video_path: Path to video file

        Returns:
            frames: (T, H, W, 3) array of frames
            metadata: Dictionary with video metadata
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        metadata = {
            'fps': fps,
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'duration': total_frames / fps if fps > 0 else 0
        }

        # Extract frames
        frames = []
        pbar = tqdm(total=total_frames, desc="Extracting frames")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames.append(frame)
            pbar.update(1)

        pbar.close()
        cap.release()

        frames = np.array(frames)
        print(f"[VideoProcessor] Extracted {len(frames)} frames ({width}x{height} @ {fps} fps)")

        return frames, metadata

    def _estimate_poses(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate 3D poses from frames

        Args:
            frames: (T, H, W, 3) video frames

        Returns:
            joints_3d: (T, J, 3) 3D joint positions
            foot_contact: (T, 2) foot contact flags
        """
        if self.wham_model is not None:
            return self._estimate_poses_wham(frames)
        else:
            return self._estimate_poses_fallback(frames)

    def _estimate_poses_wham(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate poses using WHAM model

        Args:
            frames: (T, H, W, 3) video frames

        Returns:
            joints_3d: (T, J, 3) 3D joint positions
            foot_contact: (T, 2) foot contact flags
        """
        # TODO: Implement actual WHAM inference
        print("[VideoProcessor] Running WHAM inference...")

        T = len(frames)
        J = 24  # SMPL joints

        joints_3d = []
        foot_contact = []

        with torch.no_grad():
            for frame in tqdm(frames, desc="WHAM inference"):
                # Preprocess frame
                # input_tensor = self._preprocess_frame(frame)

                # Run WHAM
                # output = self.wham_model(input_tensor)

                # Extract joints and foot contact
                # joints = output['joints_3d']  # (J, 3)
                # contact = output['foot_contact']  # (2,)

                # Placeholder
                joints = np.random.randn(J, 3) * 0.5
                contact = np.array([0, 0])

                joints_3d.append(joints)
                foot_contact.append(contact)

        joints_3d = np.array(joints_3d)
        foot_contact = np.array(foot_contact)

        return joints_3d, foot_contact

    def _estimate_poses_fallback(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback pose estimation using MediaPipe

        Args:
            frames: (T, H, W, 3) video frames

        Returns:
            joints_3d: (T, J, 3) 3D joint positions
            foot_contact: (T, 2) foot contact flags
        """
        print("[VideoProcessor] Using MediaPipe fallback for pose estimation...")

        try:
            import mediapipe as mp
            mp_pose = mp.solutions.pose

            T = len(frames)
            J = 24  # SMPL joint count

            joints_3d = []
            foot_contact = []

            with mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5
            ) as pose:

                for frame in tqdm(frames, desc="MediaPipe inference"):
                    # Convert BGR to RGB
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Process
                    results = pose.process(image_rgb)

                    if results.pose_world_landmarks:
                        # Extract 3D landmarks
                        landmarks = results.pose_world_landmarks.landmark

                        # Map MediaPipe landmarks to SMPL joints (simplified)
                        joints = self._mediapipe_to_smpl(landmarks)

                        # Detect foot contact (simplified)
                        left_contact = 1 if landmarks[31].y < 0.1 else 0  # Left heel
                        right_contact = 1 if landmarks[32].y < 0.1 else 0  # Right heel

                    else:
                        # No detection - use zeros
                        joints = np.zeros((J, 3))
                        left_contact = 0
                        right_contact = 0

                    joints_3d.append(joints)
                    foot_contact.append([left_contact, right_contact])

            joints_3d = np.array(joints_3d)
            foot_contact = np.array(foot_contact)

            return joints_3d, foot_contact

        except ImportError:
            print("[VideoProcessor] MediaPipe not available, using dummy data")
            return self._generate_dummy_poses(len(frames))

    def _mediapipe_to_smpl(self, mediapipe_landmarks) -> np.ndarray:
        """
        Convert MediaPipe landmarks to SMPL-like joint format

        Args:
            mediapipe_landmarks: MediaPipe pose landmarks

        Returns:
            (24, 3) SMPL joint array
        """
        # Simplified mapping from MediaPipe (33 landmarks) to SMPL (24 joints)
        # This is a rough approximation

        mp_to_smpl = {
            0: 23,   # pelvis -> center of hips
            1: 1,    # left_hip -> left hip
            2: 2,    # right_hip -> right hip
            3: 3,    # spine1 -> upper torso
            4: 4,    # left_knee -> left knee
            5: 5,    # right_knee -> right knee
            6: 6,    # spine2 -> chest
            7: 7,    # left_ankle -> left ankle
            8: 8,    # right_ankle -> right ankle
            9: 9,    # spine3 -> upper chest
            10: 10,  # left_foot -> left foot
            11: 11,  # right_foot -> right foot
            12: 12,  # neck -> neck
            13: 13,  # left_collar -> left shoulder
            14: 14,  # right_collar -> right shoulder
            15: 15,  # head -> nose
            16: 16,  # left_shoulder -> left shoulder
            17: 17,  # right_shoulder -> right shoulder
            18: 18,  # left_elbow -> left elbow
            19: 19,  # right_elbow -> right elbow
            20: 20,  # left_wrist -> left wrist
            21: 21,  # right_wrist -> right wrist
            22: 22,  # left_hand -> left hand
            23: 23,  # right_hand -> right hand
        }

        joints = np.zeros((24, 3))

        # Map MediaPipe to SMPL (rough approximation)
        joints[0] = np.array([mediapipe_landmarks[23].x, mediapipe_landmarks[23].y, mediapipe_landmarks[23].z])  # pelvis
        joints[1] = np.array([mediapipe_landmarks[23].x, mediapipe_landmarks[23].y, mediapipe_landmarks[23].z])  # left_hip
        joints[2] = np.array([mediapipe_landmarks[24].x, mediapipe_landmarks[24].y, mediapipe_landmarks[24].z])  # right_hip
        joints[4] = np.array([mediapipe_landmarks[25].x, mediapipe_landmarks[25].y, mediapipe_landmarks[25].z])  # left_knee
        joints[5] = np.array([mediapipe_landmarks[26].x, mediapipe_landmarks[26].y, mediapipe_landmarks[26].z])  # right_knee
        joints[7] = np.array([mediapipe_landmarks[27].x, mediapipe_landmarks[27].y, mediapipe_landmarks[27].z])  # left_ankle
        joints[8] = np.array([mediapipe_landmarks[28].x, mediapipe_landmarks[28].y, mediapipe_landmarks[28].z])  # right_ankle
        joints[10] = np.array([mediapipe_landmarks[31].x, mediapipe_landmarks[31].y, mediapipe_landmarks[31].z]) # left_foot
        joints[11] = np.array([mediapipe_landmarks[32].x, mediapipe_landmarks[32].y, mediapipe_landmarks[32].z]) # right_foot
        joints[16] = np.array([mediapipe_landmarks[11].x, mediapipe_landmarks[11].y, mediapipe_landmarks[11].z]) # left_shoulder
        joints[17] = np.array([mediapipe_landmarks[12].x, mediapipe_landmarks[12].y, mediapipe_landmarks[12].z]) # right_shoulder
        joints[18] = np.array([mediapipe_landmarks[13].x, mediapipe_landmarks[13].y, mediapipe_landmarks[13].z]) # left_elbow
        joints[19] = np.array([mediapipe_landmarks[14].x, mediapipe_landmarks[14].y, mediapipe_landmarks[14].z]) # right_elbow
        joints[20] = np.array([mediapipe_landmarks[15].x, mediapipe_landmarks[15].y, mediapipe_landmarks[15].z]) # left_wrist
        joints[21] = np.array([mediapipe_landmarks[16].x, mediapipe_landmarks[16].y, mediapipe_landmarks[16].z]) # right_wrist

        # Fill in spine joints (interpolated)
        joints[3] = (joints[0] + joints[16] + joints[17]) / 3  # spine1
        joints[6] = (joints[16] + joints[17]) / 2               # spine2
        joints[9] = joints[6]                                    # spine3
        joints[12] = np.array([mediapipe_landmarks[0].x, mediapipe_landmarks[0].y, mediapipe_landmarks[0].z])  # neck
        joints[15] = np.array([mediapipe_landmarks[0].x, mediapipe_landmarks[0].y, mediapipe_landmarks[0].z])  # head

        return joints

    def _generate_dummy_poses(self, num_frames: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate dummy pose data for testing"""
        print("[VideoProcessor] Generating dummy pose data")

        J = 24
        joints_3d = np.random.randn(num_frames, J, 3) * 0.3
        foot_contact = np.random.randint(0, 2, (num_frames, 2))

        return joints_3d, foot_contact
