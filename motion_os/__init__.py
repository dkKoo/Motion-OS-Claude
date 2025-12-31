"""
Motion OS Phase 1: Physics-Informed 3D Gait Analysis System

A comprehensive system for analyzing human gait from video using:
- 3D pose estimation (WHAM)
- Physics-informed refinement
- Biomechanical analysis
"""

__version__ = "1.0.0"
__author__ = "Motion OS Team"

from .core.pipeline import MotionOSPipeline

__all__ = ["MotionOSPipeline"]
