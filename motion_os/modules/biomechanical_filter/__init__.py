"""
Module C: Biomechanical Filter (CRITICAL MODULE)
Physics-informed refinement of 3D poses
- Bone Length Constraints
- Zero Velocity Update (ZUPT)
- Signal Smoothing
"""

from .biomechanical_filter import BiomechanicalFilter

__all__ = ["BiomechanicalFilter"]
