"""Commercial-safe synthetic data sources for training.

Replaces the non-commercial UnityEyes + FFHQ dependency with three
MIT-licensed sources: Microsoft DigiFace-1M, Microsoft FaceSynthetics,
and custom Blender eye renders. Each source exposes an
Iterable[np.ndarray] of BGR uint8 eye patches at (EYE_ROI_H, EYE_ROI_W, 3).
"""
