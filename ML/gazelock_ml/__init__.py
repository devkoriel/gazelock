"""GazeLock ML training & export pipeline.

Subpackages:
    warp     — analytic 3D-eyeball + TPS warp (mirrors Swift spec §6.3)
    data     — dataset loaders (UnityEyes, FFHQ) and synthetic fixtures
    models   — refiner UNet and iris identity encoder
    losses   — reconstruction, perceptual, identity, temporal
"""

__version__ = "0.1.0"
