"""Blender-side eye-render pipeline.

WARNING: modules in this package that import `bpy` (scene.py) only run
inside Blender (headless or GUI). The parent synthesis package does
NOT import this subpackage, keeping `bpy` out of the ML dependency
graph. Entry is `render_eyes.py`, invoked via:

    blender --background --python render_eyes.py -- --count 500000
"""
