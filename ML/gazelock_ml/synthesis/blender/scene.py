"""Blender scene construction for eye rendering.

Imports `bpy` / `bmesh` / `mathutils` — MUST run inside Blender.
Do not import from the main ML runtime.

Since Phase 2c found zero CC0 3D eye models, the primary geometry
source is `make_procedural_eye`, which constructs an eye from
Blender primitives (sclera, iris, pupil, cornea) with shader-node
materials. `load_eye_asset` is retained for future use when CC0
assets become available.
"""

from __future__ import annotations

from pathlib import Path

try:
    import bmesh
    import bpy
    import mathutils

    _BPY_AVAILABLE = True
except ImportError:
    _BPY_AVAILABLE = False
    bpy = None  # type: ignore[assignment]
    bmesh = None  # type: ignore[assignment]
    mathutils = None  # type: ignore[assignment]


def _require_bpy() -> None:
    if not _BPY_AVAILABLE:
        raise RuntimeError(
            "scene.py must run inside Blender (bpy not available). "
            "Invoke via: blender --background --python render_eyes.py"
        )


def setup_scene(render_w: int = 256, render_h: int = 256, samples: int = 64) -> None:
    _require_bpy()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.device = "GPU"
    scene.render.resolution_x = render_w
    scene.render.resolution_y = render_h
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    if "RenderCam" not in bpy.data.objects:
        cam_data = bpy.data.cameras.new(name="RenderCam")
        cam_obj = bpy.data.objects.new("RenderCam", cam_data)
        bpy.context.collection.objects.link(cam_obj)
        cam_obj.location = (0.0, 0.0, 0.05)
        # Default Blender camera looks down its local -Z; at (0, 0, +0.05)
        # with rotation (0, 0, 0) it looks toward world origin where the
        # eye is built. Do NOT rotate — doing so points the camera away.
        cam_obj.rotation_euler = (0.0, 0.0, 0.0)
        scene.camera = cam_obj
    if "Cube" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Cube"], do_unlink=True)


def clear_scene() -> None:
    _require_bpy()
    for obj in list(bpy.context.scene.objects):
        if obj.type == "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)


def _hex_to_rgba(hex_rgb: str) -> tuple[float, float, float, float]:
    if hex_rgb.startswith("#"):
        hex_rgb = hex_rgb[1:]
    r = int(hex_rgb[0:2], 16) / 255.0
    g = int(hex_rgb[2:4], 16) / 255.0
    b = int(hex_rgb[4:6], 16) / 255.0
    return (r, g, b, 1.0)


def _new_mesh_obj(name: str) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def _principled_material(
    name: str, base_color: tuple[float, float, float, float], roughness: float = 0.5
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Roughness"].default_value = roughness
    out = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _make_sclera() -> bpy.types.Object:
    """White of the eye: UV sphere radius 12mm, soft off-white with faint pink tint."""
    obj = _new_mesh_obj("Sclera")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=0.012)
    bm.to_mesh(obj.data)
    bm.free()
    for p in obj.data.polygons:
        p.use_smooth = True
    mat = _principled_material("ScleraMat", (0.93, 0.90, 0.89, 1.0), roughness=0.4)
    # Add subtle noise variation for vascular hint
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 25.0
    noise.inputs["Detail"].default_value = 4.0
    color_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp.color_ramp.elements[0].color = (0.93, 0.90, 0.89, 1.0)
    color_ramp.color_ramp.elements[1].color = (0.85, 0.78, 0.78, 1.0)
    mat.node_tree.links.new(noise.outputs["Fac"], color_ramp.inputs["Fac"])
    mat.node_tree.links.new(color_ramp.outputs["Color"], bsdf.inputs["Base Color"])
    obj.data.materials.append(mat)
    return obj


def _make_iris(iris_colour_hex: str) -> bpy.types.Object:
    """Coloured iris disc, radius 6mm, inset slightly behind cornea."""
    obj = _new_mesh_obj("Iris")
    bm = bmesh.new()
    bmesh.ops.create_circle(bm, cap_ends=True, segments=64, radius=0.006)
    bm.to_mesh(obj.data)
    bm.free()
    obj.location = (0.0, 0.0, -0.011)
    mat = _principled_material("IrisMat", _hex_to_rgba(iris_colour_hex), roughness=0.35)
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    # Voronoi striations for iris texture
    voronoi = nodes.new("ShaderNodeTexVoronoi")
    voronoi.inputs["Scale"].default_value = 40.0
    mix = nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MULTIPLY"
    mix.inputs["Fac"].default_value = 0.4
    mix.inputs["Color1"].default_value = _hex_to_rgba(iris_colour_hex)
    mat.node_tree.links.new(voronoi.outputs["Distance"], mix.inputs["Color2"])
    mat.node_tree.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    obj.data.materials.append(mat)
    return obj


def _make_pupil() -> bpy.types.Object:
    """Black pupil disc, radius 1.5mm, in front of iris."""
    obj = _new_mesh_obj("Pupil")
    bm = bmesh.new()
    bmesh.ops.create_circle(bm, cap_ends=True, segments=48, radius=0.0015)
    bm.to_mesh(obj.data)
    bm.free()
    obj.location = (0.0, 0.0, -0.0108)
    mat = _principled_material("PupilMat", (0.0, 0.0, 0.0, 1.0), roughness=1.0)
    obj.data.materials.append(mat)
    return obj


def _make_cornea() -> bpy.types.Object:
    """Transparent front cap: UV sphere trimmed to front 30%, glass BSDF."""
    obj = _new_mesh_obj("Cornea")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=0.013)
    # Delete everything behind z = -0.013 * 0.7 (keep front ~30%)
    verts_to_remove = [v for v in bm.verts if v.co.z < -0.013 * 0.4]
    bmesh.ops.delete(bm, geom=verts_to_remove, context="VERTS")
    bm.to_mesh(obj.data)
    bm.free()
    for p in obj.data.polygons:
        p.use_smooth = True
    mat = bpy.data.materials.new("CorneaMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    glass = nodes.new("ShaderNodeBsdfGlass")
    glass.inputs["IOR"].default_value = 1.376
    glass.inputs["Roughness"].default_value = 0.02
    out = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(glass.outputs["BSDF"], out.inputs["Surface"])
    obj.data.materials.append(mat)
    return obj


def make_procedural_eye(iris_colour_hex: str) -> bpy.types.Object:
    """Build a complete procedural eye; returns the sclera (root) object."""
    _require_bpy()
    sclera = _make_sclera()
    iris = _make_iris(iris_colour_hex)
    pupil = _make_pupil()
    cornea = _make_cornea()
    for child in (iris, pupil, cornea):
        child.parent = sclera
    return sclera


def load_eye_asset(asset_path: Path) -> bpy.types.Object:
    """Load external 3D eye model (CC0). Currently unused — retained
    for when CC0 eye assets become available."""
    _require_bpy()
    suffix = asset_path.suffix.lower()
    if suffix == ".glb":
        bpy.ops.import_scene.gltf(filepath=str(asset_path))
    elif suffix == ".blend":
        with bpy.data.libraries.load(str(asset_path), link=False) as (src, dst):
            dst.objects = list(src.objects)
        for obj in dst.objects:
            if obj is not None:
                bpy.context.scene.collection.objects.link(obj)
    else:
        raise ValueError(f"Unsupported eye asset format: {suffix}")
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh loaded from {asset_path}")
    return meshes[-1]


def apply_gaze(obj: bpy.types.Object, gaze_vec: tuple[float, float, float]) -> None:
    _require_bpy()
    vec = mathutils.Vector(gaze_vec).normalized()
    matrix = vec.to_track_quat("-Z", "Y").to_matrix().to_4x4()
    obj.matrix_world = matrix


def apply_lighting(hdri_path: Path, rotation_z_rad: float) -> None:
    _require_bpy()
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    env = nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(hdri_path))
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value[2] = rotation_z_rad
    tex_coord = nodes.new("ShaderNodeTexCoord")
    bg = nodes.new("ShaderNodeBackground")
    output = nodes.new("ShaderNodeOutputWorld")
    links = world.node_tree.links
    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    links.new(env.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], output.inputs["Surface"])


def render_frame(output_path: Path) -> None:
    _require_bpy()
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


__all__ = [
    "apply_gaze",
    "apply_lighting",
    "clear_scene",
    "load_eye_asset",
    "make_procedural_eye",
    "render_frame",
    "setup_scene",
]
