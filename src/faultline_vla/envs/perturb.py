"""Deterministic visual perturbations for LIBERO scenes in MuJoCo.

Every axis takes a magnitude ``m`` in [0, 1], where 0 is the unperturbed scene.
Perturbations change rendering only: they never change physics state.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

AXES = (
    "light_dim",
    "light_bright",
    "light_direction",
    "light_color",
    "texture",
    "camera_pose",
    "camera_fov",
    "distractors",
    "sensor_noise",
)
SCENE_AXES = tuple(axis for axis in AXES if axis != "sensor_noise")

AGENTVIEW = "agentview"
DIM_FACTOR = 0.2
BRIGHT_FACTOR = 2.0
MAX_LIGHT_ANGLE_DEG = 90.0
MAX_CAMERA_ANGLE_DEG = 20.0
MAX_FOV_CHANGE_DEG = 15.0
MAX_NOISE_SIGMA = 0.15
MAX_DISTRACTORS = 4
PARKED_Z = -5.0
SURFACE_GEOMS = ("table_visual", "floor")
LIGHT_TINTS = np.array(
    [
        [1.0, 0.25, 0.25],
        [1.0, 0.65, 0.2],
        [0.3, 1.0, 0.3],
        [0.25, 0.9, 1.0],
        [0.3, 0.4, 1.0],
        [1.0, 0.3, 0.9],
    ]
)
DISTRACTOR_NAMES = ("red_coffee_mug", "porcelain_mug", "moka_pot", "black_book")
DISTRACTOR_PREFIX = "distractor_"
DISTRACTOR_MARGIN = 0.07
SAMPLE_HALF_RANGE = np.array([0.22, 0.28])
MAX_PLACEMENT_TRIES = 200


def axis_rng(axis: str, seed: int) -> np.random.Generator:
    """Return a generator determined only by the axis and seed."""
    digest = hashlib.sha256(f"{axis}:{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def validate(axis: str, m: float) -> None:
    if axis not in AXES:
        raise ValueError(f"Unknown axis {axis!r}. Expected one of {AXES}.")
    if not 0.0 <= m <= 1.0:
        raise ValueError(f"Magnitude must be in [0, 1], got {m}.")


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return the matrix for a rotation of ``angle`` radians about ``axis``."""
    x, y, z = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def add_distractor_bodies(env) -> dict[str, float]:
    """Add visual-only distractor bodies, parked out of view, to a LIBERO env.

    The bodies have no joints and no collisions, so the state vector and physics
    are the same as in the unmodified task. Returns each body's bottom offset.
    """
    from libero.libero.envs.objects import get_object_fn

    base = env.env
    # Reloading from XML loses what the last reset wrote into the compiled model: fixture
    # placements (cabinet, stove) go back to the origin. Parked bodies also widen the model
    # statistics that scale rendering. Save both and put them back after the reload.
    old_model = base.sim.model._model
    placements = {}
    for body in range(old_model.nbody):
        name = mujoco.mj_id2name(old_model, mujoco.mjtObj.mjOBJ_BODY, body)
        placements[name] = (old_model.body_pos[body].copy(), old_model.body_quat[body].copy())
    statistics = {
        field: copy.copy(getattr(old_model.stat, field))
        for field in ("meaninertia", "meanmass", "meansize", "extent", "center")
    }
    bottom_offsets = {}
    for index, object_name in enumerate(DISTRACTOR_NAMES):
        instance = f"{DISTRACTOR_PREFIX}{index}"
        distractor = get_object_fn(object_name)(name=instance)
        element = distractor.get_obj()
        for parent in element.iter():
            for child in list(parent):
                if child.tag in ("joint", "freejoint"):
                    parent.remove(child)
        for geom in element.iter("geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
        element.set("name", instance)
        element.set("pos", f"0 0 {PARKED_Z}")
        base.model.merge_assets(distractor)
        base.model.worldbody.append(copy.deepcopy(element))
        bottom_offsets[instance] = float(distractor.bottom_offset[2])
    base.reset_from_xml_string(base.model.get_xml())
    model = base.sim.model._model
    for name, (position, quaternion) in placements.items():
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body >= 0:
            model.body_pos[body] = position
            model.body_quat[body] = quaternion
    for field, value in statistics.items():
        setattr(model.stat, field, value)
    base.sim.forward()
    return bottom_offsets


class Perturber:
    """Applies one perturbation at a time to a LIBERO env's render settings."""

    def __init__(self, env, textures_dir: str | Path, bottom_offsets: dict[str, float]):
        self.env = env
        self.sim = env.env.sim
        self.model = self.sim.model._model
        self.data = self.sim.data._data
        self.textures_dir = Path(textures_dir)
        self.bottom_offsets = bottom_offsets
        self.camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, AGENTVIEW
        )
        self.texture_id = self._find_surface_texture()
        self.distractor_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in bottom_offsets
        ]
        if min(self.distractor_ids, default=0) < 0:
            raise ValueError("A distractor body is missing from the model.")
        self.texture_files = sorted(self.textures_dir.glob("*.png"))
        self._save_defaults()

    def _find_surface_texture(self) -> int:
        for name in SURFACE_GEOMS:
            geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom < 0:
                continue
            material = self.model.geom_matid[geom]
            texture = self.model.mat_texid[material, mujoco.mjtTextureRole.mjTEXROLE_RGB]
            if material >= 0 and texture >= 0:
                return int(texture)
        raise ValueError("No known surface texture in this scene.")

    def _texture_slice(self) -> tuple[slice, tuple[int, int, int]]:
        i = self.texture_id
        start = int(self.model.tex_adr[i])
        height = int(self.model.tex_height[i])
        width = int(self.model.tex_width[i])
        channels = int(self.model.tex_nchannel[i])
        return slice(start, start + height * width * channels), (height, width, channels)

    def _save_defaults(self) -> None:
        model = self.model
        self.default_diffuse = model.light_diffuse.copy()
        self.default_ambient = model.light_ambient.copy()
        self.default_specular = model.light_specular.copy()
        self.default_light_pos = model.light_pos.copy()
        self.default_light_dir = model.light_dir.copy()
        self.default_cam_pos = model.cam_pos[self.camera_id].copy()
        self.default_cam_quat = model.cam_quat[self.camera_id].copy()
        self.default_fovy = float(model.cam_fovy[self.camera_id])
        data_slice, _ = self._texture_slice()
        self.default_texture = model.tex_data[data_slice].copy()
        self.default_body_pos = model.body_pos[self.distractor_ids].copy()
        self.texture_changed = False

    def reset(self) -> None:
        """Restore every render setting to the unperturbed scene."""
        model = self.model
        model.light_diffuse[:] = self.default_diffuse
        model.light_ambient[:] = self.default_ambient
        model.light_specular[:] = self.default_specular
        model.light_pos[:] = self.default_light_pos
        model.light_dir[:] = self.default_light_dir
        model.cam_pos[self.camera_id] = self.default_cam_pos
        model.cam_quat[self.camera_id] = self.default_cam_quat
        model.cam_fovy[self.camera_id] = self.default_fovy
        model.body_pos[self.distractor_ids] = self.default_body_pos
        if self.texture_changed:
            data_slice, _ = self._texture_slice()
            model.tex_data[data_slice] = self.default_texture
            self._upload_texture()
            self.texture_changed = False
        self.sim.forward()

    def apply(self, axis: str, m: float, seed: int) -> None:
        """Reset the scene, then apply ``axis`` at magnitude ``m``."""
        validate(axis, m)
        self.reset()
        if m == 0.0 or axis == "sensor_noise":
            return
        rng = axis_rng(axis, seed)
        getattr(self, f"_apply_{axis}")(m, rng)
        self.sim.forward()

    def snapshot(self) -> dict[str, np.ndarray | None]:
        """Copy the current render settings so a perturbation can be held fixed."""
        model = self.model
        data_slice, _ = self._texture_slice()
        return {
            "light_diffuse": model.light_diffuse.copy(),
            "light_ambient": model.light_ambient.copy(),
            "light_specular": model.light_specular.copy(),
            "light_pos": model.light_pos.copy(),
            "light_dir": model.light_dir.copy(),
            "cam_pos": model.cam_pos[self.camera_id].copy(),
            "cam_quat": model.cam_quat[self.camera_id].copy(),
            "cam_fovy": model.cam_fovy[self.camera_id].copy(),
            "body_pos": model.body_pos[self.distractor_ids].copy(),
            "texture": model.tex_data[data_slice].copy() if self.texture_changed else None,
        }

    def restore(self, saved: dict[str, np.ndarray | None]) -> None:
        """Apply settings from :meth:`snapshot` without drawing a new perturbation."""
        model = self.model
        model.light_diffuse[:] = saved["light_diffuse"]
        model.light_ambient[:] = saved["light_ambient"]
        model.light_specular[:] = saved["light_specular"]
        model.light_pos[:] = saved["light_pos"]
        model.light_dir[:] = saved["light_dir"]
        model.cam_pos[self.camera_id] = saved["cam_pos"]
        model.cam_quat[self.camera_id] = saved["cam_quat"]
        model.cam_fovy[self.camera_id] = saved["cam_fovy"]
        model.body_pos[self.distractor_ids] = saved["body_pos"]
        if saved["texture"] is not None:
            data_slice, _ = self._texture_slice()
            model.tex_data[data_slice] = saved["texture"]
            self._upload_texture()
            self.texture_changed = True
        self.sim.forward()

    def _scale_lights(self, factor: float) -> None:
        self.model.light_diffuse[:] = self.default_diffuse * factor
        self.model.light_ambient[:] = self.default_ambient * factor
        self.model.light_specular[:] = self.default_specular * factor

    def _apply_light_dim(self, m: float, rng: np.random.Generator) -> None:
        self._scale_lights(DIM_FACTOR**m)

    def _apply_light_bright(self, m: float, rng: np.random.Generator) -> None:
        self._scale_lights(BRIGHT_FACTOR**m)

    def _apply_light_direction(self, m: float, rng: np.random.Generator) -> None:
        heading = rng.uniform(0.0, 2.0 * np.pi)
        axis = np.array([np.cos(heading), np.sin(heading), 0.0])
        rotation = rotation_matrix(axis, np.deg2rad(m * MAX_LIGHT_ANGLE_DEG))
        self.model.light_dir[:] = (rotation @ self.default_light_dir.T).T
        self.model.light_pos[:] = (rotation @ self.default_light_pos.T).T

    def _apply_light_color(self, m: float, rng: np.random.Generator) -> None:
        tint = LIGHT_TINTS[rng.integers(len(LIGHT_TINTS))]
        tinted = self.default_diffuse.max(axis=1, keepdims=True) * tint
        self.model.light_diffuse[:] = (1.0 - m) * self.default_diffuse + m * tinted

    def _apply_texture(self, m: float, rng: np.random.Generator) -> None:
        path = self.texture_files[rng.integers(len(self.texture_files))]
        data_slice, (height, width, channels) = self._texture_slice()
        with Image.open(path) as image:
            mode = "RGB" if channels == 3 else "RGBA"
            target = np.asarray(image.convert(mode).resize((width, height)), dtype=np.float32)
        blended = (1.0 - m) * self.default_texture.reshape(height, width, channels) + m * target
        self.model.tex_data[data_slice] = np.clip(blended, 0, 255).astype(np.uint8).ravel()
        self._upload_texture()
        self.texture_changed = True

    def _upload_texture(self) -> None:
        # robosuite's upload_texture passes its own model wrapper, which MuJoCo rejects.
        context = self.sim._render_context_offscreen
        context.gl_ctx.make_current()
        mujoco.mjr_uploadTexture(self.model, context.con, self.texture_id)

    def _look_target(self) -> np.ndarray:
        """Return where the camera's axis meets the scene, or a point 1 m ahead."""
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, self.default_cam_quat)
        forward = -rotation.reshape(3, 3)[:, 2]
        origin = self.default_cam_pos.astype(np.float64)
        geom_id = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
            self.model, self.data, origin, forward, None, 1, -1, geom_id
        )
        return origin + forward * (distance if distance > 0 else 1.0)

    def _apply_camera_pose(self, m: float, rng: np.random.Generator) -> None:
        target = self._look_target()
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, self.default_cam_quat)
        rotation = rotation.reshape(3, 3)
        heading = rng.uniform(0.0, 2.0 * np.pi)
        axis = rotation[:, 0] * np.cos(heading) + rotation[:, 1] * np.sin(heading)
        orbit = rotation_matrix(axis, np.deg2rad(m * MAX_CAMERA_ANGLE_DEG))
        self.model.cam_pos[self.camera_id] = target + orbit @ (self.default_cam_pos - target)
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, (orbit @ rotation).ravel())
        self.model.cam_quat[self.camera_id] = quaternion

    def _apply_camera_fov(self, m: float, rng: np.random.Generator) -> None:
        sign = 1.0 if rng.random() < 0.5 else -1.0
        self.model.cam_fovy[self.camera_id] = self.default_fovy + sign * m * MAX_FOV_CHANGE_DEG

    def _surface_height(self, xy: np.ndarray, margin: float) -> float | None:
        """Return the surface height at ``xy`` if the spot and its margin are clear."""
        heights = []
        offsets = [(0, 0), (margin, 0), (-margin, 0), (0, margin), (0, -margin)]
        direction = np.array([0.0, 0.0, -1.0])
        for dx, dy in offsets:
            origin = np.array([xy[0] + dx, xy[1] + dy, 3.0])
            geom_id = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(self.model, self.data, origin, direction, None, 1, -1, geom_id)
            if distance < 0 or geom_id[0] < 0:
                return None
            body = self.model.geom_bodyid[geom_id[0]]
            if self.model.body_rootid[body] != 0 or self.model.body_parentid[body] != 0:
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
                if not name.startswith(("table", "floor")):
                    return None
            heights.append(3.0 - distance)
        if max(heights) - min(heights) > 0.01:
            return None
        return float(np.median(heights))

    def _scene_center(self) -> np.ndarray:
        names = getattr(self.env.env, "obj_of_interest", [])
        positions = []
        for name in names:
            body = self.env.env.obj_body_id.get(name)
            if body is not None:
                positions.append(self.data.xpos[body][:2])
        return np.mean(positions, axis=0) if positions else np.zeros(2)

    def _apply_distractors(self, m: float, rng: np.random.Generator) -> None:
        count = round(m * MAX_DISTRACTORS)
        center = self._scene_center()
        placed: list[np.ndarray] = []
        for body, offset in zip(self.distractor_ids, self.bottom_offsets.values(), strict=True):
            if len(placed) == count:
                break
            for _ in range(MAX_PLACEMENT_TRIES):
                xy = center + rng.uniform(-SAMPLE_HALF_RANGE, SAMPLE_HALF_RANGE)
                if any(np.linalg.norm(xy - other) < 2 * DISTRACTOR_MARGIN for other in placed):
                    continue
                height = self._surface_height(xy, DISTRACTOR_MARGIN)
                if height is None:
                    continue
                self.model.body_pos[body] = [xy[0], xy[1], height - offset]
                placed.append(xy)
                break
            else:
                raise RuntimeError(f"Could not place distractor {len(placed) + 1}.")


def apply_image_axis(image: np.ndarray, axis: str, m: float, seed: int, key: int = 0) -> np.ndarray:
    """Apply image-space perturbations to a uint8 image. Other axes pass through.

    ``key`` distinguishes frames, so each state gets its own noise sample.
    """
    validate(axis, m)
    if axis != "sensor_noise" or m == 0.0:
        return image
    digest = hashlib.sha256(f"{axis}:{seed}:{key}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    noise = rng.normal(0.0, 1.0, image.shape) * m * MAX_NOISE_SIGMA
    return (np.clip(image / 255.0 + noise, 0.0, 1.0) * 255.0).round().astype(np.uint8)
