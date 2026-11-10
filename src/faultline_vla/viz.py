"""Shared figure style. Colors are the validated default categorical slots."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SUITE_COLORS = {"libero_spatial": "#2a78d6", "libero_object": "#eb6834"}
SUITE_LABELS = {"libero_spatial": "LIBERO-Spatial", "libero_object": "LIBERO-Object"}
AXIS_LABELS = {
    "light_dim": "Light dimming",
    "light_bright": "Light brightening",
    "light_direction": "Light direction",
    "light_color": "Light color",
    "texture": "Surface texture",
    "camera_pose": "Camera pose",
    "camera_fov": "Camera FOV",
    "distractors": "Distractors",
    "sensor_noise": "Sensor noise",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "text.color": TEXT,
            "font.size": 10,
            "axes.titlesize": 11,
            "lines.linewidth": 2,
            "legend.frameon": False,
        }
    )
