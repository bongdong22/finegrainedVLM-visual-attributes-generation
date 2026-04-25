from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

PathLike = str | Path
MaskPathInput = PathLike | tuple[PathLike, str]
Proposition = list[str]


@dataclass(slots=True)
class AttributeRules:
    """Configurable thresholds for turning measurements into labels."""

    location_x_splits: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)
    location_y_splits: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)
    round_aspect_ratio_max: float = 1.2
    round_circularity_min: float = 0.80
    irregular_shape_circularity_max: float = 0.45
    elongated_aspect_ratio_min: float = 2.2
    horizontal_angle_max: float = 20.0
    vertical_angle_min: float = 70.0
    vertical_angle_max: float = 110.0
    smooth_solidity_min: float = 0.95
    smooth_circularity_min: float = 0.80
    mild_solidity_min: float = 0.85
    mild_circularity_min: float = 0.55


@dataclass(slots=True)
class ObjectMaskInput:
    """Container for one object mask and its metadata."""

    mask: np.ndarray
    object_name: str = "lesion"
    mask_path: str | None = None


@dataclass(slots=True)
class ObjectMeasurements:
    """Numeric mask-derived measurements used for attribute classification."""

    centroid: list[float]
    area_ratio: float
    aspect_ratio: float
    circularity: float
    solidity: float
    orientation_angle: float

    def to_dict(self) -> dict[str, float | list[float]]:
        return {
            "centroid": self.centroid,
            "area_ratio": self.area_ratio,
            "aspect_ratio": self.aspect_ratio,
            "circularity": self.circularity,
            "solidity": self.solidity,
            "orientation_angle": self.orientation_angle,
        }


@dataclass(slots=True)
class SizeQuantileState:
    """Persisted percentile state for size labeling."""

    quantiles: tuple[float, ...]
    thresholds: tuple[float, ...] = field(default_factory=tuple)
    fitted_area_ratios: tuple[float, ...] = field(default_factory=tuple)

    @property
    def is_fitted(self) -> bool:
        return len(self.thresholds) == len(self.quantiles)
