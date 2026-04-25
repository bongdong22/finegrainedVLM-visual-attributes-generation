from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .types import ObjectMeasurements, PathLike

try:
    from skimage import measure as sk_measure
except Exception:  # pragma: no cover - exercised indirectly via fallback tests
    sk_measure = None


def load_image_array(image_path: PathLike) -> np.ndarray:
    """Load an image file as a NumPy array."""

    with Image.open(image_path) as image:
        return np.asarray(image)


def load_mask_array(mask_path: PathLike) -> np.ndarray:
    """Load and normalize a mask file into a 2D boolean array."""

    with Image.open(mask_path) as image:
        return normalize_mask(np.asarray(image))


def normalize_image(image: Any) -> np.ndarray:
    """Normalize an image input into a 2D or 3D NumPy array."""

    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError(
            f"Image must be 2D grayscale or 3D RGB-like. Received shape {array.shape}."
        )
    return array


def normalize_mask(mask: Any) -> np.ndarray:
    """Normalize a mask into a 2D boolean array."""

    array = np.asarray(mask)
    if array.ndim == 2:
        return array > 0
    if array.ndim == 3:
        return np.any(array > 0, axis=-1)
    raise ValueError(f"Mask must be 2D or 3D. Received shape {array.shape}.")


def validate_image_and_mask(image: Any, mask: Any) -> tuple[np.ndarray, np.ndarray]:
    """Normalize image and mask and ensure their spatial sizes match."""

    image_array = normalize_image(image)
    mask_array = normalize_mask(mask)
    if image_array.shape[:2] != mask_array.shape:
        raise ValueError(
            "Image and mask must have the same spatial size. "
            f"Got image {image_array.shape[:2]} and mask {mask_array.shape}."
        )
    return image_array, mask_array


def ensure_non_empty_mask(mask: np.ndarray, *, context: str = "mask") -> None:
    """Raise a clear error when a normalized mask is empty."""

    if not np.any(mask):
        raise ValueError(f"The {context} is empty. Expected at least one foreground pixel.")


def split_connected_components(mask: Any) -> list[np.ndarray]:
    """Split a binary mask into connected-component masks."""

    mask_array = normalize_mask(mask)
    ensure_non_empty_mask(mask_array)

    if sk_measure is not None:
        labeled = sk_measure.label(mask_array.astype(np.uint8), connectivity=2)
        return [(labeled == index) for index in range(1, int(labeled.max()) + 1)]

    num_labels, labeled = cv2.connectedComponents(mask_array.astype(np.uint8), connectivity=8)
    return [(labeled == index) for index in range(1, num_labels)]


def compute_measurements(image: Any, mask: Any) -> ObjectMeasurements:
    """Compute numeric mask measurements for one object."""

    image_array, mask_array = validate_image_and_mask(image, mask)
    ensure_non_empty_mask(mask_array)

    foreground = np.column_stack(np.nonzero(mask_array))
    ys = foreground[:, 0].astype(np.float64)
    xs = foreground[:, 1].astype(np.float64)
    area = float(mask_array.sum())
    image_area = float(mask_array.size)

    centroid = [float(xs.mean()), float(ys.mean())]
    width = float(xs.max() - xs.min() + 1.0)
    height = float(ys.max() - ys.min() + 1.0)
    aspect_ratio = float(max(width, height) / max(1.0, min(width, height)))

    perimeter = _compute_perimeter(mask_array)
    circularity = float((4.0 * math.pi * area) / max(perimeter * perimeter, 1e-8))
    circularity = float(np.clip(circularity, 0.0, 1.0))

    hull_area = _compute_convex_hull_area(mask_array, pixel_area=area)
    solidity = float(area / max(hull_area, 1e-8))
    solidity = float(np.clip(solidity, 0.0, 1.0))

    orientation_angle = _compute_orientation_angle(mask_array)

    return ObjectMeasurements(
        centroid=centroid,
        area_ratio=float(area / image_area),
        aspect_ratio=aspect_ratio,
        circularity=circularity,
        solidity=solidity,
        orientation_angle=orientation_angle,
    )


def centroid_to_location_label(
    centroid: list[float] | tuple[float, float],
    image_shape: tuple[int, int],
    *,
    x_splits: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0),
    y_splits: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0),
) -> str:
    """Map a centroid to a 3x3 image-plane location label."""

    width = float(image_shape[1])
    height = float(image_shape[0])
    x_norm = float(centroid[0]) / width
    y_norm = float(centroid[1]) / height

    col_label = _grid_axis_label(x_norm, x_splits, ("left", "center", "right"))
    row_label = _grid_axis_label(y_norm, y_splits, ("upper", "middle", "lower"))
    if row_label == "middle" and col_label == "center":
        return "center"
    return f"{row_label}-{col_label}"


def _grid_axis_label(
    value: float,
    splits: tuple[float, float],
    labels: tuple[str, str, str],
) -> str:
    if value < splits[0]:
        return labels[0]
    if value < splits[1]:
        return labels[1]
    return labels[2]


def _compute_perimeter(mask: np.ndarray) -> float:
    if sk_measure is not None:
        try:
            perimeter = float(sk_measure.perimeter(mask, neighborhood=8))
            if perimeter > 0.0:
                return perimeter
        except Exception:
            pass

    contours = _find_contours(mask)
    if not contours:
        return 0.0
    return float(sum(cv2.arcLength(contour, True) for contour in contours))


def _compute_convex_hull_area(mask: np.ndarray, *, pixel_area: float) -> float:
    contours = _find_contours(mask)
    if not contours:
        return pixel_area

    points = np.vstack(contours)
    if len(points) < 3:
        return pixel_area

    hull = cv2.convexHull(points)
    hull_area = float(cv2.contourArea(hull))
    return hull_area if hull_area > 0.0 else pixel_area


def _compute_orientation_angle(mask: np.ndarray) -> float:
    coords = np.column_stack(np.nonzero(mask))
    if len(coords) < 2:
        return 0.0

    xy = coords[:, ::-1].astype(np.float64)
    xy -= xy.mean(axis=0, keepdims=True)
    covariance = np.cov(xy, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle = float(np.degrees(np.arctan2(major_axis[1], major_axis[0])) % 180.0)
    if math.isnan(angle):
        return 0.0
    return angle


def _find_contours(mask: np.ndarray) -> list[np.ndarray]:
    contours_info = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_NONE,
    )
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    return list(contours)
