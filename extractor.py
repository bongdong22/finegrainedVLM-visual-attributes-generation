from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from .geometry import (
    centroid_to_location_label,
    compute_measurements,
    load_image_array,
    load_mask_array,
    normalize_image,
    normalize_mask,
    split_connected_components,
)
from .size import SizeQuantileCalibrator
from .types import AttributeRules, MaskPathInput, ObjectMaskInput, PathLike, Proposition


def extract_object_attributes(
    image: Any,
    mask: Any,
    *,
    object_name: str = "lesion",
    size_calibrator: SizeQuantileCalibrator,
    rules: AttributeRules | None = None,
    instance_index: int = 0,
    image_path: str | None = None,
    mask_path: str | None = None,
) -> dict[str, Any]:
    """Extract one object's visual attributes from an image and mask."""

    active_rules = rules or AttributeRules()
    image_array = normalize_image(image)
    measurements = compute_measurements(image_array, mask)

    location_label = centroid_to_location_label(
        measurements.centroid,
        image_array.shape[:2],
        x_splits=active_rules.location_x_splits,
        y_splits=active_rules.location_y_splits,
    )
    size_label = size_calibrator.classify(measurements.area_ratio)
    shape_label = _classify_shape(measurements.aspect_ratio, measurements.circularity, active_rules)
    orientation_label = _classify_orientation(measurements.orientation_angle, active_rules)
    boundary_label = _classify_boundary(
        measurements.solidity,
        measurements.circularity,
        active_rules,
    )

    propositions: list[Proposition] = [
        [object_name, "location", location_label],
        [object_name, "size", size_label],
        [object_name, "shape", shape_label],
        [object_name, "orientation", orientation_label],
        [object_name, "boundary", boundary_label],
    ]

    result: dict[str, Any] = {
        "object": object_name,
        "instance_index": instance_index,
        "measurements": measurements.to_dict(),
        "propositions": propositions,
    }
    if image_path is not None:
        result["image_path"] = image_path
    if mask_path is not None:
        result["mask_path"] = mask_path
    return result


def extract_image_attributes(
    image: Any,
    object_masks: Sequence[ObjectMaskInput],
    *,
    image_path: str | None = None,
    size_calibrator: SizeQuantileCalibrator,
    rules: AttributeRules | None = None,
    split_components: bool = False,
) -> list[dict[str, Any]]:
    """Extract attributes for every object described by a list of masks."""

    image_array = normalize_image(image)
    active_rules = rules or AttributeRules()
    results: list[dict[str, Any]] = []
    instance_index = 0

    for object_input in object_masks:
        normalized_mask = normalize_mask(object_input.mask)
        masks = (
            split_connected_components(normalized_mask)
            if split_components
            else [normalized_mask]
        )
        for component_mask in masks:
            results.append(
                extract_object_attributes(
                    image_array,
                    component_mask,
                    object_name=object_input.object_name,
                    size_calibrator=size_calibrator,
                    rules=active_rules,
                    instance_index=instance_index,
                    image_path=image_path,
                    mask_path=object_input.mask_path,
                )
            )
            instance_index += 1

    return results


def extract_image_attributes_from_paths(
    image_path: PathLike,
    mask_inputs: Sequence[MaskPathInput],
    *,
    size_calibrator: SizeQuantileCalibrator,
    rules: AttributeRules | None = None,
    split_components: bool = False,
) -> list[dict[str, Any]]:
    """Load one image and its mask files from disk and extract object attributes."""

    image_array = load_image_array(image_path)
    object_masks = [
        ObjectMaskInput(mask=load_mask_array(mask_path), object_name=object_name, mask_path=str(mask_path))
        for mask_path, object_name in _normalize_mask_inputs(mask_inputs)
    ]
    return extract_image_attributes(
        image_array,
        object_masks,
        image_path=str(image_path),
        size_calibrator=size_calibrator,
        rules=rules,
        split_components=split_components,
    )


def discover_mask_inputs(
    image_path: PathLike,
    mask_dir: PathLike,
    *,
    glob_pattern: str = "*.png",
    stem_match: bool = True,
    object_name: str = "lesion",
) -> list[tuple[str, str]]:
    """Discover likely mask files for an image from a directory."""

    image_stem = Path(image_path).stem
    candidates = sorted(Path(mask_dir).glob(glob_pattern))
    matched: list[tuple[str, str]] = []
    for candidate in candidates:
        if not stem_match or _mask_matches_image_stem(image_stem, candidate.stem):
            matched.append((str(candidate), object_name))
    return matched


def _normalize_mask_inputs(mask_inputs: Sequence[MaskPathInput]) -> list[tuple[PathLike, str]]:
    normalized: list[tuple[PathLike, str]] = []
    for item in mask_inputs:
        if isinstance(item, tuple):
            normalized.append((item[0], item[1]))
        else:
            normalized.append((item, "lesion"))
    return normalized


def _mask_matches_image_stem(image_stem: str, mask_stem: str) -> bool:
    return (
        mask_stem == image_stem
        or mask_stem.startswith(f"{image_stem}_")
        or image_stem in mask_stem.split("_")
        or image_stem in mask_stem
    )


def _classify_shape(
    aspect_ratio: float,
    circularity: float,
    rules: AttributeRules,
) -> str:
    if circularity < rules.irregular_shape_circularity_max:
        return "irregular"
    if aspect_ratio <= rules.round_aspect_ratio_max and circularity >= rules.round_circularity_min:
        return "round"
    if aspect_ratio >= rules.elongated_aspect_ratio_min:
        return "elongated"
    return "oval"


def _classify_orientation(angle: float, rules: AttributeRules) -> str:
    if angle <= rules.horizontal_angle_max or angle >= (180.0 - rules.horizontal_angle_max):
        return "horizontal"
    if rules.vertical_angle_min <= angle <= rules.vertical_angle_max:
        return "vertical"
    return "oblique"


def _classify_boundary(solidity: float, circularity: float, rules: AttributeRules) -> str:
    if solidity >= rules.smooth_solidity_min and circularity >= rules.smooth_circularity_min:
        return "smooth"
    if solidity >= rules.mild_solidity_min and circularity >= rules.mild_circularity_min:
        return "mildly-irregular"
    return "irregular"
