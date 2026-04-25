"""Utilities for extracting mask-computable visual attributes from images.

Example:
```python
from pathlib import Path

from fgvlm.visual_attributes import (
    SizeQuantileCalibrator,
    discover_mask_inputs,
    extract_image_attributes_from_paths,
)

mask_paths = list(Path("/path/to/train_masks").glob("*.png"))
size_calibrator = SizeQuantileCalibrator().fit(mask_paths)

image_path = "/path/to/image.png"
mask_inputs = [
    ("/path/to/image_lesion.png", "lesion"),
    ("/path/to/image_cyst.png", "cyst"),
]
results = extract_image_attributes_from_paths(
    image_path,
    mask_inputs,
    size_calibrator=size_calibrator,
)

auto_mask_inputs = discover_mask_inputs(image_path, "/path/to/masks", object_name="lesion")
more_results = extract_image_attributes_from_paths(
    image_path,
    auto_mask_inputs,
    size_calibrator=size_calibrator,
    split_components=True,
)
```
"""

from .extractor import (
    discover_mask_inputs,
    extract_image_attributes,
    extract_image_attributes_from_paths,
    extract_object_attributes,
)
from .batch import run_folder_extraction
from .size import SizeQuantileCalibrator
from .types import AttributeRules, ObjectMaskInput

__all__ = [
    "AttributeRules",
    "ObjectMaskInput",
    "SizeQuantileCalibrator",
    "discover_mask_inputs",
    "extract_image_attributes",
    "extract_image_attributes_from_paths",
    "extract_object_attributes",
    "run_folder_extraction",
]
