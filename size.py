from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .geometry import load_mask_array, normalize_mask, split_connected_components
from .types import MaskPathInput, ObjectMaskInput, PathLike, SizeQuantileState


class SizeQuantileCalibrator:
    """Fit dataset-level quantile thresholds for lesion size labels."""

    def __init__(self, quantiles: Sequence[float] = (1.0 / 3.0, 2.0 / 3.0)) -> None:
        if len(quantiles) != 2:
            raise ValueError("SizeQuantileCalibrator expects exactly two quantiles.")
        ordered = tuple(float(value) for value in quantiles)
        if not 0.0 < ordered[0] < ordered[1] < 1.0:
            raise ValueError(f"Quantiles must satisfy 0 < q1 < q2 < 1. Received {ordered}.")
        self.state = SizeQuantileState(quantiles=ordered)

    @property
    def thresholds_(self) -> tuple[float, float]:
        if not self.state.is_fitted:
            raise RuntimeError("Size thresholds are not fitted yet.")
        return self.state.thresholds  # type: ignore[return-value]

    @property
    def area_ratios_(self) -> tuple[float, ...]:
        if not self.state.is_fitted:
            raise RuntimeError("Size thresholds are not fitted yet.")
        return self.state.fitted_area_ratios

    def fit(
        self,
        masks_or_object_inputs: Iterable[np.ndarray | ObjectMaskInput | MaskPathInput | PathLike],
        *,
        split_components: bool = True,
    ) -> "SizeQuantileCalibrator":
        """Fit tertile-like thresholds from object masks."""

        area_ratios: list[float] = []
        for mask in _iter_normalized_masks(masks_or_object_inputs):
            if not np.any(mask):
                continue
            if split_components:
                components = split_connected_components(mask)
            else:
                components = [mask]
            for component in components:
                if np.any(component):
                    area_ratios.append(float(component.sum() / component.size))

        if len(area_ratios) < 3:
            raise ValueError(
                "At least three non-empty object masks are required to fit size tertiles."
            )

        thresholds = tuple(
            float(value) for value in np.quantile(area_ratios, self.state.quantiles)
        )
        self.state = SizeQuantileState(
            quantiles=self.state.quantiles,
            thresholds=thresholds,
            fitted_area_ratios=tuple(float(value) for value in area_ratios),
        )
        return self

    def classify(self, area_ratio: float) -> str:
        """Map an area ratio to `small`, `medium`, or `large`."""

        small_max, medium_max = self.thresholds_
        if area_ratio <= small_max:
            return "small"
        if area_ratio <= medium_max:
            return "medium"
        return "large"


def _iter_normalized_masks(
    masks_or_object_inputs: Iterable[np.ndarray | ObjectMaskInput | MaskPathInput | PathLike],
) -> Iterable[np.ndarray]:
    for item in masks_or_object_inputs:
        if isinstance(item, ObjectMaskInput):
            yield normalize_mask(item.mask)
            continue
        if isinstance(item, tuple):
            mask_path = item[0]
            yield load_mask_array(mask_path)
            continue
        if isinstance(item, (str, Path)):
            yield load_mask_array(item)
            continue
        yield normalize_mask(item)
