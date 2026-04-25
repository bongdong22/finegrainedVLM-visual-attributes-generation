from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from .extractor import extract_image_attributes_from_paths
from .size import SizeQuantileCalibrator
from .types import PathLike

GENERIC_OBJECT_TOKENS = {
    "",
    "mask",
    "masks",
    "seg",
    "segs",
    "segmentation",
    "label",
    "labels",
    "gt",
    "annotation",
    "annotations",
}


def run_folder_extraction(
    images_dir: PathLike,
    masks_dir: PathLike,
    output_dir: PathLike,
    *,
    calibration_masks_dir: PathLike | None = None,
    image_glob: str = "*.png",
    mask_glob: str = "*.png",
    object_name: str | None = None,
    object_map_csv: PathLike | None = None,
    infer_object_name: bool = False,
    split_components: bool = False,
    fit_split_components: bool = True,
    fail_on_missing_masks: bool = False,
) -> dict[str, Any]:
    """Run folder-based attribute extraction and save JSON outputs."""

    images_root = Path(images_dir)
    masks_root = Path(masks_dir)
    output_root = Path(output_dir)
    calibration_root = Path(calibration_masks_dir) if calibration_masks_dir else masks_root

    if object_map_csv is None and object_name is None and not infer_object_name:
        raise ValueError(
            "Provide either `object_name`, `object_map_csv`, or enable `infer_object_name`."
        )

    image_paths = _collect_files(images_root, image_glob)
    if not image_paths:
        raise ValueError(f"No images found in {images_root} with pattern {image_glob}.")

    all_mask_paths = _collect_files(masks_root, mask_glob)
    if not all_mask_paths:
        raise ValueError(f"No masks found in {masks_root} with pattern {mask_glob}.")

    calibration_mask_paths = _collect_files(calibration_root, mask_glob)
    if not calibration_mask_paths:
        raise ValueError(
            f"No calibration masks found in {calibration_root} with pattern {mask_glob}."
        )

    size_calibrator = SizeQuantileCalibrator().fit(
        calibration_mask_paths,
        split_components=fit_split_components,
    )
    object_map = (
        load_object_map_csv(
            object_map_csv,
            images_dir=images_root,
            masks_dir=masks_root,
        )
        if object_map_csv is not None
        else None
    )

    per_image_root = output_root / "per_image"
    per_image_root.mkdir(parents=True, exist_ok=True)

    processed_images: list[dict[str, Any]] = []
    skipped_images: list[dict[str, str]] = []
    total_objects = 0

    for image_path in image_paths:
        if object_map is not None:
            mask_inputs = object_map.get(_normalize_path_key(image_path), [])
        else:
            mask_inputs = build_mask_inputs_for_image(
                image_path=image_path,
                masks_dir=masks_root,
                mask_paths=all_mask_paths,
                object_name=object_name,
                infer_object_name=infer_object_name,
            )

        if not mask_inputs:
            if fail_on_missing_masks:
                raise ValueError(f"No masks matched image {image_path}.")
            skipped_images.append(
                {
                    "image_path": str(image_path),
                    "reason": "no_matching_masks",
                }
            )
            continue

        objects = extract_image_attributes_from_paths(
            image_path=image_path,
            mask_inputs=mask_inputs,
            size_calibrator=size_calibrator,
            split_components=split_components,
        )
        total_objects += len(objects)

        image_relative = image_path.relative_to(images_root)
        per_image_output_path = (per_image_root / image_relative).with_suffix(".json")
        per_image_output_path.parent.mkdir(parents=True, exist_ok=True)

        image_record = {
            "image_path": str(image_path),
            "image_relative_path": str(image_relative),
            "num_objects": len(objects),
            "objects": objects,
        }
        _write_json(per_image_output_path, image_record)

        processed_images.append(
            {
                "image_path": str(image_path),
                "image_relative_path": str(image_relative),
                "num_objects": len(objects),
                "result_path": str(per_image_output_path),
            }
        )

    dataset_results = {
        "config": {
            "images_dir": str(images_root),
            "masks_dir": str(masks_root),
            "output_dir": str(output_root),
            "calibration_masks_dir": str(calibration_root),
            "image_glob": image_glob,
            "mask_glob": mask_glob,
            "object_name": object_name,
            "object_map_csv": str(object_map_csv) if object_map_csv is not None else None,
            "infer_object_name": infer_object_name,
            "split_components": split_components,
            "fit_split_components": fit_split_components,
            "fail_on_missing_masks": fail_on_missing_masks,
        },
        "size_thresholds": list(size_calibrator.thresholds_),
        "num_images_total": len(image_paths),
        "num_images_processed": len(processed_images),
        "num_images_skipped": len(skipped_images),
        "num_objects_total": total_objects,
        "processed_images": processed_images,
        "skipped_images": skipped_images,
    }

    _write_json(output_root / "summary.json", dataset_results)

    full_results = {
        **dataset_results,
        "images": [
            json.loads(Path(item["result_path"]).read_text(encoding="utf-8"))
            for item in processed_images
        ],
    }
    _write_json(output_root / "all_results.json", full_results)
    return dataset_results


def build_mask_inputs_for_image(
    image_path: PathLike,
    masks_dir: PathLike,
    mask_paths: list[Path],
    *,
    object_name: str | None = None,
    infer_object_name: bool = False,
) -> list[tuple[str, str]]:
    """Build `(mask_path, object_name)` inputs for one image."""

    image_path = Path(image_path)
    masks_root = Path(masks_dir)
    image_stem = image_path.stem
    matched_paths = [mask_path for mask_path in mask_paths if _mask_matches_image_stem(image_stem, mask_path.stem)]

    mask_inputs: list[tuple[str, str]] = []
    for mask_path in matched_paths:
        resolved_object_name = (
            infer_object_name_from_mask_path(
                image_stem=image_stem,
                mask_path=mask_path,
                masks_dir=masks_root,
                fallback_object_name=object_name,
            )
            if infer_object_name
            else object_name
        )
        if resolved_object_name is None:
            raise ValueError(
                "Object name is required. Provide `object_name`, `object_map_csv`, "
                "or enable `infer_object_name` with informative mask filenames."
            )
        mask_inputs.append((str(mask_path), resolved_object_name))
    return sorted(mask_inputs, key=lambda item: item[0])


def infer_object_name_from_mask_path(
    *,
    image_stem: str,
    mask_path: PathLike,
    masks_dir: PathLike,
    fallback_object_name: str | None = None,
) -> str:
    """Infer an object name from a mask filename or its parent directory."""

    mask_path = Path(mask_path)
    masks_root = Path(masks_dir)

    suffix_candidate = _extract_object_suffix(image_stem, mask_path.stem)
    cleaned_suffix = _clean_object_token(suffix_candidate)
    if cleaned_suffix:
        return cleaned_suffix

    try:
        relative_parent = mask_path.parent.relative_to(masks_root)
        parent_parts = relative_parent.parts
    except ValueError:
        parent_parts = mask_path.parent.parts

    for part in reversed(parent_parts):
        cleaned_part = _clean_object_token(part)
        if cleaned_part:
            return cleaned_part

    if fallback_object_name is not None:
        return fallback_object_name
    raise ValueError(
        f"Could not infer object name for mask `{mask_path}`. "
        "Provide `object_name` or `object_map_csv`."
    )


def load_object_map_csv(
    object_map_csv: PathLike,
    *,
    images_dir: PathLike,
    masks_dir: PathLike,
) -> dict[str, list[tuple[str, str]]]:
    """Load explicit image-mask-object mappings from a CSV file."""

    csv_path = Path(object_map_csv)
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file `{csv_path}` must have a header row.")

        image_column = _find_csv_column(
            reader.fieldnames,
            ("image_path", "image_filename", "image_name", "image"),
        )
        mask_column = _find_csv_column(
            reader.fieldnames,
            ("mask_path", "mask_filename", "mask_name", "mask"),
        )
        object_column = _find_csv_column(
            reader.fieldnames,
            ("object_name", "object", "label", "class_name"),
        )

        mapping: dict[str, list[tuple[str, str]]] = {}
        for row_index, row in enumerate(reader, start=2):
            image_value = _require_csv_value(row, image_column, csv_path, row_index)
            mask_value = _require_csv_value(row, mask_column, csv_path, row_index)
            object_name = _require_csv_value(row, object_column, csv_path, row_index)

            resolved_image_path = _resolve_dataset_path(image_value, images_dir)
            resolved_mask_path = _resolve_dataset_path(mask_value, masks_dir)

            if not resolved_image_path.is_file():
                raise ValueError(
                    f"CSV row {row_index} references missing image file `{resolved_image_path}`."
                )
            if not resolved_mask_path.is_file():
                raise ValueError(
                    f"CSV row {row_index} references missing mask file `{resolved_mask_path}`."
                )

            mapping.setdefault(_normalize_path_key(resolved_image_path), []).append(
                (str(resolved_mask_path), object_name)
            )

    return {
        image_key: sorted(mask_inputs, key=lambda item: item[0])
        for image_key, mask_inputs in mapping.items()
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for folder-based extraction."""

    parser = argparse.ArgumentParser(
        description="Extract structured visual attributes from image and mask folders."
    )
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--masks-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--calibration-masks-dir", type=str, default=None)
    parser.add_argument("--image-glob", type=str, default="*.png")
    parser.add_argument("--mask-glob", type=str, default="*.png")
    parser.add_argument(
        "--object-name",
        type=str,
        default=None,
        help="Single object name applied to all matched masks, e.g. `tumor`.",
    )
    parser.add_argument(
        "--object-map-csv",
        type=str,
        default=None,
        help=(
            "CSV file that explicitly maps each mask to an object name. "
            "Supported columns include image_path/image_filename, "
            "mask_path/mask_filename, object_name/object."
        ),
    )
    parser.add_argument(
        "--infer-object-name",
        action="store_true",
        help=(
            "Infer object names from mask filenames such as `case001_tumor.png`. "
            "If a mask filename does not contain an object name, combine this "
            "with --object-name to provide a fallback."
        ),
    )
    parser.add_argument(
        "--split-components",
        action="store_true",
        help="Split disconnected regions inside one mask into separate object instances.",
    )
    parser.add_argument(
        "--no-fit-split-components",
        dest="fit_split_components",
        action="store_false",
        help="Fit size quantiles on whole masks instead of connected components.",
    )
    parser.add_argument(
        "--fail-on-missing-masks",
        action="store_true",
        help="Stop immediately if an image has no matching masks.",
    )
    parser.set_defaults(fit_split_components=True)
    return parser


def main() -> None:
    """CLI entry point for folder-based extraction."""

    args = build_arg_parser().parse_args()
    summary = run_folder_extraction(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
        calibration_masks_dir=args.calibration_masks_dir,
        image_glob=args.image_glob,
        mask_glob=args.mask_glob,
        object_name=args.object_name,
        object_map_csv=args.object_map_csv,
        infer_object_name=args.infer_object_name,
        split_components=args.split_components,
        fit_split_components=args.fit_split_components,
        fail_on_missing_masks=args.fail_on_missing_masks,
    )
    print(f"Processed images: {summary['num_images_processed']} / {summary['num_images_total']}")
    print(f"Skipped images: {summary['num_images_skipped']}")
    print(f"Extracted objects: {summary['num_objects_total']}")
    print(f"Summary file: {Path(args.output_dir) / 'summary.json'}")
    print(f"All results file: {Path(args.output_dir) / 'all_results.json'}")
    print(f"Per-image results dir: {Path(args.output_dir) / 'per_image'}")


def _collect_files(root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in root.glob(pattern) if path.is_file())


def _write_json(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _mask_matches_image_stem(image_stem: str, mask_stem: str) -> bool:
    tokens = re.split(r"[_\-.]+", mask_stem)
    return (
        mask_stem == image_stem
        or mask_stem.startswith(f"{image_stem}_")
        or mask_stem.startswith(f"{image_stem}-")
        or image_stem in tokens
    )


def _extract_object_suffix(image_stem: str, mask_stem: str) -> str:
    if mask_stem == image_stem:
        return ""
    if mask_stem.startswith(f"{image_stem}_") or mask_stem.startswith(f"{image_stem}-"):
        return mask_stem[len(image_stem) + 1 :]

    tokens = re.split(r"[_\-.]+", mask_stem)
    if image_stem in tokens:
        image_index = tokens.index(image_stem)
        return "_".join(tokens[image_index + 1 :])
    return ""


def _clean_object_token(value: str) -> str:
    cleaned = re.sub(r"[_\-.]+", "_", value.strip()).strip("_")
    if cleaned.lower() in GENERIC_OBJECT_TOKENS:
        return ""
    return cleaned


def _find_csv_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    raise ValueError(
        f"CSV must contain one of the following columns: {', '.join(candidates)}. "
        f"Received columns: {', '.join(fieldnames)}."
    )


def _require_csv_value(
    row: dict[str, str],
    column_name: str,
    csv_path: Path,
    row_index: int,
) -> str:
    value = row.get(column_name, "").strip()
    if not value:
        raise ValueError(
            f"CSV row {row_index} in `{csv_path}` is missing required value for `{column_name}`."
        )
    return value


def _resolve_dataset_path(path_value: str, root_dir: PathLike) -> Path:
    candidate = Path(path_value)
    return candidate if candidate.is_absolute() else Path(root_dir) / candidate


def _normalize_path_key(path: PathLike) -> str:
    return str(Path(path).resolve())


if __name__ == "__main__":
    main()
