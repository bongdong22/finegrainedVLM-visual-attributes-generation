# finegrainedVLM-visual-attributes-generation

이미지와 segmentation mask를 입력으로 받아, 객체의 시각적 속성을 구조화된 JSON 형태로 추출하는 모듈입니다.

캡션이나 언어 모델 없이 **mask에서 직접 계산 가능한 속성만** 추출합니다.

지원하는 입력 형태:

- 이미지 1장 + 마스크 1개 (객체 1개)
- 이미지 1장 + 마스크 파일 여러 개 (객체 여러 개)
- 이미지 1장 + 마스크 1개 안에 disconnected region 여러 개

추출되는 속성: `location`, `size`, `shape`, `orientation`, `boundary`

---

## 1. 실행 방법

### 기본 실행 (단일 객체)

모든 mask가 동일한 객체 유형일 때 사용합니다.

```bash
python -m finegrainedVLM-visual-attributes-generation \
  --images-dir /data/my_dataset/images \
  --masks-dir  /data/my_dataset/masks \
  --output-dir /data/my_dataset/output \
  --object-name tumor
```

### 여러 객체 (명시적 매핑 CSV 사용)

mask마다 객체 이름이 다를 때 사용합니다.

```bash
python -m finegrainedVLM-visual-attributes-generation \
  --images-dir    /data/my_dataset/images \
  --masks-dir     /data/my_dataset/masks \
  --output-dir    /data/my_dataset/output \
  --object-map-csv /data/my_dataset/object_map.csv
```

### disconnected region 분리

mask 1개 안에 떨어진 객체가 여러 개 있을 때 사용합니다.

```bash
python -m finegrainedVLM-visual-attributes-generation \
  --images-dir /data/my_dataset/images \
  --masks-dir  /data/my_dataset/masks \
  --output-dir /data/my_dataset/output \
  --object-name tumor \
  --split-components
```

### 쉘 스크립트로 실행

```bash
# 단일 객체
bash run_visual_attributes_split_components.sh \
  /data/my_dataset/images \
  /data/my_dataset/masks \
  /data/my_dataset/output \
  tumor

# object_map.csv를 사용하는 경우 (4번째 인자가 파일이면 자동으로 CSV 모드로 실행됨)
bash run_visual_attributes_split_components.sh \
  /data/my_dataset/images \
  /data/my_dataset/masks \
  /data/my_dataset/output \
  /data/my_dataset/object_map.csv
```

### 주요 CLI 옵션

| 옵션 | 설명 |
|---|---|
| `--images-dir` | 이미지 폴더 경로 (필수) |
| `--masks-dir` | 마스크 폴더 경로 (필수) |
| `--output-dir` | 결과 저장 폴더 (필수) |
| `--object-name` | 모든 마스크에 동일하게 적용할 객체 이름 |
| `--object-map-csv` | image-mask-object 매핑 CSV 파일 경로 |
| `--infer-object-name` | 마스크 파일명에서 객체 이름 자동 추론 (`case001_tumor.png` → `tumor`) |
| `--split-components` | mask 안의 disconnected region을 각각 별도 객체 인스턴스로 분리 |
| `--calibration-masks-dir` | size 기준 학습에 사용할 별도 마스크 폴더 (지정 안 하면 `--masks-dir` 사용) |
| `--image-glob` | 이미지 파일 패턴 (기본: `*.png`) |
| `--mask-glob` | 마스크 파일 패턴 (기본: `*.png`) |
| `--fail-on-missing-masks` | 매칭되는 마스크가 없는 이미지에서 즉시 오류 발생 |

---

## 2. 생성되는 파일 형식

실행이 완료되면 `--output-dir` 아래에 파일이 **하나** 생성됩니다.

```
output/
└── all_results.json
```

> `summary.json`이나 `per_image/*.json`은 이 코드가 생성하지 않습니다.
> 실행 요약(처리 이미지 수 등)은 `run_folder_extraction()`의 **반환값**으로만 전달되며,
> `python -m ...`으로 실행할 경우 터미널 stdout에 출력됩니다.

### `all_results.json`

객체 단위로 **flat하게** 나열된 리스트입니다. 이미지마다 객체가 1개면 이미지 수 = 리스트 길이, 여러 개면 더 길어집니다.

```json
[
  {
    "attributes": {
      "location": "upper-center",
      "size": "small",
      "shape": "oval",
      "orientation": "horizontal",
      "boundary": "mildly-irregular"
    },
    "image_path": "000001.png",
    "mask_path": "000001.png"
  },
  {
    "attributes": {
      "location": "center",
      "size": "large",
      "shape": "irregular",
      "orientation": "oblique",
      "boundary": "irregular"
    },
    "image_path": "000002.png",
    "mask_path": "000002.png"
  },
  ...
]
```

> - `image_path`: `--images-dir` 기준의 **상대 경로**
> - `mask_path`: `--masks-dir` 기준의 **상대 경로** (절대 경로로 변환 불가하면 절대 경로 그대로)
> - `measurements`(centroid, area_ratio 등 수치값)는 `all_results.json`에 **포함되지 않습니다**.
>   수치값은 Python API(`extract_image_attributes_from_paths`)를 직접 사용할 때만 접근 가능합니다.

---

## 3. 입력 데이터 구조

### 디렉토리 구조 예시

```
/data/my_dataset/
├── images/
│   ├── 000001.png
│   ├── 000002.png
│   └── ...
└── masks/
    ├── 000001.png      ← 이미지와 같은 stem (단일 객체 모드)
    ├── 000002.png
    └── ...
```

마스크와 이미지는 **파일명 stem이 같아야** 자동으로 매칭됩니다.
예: `000001.png` ↔ `000001.png`, 또는 `000001.png` ↔ `000001_tumor.png`

### `object_map.csv` 형식 (여러 객체)

mask마다 객체 이름이 다를 때 사용합니다.

```csv
image_filename,mask_filename,object_name
000001.png,000001_tumor.png,tumor
000001.png,000001_cyst.png,cyst
000002.png,000002_calcification.png,calcification
```

지원하는 컬럼 이름:

| 역할 | 허용하는 컬럼명 |
|---|---|
| 이미지 | `image_path`, `image_filename`, `image_name`, `image` |
| 마스크 | `mask_path`, `mask_filename`, `mask_name`, `mask` |
| 객체 이름 | `object_name`, `object`, `label`, `class_name` |

---

## 4. 추가 메타데이터 병합하기

생성된 `all_results.json`에 커스텀 메타데이터(예: 병명, 촬영 장비, fold 인덱스 등)를 결합하려면,
아래 형식으로 별도 CSV를 준비한 뒤 `image_path`를 키로 merge합니다.

### 4-1. 추가 메타데이터 CSV 형식

```csv
image_path,disease_type,scanner,fold
000001.png,benign,Siemens,0
000002.png,malignant,GE,1
```

- `image_path` 컬럼은 `all_results.json`의 `image_path` 값과 **정확히 일치**해야 합니다
  (`--images-dir` 기준 상대 경로, 파일명만인 경우가 많습니다).

### 4-2. 병합 스크립트

```python
import json
import pandas as pd

# all_results.json 로드
with open("output/all_results.json") as f:
    records = json.load(f)  # flat list of {attributes, image_path, mask_path}

# DataFrame으로 변환 (attributes를 개별 컬럼으로 펼침)
rows = []
for record in records:
    rows.append({
        "image_path": record["image_path"],
        "mask_path": record["mask_path"],
        **record["attributes"],
    })
df = pd.DataFrame(rows)

# 추가 메타데이터 CSV 로드 및 병합
meta_df = pd.read_csv("my_metadata.csv")
merged_df = df.merge(meta_df, on="image_path", how="left")

# 저장
merged_df.to_csv("output/attributes_with_meta.csv", index=False)
print(merged_df.columns.tolist())
print(merged_df.head())
```

결과 컬럼 예시:

```
image_path | mask_path | location | size | shape | orientation | boundary | disease_type | scanner | fold
```

---

## 5. 속성 계산 방식

### 측정값 계산 (`geometry.py`)

| 측정값 | 계산 방법 |
|---|---|
| `centroid` | foreground pixel들의 평균 좌표 |
| `area_ratio` | `mask 면적 / 이미지 전체 면적` |
| `aspect_ratio` | bounding box 긴 축 / 짧은 축 |
| `circularity` | `4π × area / perimeter²` |
| `solidity` | `area / convex_hull_area` |
| `orientation_angle` | foreground 좌표에 PCA 적용, 주축 방향 각도 |

### 속성 분류 기준

| 속성 | 가능한 값 | 분류 기준 |
|---|---|---|
| `location` | `upper-left` / `upper-center` / `upper-right` / `middle-left` / `center` / `middle-right` / `lower-left` / `lower-center` / `lower-right` | centroid를 3×3 grid로 나눔 |
| `size` | `small` / `medium` / `large` | training set 마스크의 `area_ratio` 33%/66% quantile 기준 |
| `shape` | `round` / `oval` / `elongated` / `irregular` | `aspect_ratio` + `circularity` 조합 |
| `orientation` | `horizontal` / `vertical` / `oblique` | `orientation_angle` (기본 임계: 20° / 70°) |
| `boundary` | `smooth` / `mildly-irregular` / `irregular` | `solidity` + `circularity` 조합 |

분류 임계값은 `AttributeRules` dataclass에서 조정할 수 있습니다.

---

## 6. 모듈 구조

```
finegrainedVLM-visual-attributes-generation/
├── __init__.py                    # 공개 API 모음
├── __main__.py                    # python -m 진입점 → batch.main()
├── batch.py                       # 폴더 단위 배치 추출 + CLI 파서
│                                  #   → all_results.json 저장
│                                  #   → 실행 요약을 dict로 반환 (파일 저장 X)
├── extractor.py                   # 단일/다중 객체 속성 추출 핵심 로직
├── geometry.py                    # mask → 수치 측정값 계산 (scikit-image / cv2 fallback)
├── size.py                        # SizeQuantileCalibrator (학습 데이터 기반 size threshold)
├── types.py                       # AttributeRules, ObjectMaskInput, ObjectMeasurements 등
└── run_visual_attributes_split_components.sh  ← 쉘 스크립트 (split-components 포함)
```

---

## 7. Python API 사용 예시

```python
from pathlib import Path
from finegrainedVLM_visual_attributes_generation import (
    SizeQuantileCalibrator,
    extract_image_attributes_from_paths,
    run_folder_extraction,
)

# size calibrator 학습
train_mask_paths = list(Path("/data/train/masks").glob("*.png"))
size_calibrator = SizeQuantileCalibrator().fit(train_mask_paths)

# 이미지 1장 처리 (measurements 포함한 전체 결과 접근 가능)
results = extract_image_attributes_from_paths(
    image_path="/data/images/000001.png",
    mask_inputs=[
        ("/data/masks/000001_tumor.png", "tumor"),
        ("/data/masks/000001_cyst.png", "cyst"),
    ],
    size_calibrator=size_calibrator,
)
# results: list of dicts with keys: object, instance_index, measurements, attributes, image_path, mask_path

# 폴더 전체 처리 → all_results.json 저장, 요약 dict 반환
summary = run_folder_extraction(
    images_dir="/data/images",
    masks_dir="/data/masks",
    output_dir="/data/output",
    object_name="tumor",
)
print(summary)
# {
#   "all_results_path": "/data/output/all_results.json",
#   "num_images_total": 163,
#   "num_images_processed": 163,
#   "num_images_skipped": 0,
#   "num_objects_total": 163,
#   "skipped_images": []
# }
```
