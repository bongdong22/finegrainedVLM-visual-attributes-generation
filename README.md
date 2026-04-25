# Visual Attributes Module

`fgvlm.visual_attributes`는 이미지와 segmentation mask를 입력으로 받아, 객체의 시각적 속성을 구조화된 proposition 형태로 추출하는 모듈입니다.

이 모듈은 특정 데이터셋 전용 코드가 아니라, 다양한 medical image dataset에 재사용할 수 있도록 만들었습니다. 처음 설계할 때는 `data_UDIAT`를 예시로 생각했지만, 실제 구현은 더 일반적인 상황을 처리하도록 구성되어 있습니다.

지원하는 대표적인 입력 형태는 다음과 같습니다.

- 이미지 1장 + 마스크 1개 + 객체 1개
- 이미지 1장 + 마스크 파일 여러 개 + 객체 여러 개
- 이미지 1장 + 마스크 1개 안에 disconnected object 여러 개

핵심 목표는 "mask만으로 계산 가능한 속성"을 추출해 아래처럼 정리된 결과를 만드는 것입니다.

```python
{
    "object": "lesion",
    "instance_index": 0,
    "image_path": "/path/to/image.png",
    "mask_path": "/path/to/mask.png",
    "measurements": {
        "centroid": [412.3, 108.5],
        "area_ratio": 0.031,
        "aspect_ratio": 1.48,
        "circularity": 0.72,
        "solidity": 0.91,
        "orientation_angle": 84.7,
    },
    "propositions": [
        ["lesion", "location", "upper-right"],
        ["lesion", "size", "small"],
        ["lesion", "shape", "oval"],
        ["lesion", "orientation", "vertical"],
        ["lesion", "boundary", "mildly-irregular"],
    ],
}
```

## 1. 이 모듈이 하는 일

이 모듈은 크게 두 단계로 작동합니다.

1. mask에서 수치 측정값을 계산합니다.
2. 그 측정값을 사람이 읽기 쉬운 label로 바꿉니다.

계산하는 주요 측정값은 다음과 같습니다.

- `centroid`: 객체 중심 좌표
- `area_ratio`: 객체 면적 / 이미지 전체 면적
- `aspect_ratio`: 객체 bounding box의 긴 축과 짧은 축 비율
- `circularity`: 얼마나 원형에 가까운지 나타내는 값
- `solidity`: convex hull 대비 얼마나 꽉 찬 모양인지 나타내는 값
- `orientation_angle`: 객체의 주축 방향 각도

이 측정값을 기반으로 다음 속성을 결정합니다.

- `location`
- `size`
- `shape`
- `orientation`
- `boundary`

즉, 이 모듈은 "이미지 설명 생성"이 아니라 "객체 속성 정리"에 집중합니다.

## 2. 폴더 안 파일별 설명

이 디렉토리에는 다음 파일들이 있습니다.

- `__init__.py`
  - 외부에서 사용할 공개 API를 모아둡니다.
  - 보통 사용자는 여기에서 필요한 함수와 클래스를 import하면 됩니다.
- `types.py`
  - 설정값과 데이터 구조를 정의합니다.
  - `AttributeRules`, `ObjectMaskInput`, `ObjectMeasurements` 같은 dataclass가 들어 있습니다.
- `geometry.py`
  - mask를 정규화하고, 실제 기하학적 측정값을 계산합니다.
  - centroid, area ratio, aspect ratio, circularity, solidity, orientation angle 계산이 여기 있습니다.
  - connected component 분리도 여기서 처리합니다.
- `size.py`
  - dataset-level size threshold를 학습하는 코드입니다.
  - 여러 마스크를 보고 `small / medium / large`의 경계를 quantile 기반으로 맞춥니다.
- `extractor.py`
  - 가장 핵심적인 실행 파일입니다.
  - geometry에서 계산한 값과 size calibrator를 이용해 proposition을 만듭니다.
  - 단일 객체, 다중 객체, 파일 경로 기반 입력을 모두 처리합니다.
- `README.md`
  - 지금 보고 있는 설명 문서입니다.

## 3. 내부 동작 방식

이 모듈은 아래 순서로 동작합니다.

### 3-1. 입력 정규화

이미지는 grayscale 2D 또는 RGB-like 3D 배열을 받을 수 있습니다.

mask는 다음 형태를 모두 허용합니다.

- `0/1`
- `0/255`
- boolean mask
- 3채널 mask

mask가 3채널이면 `any(axis=-1)`로 2D binary mask로 변환합니다. 이후에는 내부적으로 `foreground > 0` 기준의 boolean mask로 처리합니다.

또한 image와 mask의 공간 크기가 다르면 바로 `ValueError`를 발생시켜 잘못된 입력을 초기에 잡습니다.

### 3-2. 객체 측정값 계산

`geometry.py`에서 다음을 계산합니다.

- 중심 좌표는 foreground pixel들의 평균 좌표로 계산합니다.
- 면적 비율은 `mask_area / image_area`로 계산합니다.
- aspect ratio는 bounding box의 가로/세로 중 큰 쪽을 작은 쪽으로 나눈 값입니다.
- circularity는 `4 * pi * area / perimeter^2`입니다.
- solidity는 `area / convex_hull_area`입니다.
- orientation은 foreground 좌표에 PCA를 적용해 주축 방향을 구합니다.

### 3-3. label로 변환

`extractor.py`에서 측정값을 label로 바꿉니다.

- `location`
  - centroid를 이미지 전체 좌표계에서 정규화한 뒤 3x3 grid로 나눕니다.
  - 예: `upper-left`, `center`, `lower-right`
- `size`
  - 고정 threshold를 쓰지 않고, 학습 데이터셋에서 fit한 quantile threshold를 사용합니다.
  - 기본은 tertile이라 `small`, `medium`, `large`로 나눕니다.
- `shape`
  - aspect ratio와 circularity를 이용해 `round`, `oval`, `elongated`, `irregular`로 분류합니다.
- `orientation`
  - angle을 이용해 `horizontal`, `vertical`, `oblique`로 나눕니다.
- `boundary`
  - solidity와 circularity를 이용해 `smooth`, `mildly-irregular`, `irregular`로 분류합니다.

### 3-4. proposition 생성

마지막으로 아래 형식의 proposition 리스트를 만듭니다.

```python
[
    ["lesion", "location", "upper-right"],
    ["lesion", "size", "small"],
    ["lesion", "shape", "oval"],
    ["lesion", "orientation", "vertical"],
    ["lesion", "boundary", "irregular"],
]
```

여기서 `"lesion"`은 기본값이며, 다른 객체 이름을 넣고 싶으면 `object_name="cyst"`처럼 바꿔서 쓸 수 있습니다.

## 4. 여러 객체를 처리하는 방식

이 모듈은 여러 객체를 두 가지 방식으로 처리합니다.

### 방식 A. 마스크 파일이 여러 개인 경우

예를 들어 한 이미지에 대해 아래처럼 mask 파일이 여러 개 있을 수 있습니다.

- `image_001_lesion.png`
- `image_001_cyst.png`
- `image_001_calcification.png`

이 경우 각 mask를 각각 하나의 객체로 보고 처리합니다.

### 방식 B. 마스크 파일은 하나인데 disconnected region이 여러 개인 경우

하나의 binary mask 안에 서로 떨어진 영역이 여러 개 있으면 `split_components=True`로 connected component를 분리해 각각 별도의 객체 인스턴스로 처리할 수 있습니다.

즉, 한 파일 안에 객체가 여러 개 들어 있는 경우도 대응됩니다.

## 5. 주요 API 설명

보통은 아래 API들만 알면 사용 가능합니다.

### `AttributeRules`

shape, orientation, boundary 분류 threshold를 설정하는 dataclass입니다.

기본 규칙을 그대로 써도 되고, 필요하면 일부 값만 바꿀 수 있습니다.

```python
from fgvlm.visual_attributes import AttributeRules

rules = AttributeRules(
    round_aspect_ratio_max=1.3,
    round_circularity_min=0.75,
)
```

### `SizeQuantileCalibrator`

size는 고정 임계값으로 나누지 않고, 데이터셋 전체를 보고 경계를 정합니다.

사용 순서는 다음과 같습니다.

1. 학습용 mask들을 모읍니다.
2. `fit(...)`으로 threshold를 학습합니다.
3. 이후 개별 객체 추출에서 이 calibrator를 사용합니다.

```python
from pathlib import Path
from fgvlm.visual_attributes import SizeQuantileCalibrator

mask_paths = list(Path("/path/to/train_masks").glob("*.png"))
size_calibrator = SizeQuantileCalibrator().fit(mask_paths)
```

기본은 33%, 66% quantile을 사용합니다.

### `extract_object_attributes`

단일 객체용 함수입니다.

```python
result = extract_object_attributes(
    image=image_array,
    mask=mask_array,
    object_name="lesion",
    size_calibrator=size_calibrator,
)
```

### `extract_image_attributes`

이미지 1장 안에 객체가 여러 개 있을 때 쓰는 함수입니다.

`ObjectMaskInput` 리스트를 받아 객체별 결과 리스트를 반환합니다.

```python
from fgvlm.visual_attributes import ObjectMaskInput, extract_image_attributes

object_masks = [
    ObjectMaskInput(mask=mask1, object_name="lesion", mask_path="mask1.png"),
    ObjectMaskInput(mask=mask2, object_name="cyst", mask_path="mask2.png"),
]

results = extract_image_attributes(
    image=image_array,
    object_masks=object_masks,
    size_calibrator=size_calibrator,
)
```

### `extract_image_attributes_from_paths`

파일 경로 기반으로 바로 처리하고 싶을 때 가장 편한 함수입니다.

```python
from fgvlm.visual_attributes import extract_image_attributes_from_paths

results = extract_image_attributes_from_paths(
    image_path="/path/to/image.png",
    mask_inputs=[
        ("/path/to/mask_lesion.png", "lesion"),
        ("/path/to/mask_cyst.png", "cyst"),
    ],
    size_calibrator=size_calibrator,
)
```

### `discover_mask_inputs`

이미지 파일명과 비슷한 이름의 mask를 디렉토리에서 자동으로 찾는 helper입니다.

다만 실제 프로젝트에서는 mask 대응 관계가 명확하다면, 이 helper보다 `mask_path + object_name`을 직접 넘기는 방식이 더 안전합니다.

```python
from fgvlm.visual_attributes import discover_mask_inputs

mask_inputs = discover_mask_inputs(
    image_path="/path/to/image_001.png",
    mask_dir="/path/to/masks",
    object_name="lesion",
)
```

## 6. 처음 사용하는 사람이 따라가기 좋은 기본 사용 순서

가장 추천하는 사용 흐름은 다음과 같습니다.

### Step 1. size calibrator 준비

먼저 training set의 mask들로 size 기준을 맞춥니다.

```python
from pathlib import Path
from fgvlm.visual_attributes import SizeQuantileCalibrator

train_mask_paths = list(Path("/path/to/train_masks").glob("*.png"))
size_calibrator = SizeQuantileCalibrator().fit(train_mask_paths)
```

### Step 2. 한 이미지의 객체 속성 추출

마스크가 여러 파일로 나뉘어 있다면 다음처럼 바로 처리하면 됩니다.

```python
from fgvlm.visual_attributes import extract_image_attributes_from_paths

results = extract_image_attributes_from_paths(
    image_path="/path/to/image.png",
    mask_inputs=[
        ("/path/to/image_lesion.png", "lesion"),
        ("/path/to/image_cyst.png", "cyst"),
    ],
    size_calibrator=size_calibrator,
)
```

### Step 3. 마스크 하나 안에 객체가 여러 개라면 component 분리

```python
results = extract_image_attributes_from_paths(
    image_path="/path/to/image.png",
    mask_inputs=[("/path/to/image_multi_mask.png", "lesion")],
    size_calibrator=size_calibrator,
    split_components=True,
)
```

## 7. 반환값을 읽는 법

반환값은 객체 단위 dict 또는 dict의 리스트입니다.

중요한 필드는 다음과 같습니다.

- `object`
  - 사용자가 넣은 객체 이름
- `instance_index`
  - 한 이미지 내에서 객체 순서를 나타내는 번호
- `image_path`
  - 경로 기반 API를 사용한 경우 포함됨
- `mask_path`
  - 해당 객체를 만든 마스크 파일 경로
- `measurements`
  - 실제 계산된 수치 정보
- `propositions`
  - 사람이 읽기 쉬운 속성 표현

`measurements`는 분석과 디버깅에 유용하고, `propositions`는 후속 captioning이나 structured dataset 생성에 바로 사용할 수 있습니다.

## 8. 예외 처리와 주의사항

다음 경우에는 에러가 발생할 수 있습니다.

- mask가 비어 있는 경우
  - foreground pixel이 하나도 없으면 `ValueError`
- image와 mask 크기가 다른 경우
  - spatial size가 다르면 `ValueError`
- size calibrator를 fit하지 않고 사용한 경우
  - `classify(...)` 전에 `fit(...)`이 필요함
- size calibrator에 너무 적은 mask를 넣은 경우
  - 기본 구현은 최소 3개의 non-empty object mask가 필요함

또한 shape, orientation, boundary는 모두 rule-based 분류이므로, 데이터 특성에 따라 `AttributeRules`를 조정하면 더 잘 맞을 수 있습니다.

## 9. scikit-image와 fallback 동작

구현은 `scikit-image`를 사용할 수 있으면 우선 활용합니다. 하지만 환경에 따라 `scikit-image`가 import되지 않는 경우가 있어, 아래 계산은 `cv2`와 `NumPy` fallback으로도 동작하게 만들었습니다.

- contour 추출
- perimeter 계산
- convex hull 계산
- connected component 분리
- PCA 기반 orientation 계산

즉, `scikit-image`가 불안정한 환경에서도 핵심 기능은 유지되도록 설계했습니다.

## 10. 테스트 방법

이 모듈의 테스트는 아래 파일에 있습니다.

- `fine-grained-vlm/tests/test_visual_attributes.py`

실행 방법:

```bash
cd /home/bongdong2/bongdong2_workspace/fine-grained-vlm
PYTHONPATH=src pytest tests/test_visual_attributes.py
```

이 테스트는 아래를 확인합니다.

- empty mask 처리
- image/mask size mismatch 처리
- centroid의 3x3 location mapping
- size tertile fitting
- shape 분류
- orientation 분류
- boundary 분류
- 여러 mask 파일 처리
- disconnected component 분리

## 11. 어떤 상황에서 이 모듈이 특히 유용한가

이 모듈은 아래 같은 작업에 잘 맞습니다.

- segmentation 결과를 정형 속성 데이터로 바꾸고 싶을 때
- 이미지 캡션 생성 전에 객체 속성만 먼저 정리하고 싶을 때
- 데이터셋마다 다른 mask 구조를 하나의 공통 인터페이스로 처리하고 싶을 때
- lesion, cyst, mass, nodule 같은 객체를 같은 방식으로 속성화하고 싶을 때

## 12. 한 줄 요약

이 모듈은 "이미지 + mask"에서 객체의 위치, 크기, 모양, 방향, 경계 특성을 자동으로 계산하고, 이를 후속 파이프라인에서 바로 쓸 수 있는 구조화된 proposition 형태로 정리해 주는 범용 도구입니다.
