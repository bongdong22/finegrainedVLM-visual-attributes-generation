# Visual Attributes Generation

입력으로 이미지와 segmentation mask를 받아, 객체의 시각 속성인 `location`, `size`, `shape`, `orientation`, `boundary`를 구조화된 JSON으로 만드는 코드입니다.

## 사용 방법

### 1. 데이터 준비

아래처럼 이미지 폴더와 마스크 폴더를 준비합니다.

```text
/path/to/dataset/
├── images/
│   ├── case001.png
│   ├── case002.png
│   └── ...
└── masks/
    ├── case001.png
    ├── case002.png
    └── ...
```

- 이미지와 마스크는 공간 크기가 같아야 합니다.
- 마스크는 binary mask여야 합니다.
- 객체가 하나인 데이터셋이면 마스크 파일 1개가 객체 1개를 의미합니다.
- 객체가 여러 개인 경우:
  - 마스크 파일을 객체별로 여러 개 둘 수 있습니다.
  - 또는 하나의 마스크 안에 disconnected object 여러 개가 있어도 됩니다. 이 경우 `--split-components`를 사용합니다.

### 2. 경로와 객체 이름 준비

아래 두 값만 자기 데이터에 맞게 바꾸면 됩니다.

- `--images-dir`: 이미지 폴더 경로
- `--masks-dir`: 마스크 폴더 경로
- `--output-dir`: 결과 저장 폴더 경로
- `--object-name`: 객체 이름

예를 들어 객체가 전부 종양이면 `Tumor`, 낭종이면 `Cyst`처럼 직접 적습니다.

### 3. 실행

프로젝트 루트에서 아래 명령을 실행합니다.

```bash
cd /home/bongdong2/bongdong2_workspace/fine-grained-vlm

PYTHONPATH=. python -m visual_attributes_generation \
  --images-dir /path/to/dataset/images \
  --masks-dir /path/to/dataset/masks \
  --output-dir /path/to/dataset/output \
  --object-name Tumor
```

하나의 mask 안에 떨어진 객체가 여러 개 있으면 아래처럼 실행합니다.

```bash
cd /home/bongdong2/bongdong2_workspace/fine-grained-vlm

PYTHONPATH=. python -m visual_attributes_generation \
  --images-dir /path/to/dataset/images \
  --masks-dir /path/to/dataset/masks \
  --output-dir /path/to/dataset/output \
  --object-name Tumor \
  --split-components
```

여러 객체를 따로 지정해야 하면 `image_filename,mask_filename,object_name` 형식의 CSV를 만들고 `--object-map-csv`를 사용하면 됩니다.

```csv
image_filename,mask_filename,object_name
case001.png,case001_tumor.png,Tumor
case001.png,case001_cyst.png,Cyst
case002.png,case002_mass.png,Mass
```

```bash
PYTHONPATH=. python -m visual_attributes_generation \
  --images-dir /path/to/dataset/images \
  --masks-dir /path/to/dataset/masks \
  --output-dir /path/to/dataset/output \
  --object-map-csv /path/to/dataset/object_map.csv
```

## 결과 예시

기본 결과는 이미지별 JSON과 전체 결과 JSON으로 저장됩니다. 객체 하나의 핵심 결과는 아래처럼 보입니다.

```json
{
  "object": "Tumor",
  "attributes": {
    "location": "upper-center",
    "size": "small",
    "shape": "oval",
    "orientation": "horizontal",
    "boundary": "mildly-irregular"
  }
}
```

UDIAT처럼 추가 메타데이터를 붙이면 아래처럼 `disease_type`도 함께 넣을 수 있습니다.

```json
{
  "object": "Tumor",
  "attributes": {
    "location": "upper-center",
    "size": "small",
    "shape": "oval",
    "orientation": "horizontal",
    "boundary": "mildly-irregular",
    "disease_type": "benign"
  },
  "image_path": "Benign/000001.png",
  "mask_path": "Benign_mask/000001.png"
}
```

## 어떤 코드가 어떤 작업을 하나

### 1. 위치, 모양 기하 정보 계산: `geometry.py`

- mask 정규화
- centroid 계산
- area ratio 계산
- bounding box aspect ratio 계산
- circularity 계산
- solidity 계산
- orientation angle 계산
- connected component 분리

### 2. 크기 구간 학습: `size.py`

- 데이터셋 전체 mask를 보고 size threshold를 학습
- `small`, `medium`, `large` 기준을 quantile 기반으로 계산

### 3. 속성 라벨 생성: `extractor.py`

- 측정값을 실제 속성 라벨로 변환
- `location`, `size`, `shape`, `orientation`, `boundary` 생성
- 객체별 결과 dict 생성

### 4. 폴더 단위 실행: `batch.py`

- 이미지 폴더와 마스크 폴더를 읽어 전체 실행
- `summary.json`, `all_results.json`, `per_image/*.json` 저장
- `--object-name`, `--object-map-csv`, `--split-components` 처리

### 5. 실행 진입점: `__main__.py`

- `python -m visual_attributes_generation` 실행 시 `batch.py`를 호출

### 6. 설정값 정의: `types.py`

- `AttributeRules`
- `ObjectMaskInput`
- `ObjectMeasurements`
- 크기 threshold 상태 정의

### 7. 메타데이터 속성 추가: `dataset/build_udiat_metadata_dataset.py`

- 기존 visual attribute 결과에 메타데이터를 추가
- 예: `disease_type = benign | malignant`
- 최종적으로 `all_results.json` 형태의 데이터셋 생성

## 작동 방식

### 1. 위치: `geometry.py` + `extractor.py`

- mask foreground의 중심점 `centroid`를 계산합니다.
- 중심 좌표를 이미지 전체 크기로 나눠 정규화합니다.
- 이미지를 3x3 grid로 나눠 `upper-left`, `center`, `lower-right` 같은 위치 라벨을 만듭니다.

### 2. 크기: `size.py` + `extractor.py`

- 각 객체의 면적 비율 `mask_area / image_area`를 계산합니다.
- 학습 데이터셋 전체에서 quantile threshold를 fit합니다.
- 기본은 tertile 기준이라 `small`, `medium`, `large`로 나눕니다.

### 3. 모양: `geometry.py` + `extractor.py`

- bounding box aspect ratio와 circularity를 계산합니다.
- 이 값을 기준으로 `round`, `oval`, `elongated`, `irregular`를 분류합니다.

### 4. 방향: `geometry.py` + `extractor.py`

- foreground 좌표에 PCA를 적용해 주축 방향 각도를 구합니다.
- angle을 이용해 `horizontal`, `vertical`, `oblique`로 변환합니다.

### 5. 경계: `geometry.py` + `extractor.py`

- solidity와 circularity를 계산합니다.
- 경계가 얼마나 매끈한지 기준으로 `smooth`, `mildly-irregular`, `irregular`를 분류합니다.

### 6. 다중 객체 처리: `extractor.py` + `batch.py`

- 마스크 파일이 여러 개면 객체별로 각각 처리합니다.
- 하나의 mask 안에 객체가 여러 개면 `--split-components`로 분리해서 처리합니다.
- 객체 이름이 여러 개면 `--object-map-csv`로 mask와 object를 연결합니다.

### 7. 메타데이터 결합: `dataset/build_udiat_metadata_dataset.py`

- visual attribute 결과를 읽습니다.
- 데이터셋 폴더 구조나 라벨 정보를 이용해 메타 속성을 붙입니다.
- 예를 들어 UDIAT에서는 `Benign`, `Malignant` 폴더를 보고 `disease_type`을 추가합니다.

## 한 줄 정리

`visual_attributes_generation`은 이미지와 segmentation mask로부터 객체의 위치, 크기, 모양, 방향, 경계 같은 시각 속성을 자동 추출하고, 필요하면 `disease_type` 같은 메타 속성까지 결합해 JSON 데이터셋으로 만드는 코드입니다.
