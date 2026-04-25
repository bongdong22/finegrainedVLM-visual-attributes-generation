# Visual Attributes Module

**[입력: 이미지, Segmentation Mask ➡ 출력: 위치, 크기, 모양, 방향, 경계 등의 시각 속성]**  
이 모듈은 이미지와 마스크를 입력받아 객체의 수치적 상태를 분석한 뒤, 후속 처리에 쓰기 쉽게 일관된 텍스트 속성(예: small, upper-right 등)으로 자동 분류 및 추출하는 도구입니다.

---

## 🚀 간단 사용 방법 (Quick Start)

**1. 지정된 형태의 데이터 구비**  
원본 이미지와 마스크 파일을 아래와 같은 폴더 구조로 분리하여 준비합니다.
```text
/data/my_dataset/
├── images/
│   ├── case001.png
│   └── case002.png
└── masks/
    ├── case001_tumor.png
    └── case002_tumor.png
```

**2. 실행 스크립트에 파라미터 지정**  
명령어 스크립트에 다음 순서대로 매개변수를 지정합니다.
(1) 입력 이미지 폴더  (2) 임력 마스크 폴더  (3) 결과물 저장 대상 폴더  (4) 객체 이름(예: `tumor`)

**3. 스크립트 실행**  
터미널에서 아래와 같이 스크립트를 실행하면 데이터를 분석하여 모든 속성을 일괄 추출합니다.
```bash
bash scripts/run_visual_attributes.sh \
  /data/my_dataset/images \
  /data/my_dataset/masks \
  /data/my_dataset/outputs \
  tumor
```

---

## 📂 각 시각 속성 추출 코드

- **1. 위치**: `extractor.py`
- **2. 크기**: `size.py`
- **3. 모양**: `extractor.py`
- **4. 방향**: `geometry.py`, `extractor.py`
- **5. 경계**: `extractor.py`

*(모든 객체의 기본 기하학 수치 측정과 뼈대 연산은 `geometry.py`에서 우선 진행됩니다)*

---

## ⚙️ 작동 방식

**1. 위치 (`extractor.py`)**
`geometry.py`에서 연산해 넘겨준 객체의 픽셀 중심점(`centroid`) 좌표를 0~1 사이로 정규화합니다. 이 좌표를 기준으로 이미지를 3x3 격자로 분할하여 좌표가 위치한 칸에 맞춰 `upper-left`, `center`, `lower-right` 등으로 위치를 문자열 라벨링합니다.

**2. 크기 (`size.py`)**
전체 데이터셋마다 크기 비율이 다르기 때문에 하나의 고정값으로 크기를 판단하지 않습니다. 먼저 전체 마스크를 읽어 이미지 전체 면적 대비 객체 면적인 `area_ratio` 값의 분포를 확인합니다. 분포값을 기준으로 분위수(Quantile)를 계산하여 객체를 `small`, `medium`, `large` 3단계로 상대 평가하여 분류합니다.

**3. 모양 (`extractor.py`)**
객체를 감싸는 Bounding Box의 장축/단축 비율인 `aspect_ratio`와 객체가 원의 형태와 얼마나 유사한지를 나타내는 `circularity` (원형도) 측정치를 사용합니다. 이 수치들을 자체 임계값과 대조하여 객체를 `round`, `oval`, `elongated`, `irregular` 로 치환합니다.

**4. 방향 (`geometry.py` & `extractor.py`)**
우선 `geometry.py`에서 객체를 이루는 픽셀 좌표들에 PCA(주성분 분석)를 통해 주축(Principal Axis) 각도인 `orientation_angle`을 측정해 넘깁니다. `extractor.py`에서 해당 각도가 수평, 수직, 그 외 대각선 중 어디에 가장 가까운지 판단하여 `horizontal`, `vertical`, `oblique` 로 번역합니다.

**5. 경계 (`extractor.py`)**
객체의 테두리가 얼마나 울퉁불퉁하고 불규칙한지를 파악하기 위해 면적의 채워진 정도를 뜻하는 `solidity`(Convex hull 면적 대비 실제 마스크 면적 비율)와 위 구한 `circularity`를 가져와 판별합니다. 수치 기준선에 따라 테두리를 `smooth`, `mildly-irregular`, `irregular` 등급으로 분류합니다.
