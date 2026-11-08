# Side-Scan Sonar Mine Detection Benchmark

공개 측면주사 소나(Side-Scan Sonar, SSS) 영상에서 기뢰유사물체(mine-like object) 탐지기와 패치 분류기를 촬영 연도 분할 조건으로 평가한 프로젝트입니다.

## Overview

- **문제**: 공개 AUV 측면주사 영상 자료에는 같은 장면의 축소 복사본이 많고(1,170장 중 417쌍), 연도별 물체 구성이 크게 달라 이미지 단위 무작위 분할에서는 성능이 과대평가될 수 있습니다.
- **방법**: 중복 장면을 점검한 뒤 공식 연도 분할(학습 2015 · 2017 · 2021 → 시험 2010 · 2018)로 평가했습니다. 탐지는 고전 정합필터와 YOLOv8n, 분류는 하이라이트–그림자 물리 특징과 DINOv2를 비교했습니다.
- **결과**: 처음 보는 연도에서 YOLOv8n의 객체 재현율은 78.1%(화면당 오경보 4건), 정합필터는 4.3%(1건)였습니다. 연도가 바뀔 때 분류 성능 하락은 물리 특징(0.04)이 DINOv2(0.15)보다 작았습니다.
- **연계**: YOLO 후보를 물리 특징 분류기로 다시 판정하면 오경보가 38% 줄고, 맞게 찾은 물체는 1.7%만 잃었습니다.

<p align="center"><img src="docs/figures/highlight_shadow.png" width="95%" alt="Side-scan frame, across-track highlight-shadow profiles and oriented patches"></p>

## Results

| 탐지 (시험 연도 2010 · 2018) | 재현율 @ 화면당 오경보 1건 | 재현율 @ 4건 |
|---|---|---|
| YOLOv8n | 59.6% | 78.1% |
| 고전 정합필터 | 4.3% | — |

<p align="center"><img src="docs/figures/detection_froc.png" width="95%" alt="FROC curves for YOLOv8n and the classical matched filter on test years"></p>

| 패치 3분류 macro F1 | 이미지 분할 | 장면 분할 | 연도 분할 |
|---|---|---|---|
| 물리 특징 + 로지스틱 회귀 | 0.734 | 0.719 | 0.679 |
| DINOv2 고정 특징 + 선형 분류기 | 0.814 | 0.751 | 0.606 |
| DINOv2 미세조정 | — | — | 0.645 |

<details>
<summary>분할별 분류 결과 · 혼동 행렬</summary>

<p align="center"><img src="docs/figures/classification_splits.png" width="95%" alt="Three-class macro F1 by evaluation split and year-split confusion matrices"></p>
<p align="center"><sub>분할별 3분류 macro F1과 연도 분할 혼동 행렬 (정답 위치 패치 조건)</sub></p>

</details>

- 하이라이트–그림자 대비는 물체와 해저 배경을 구분했지만(AUROC 0.830), 물체 종류(MILCO vs NOMBO) 구분력은 낮았습니다(0.513).
- 결과 수치: [`src/b3_yolo.json`](src/b3_yolo.json), [`src/b2_results.json`](src/b2_results.json), [`src/b3_patch.json`](src/b3_patch.json) · 결과 리포트: [`docs/milco-report.html`](docs/milco-report.html)

## Cascade (탐지 → 분류 연계)

YOLOv8n 후보마다 하이라이트–그림자 물리 특징으로 MILCO · NOMBO · 해저 배경을 다시 판정하고, 배경으로 판정된 후보를 버립니다. 임계값은 YOLO 학습에 쓰지 않은 학습 연도 장면(검증 26장, 물체 47개)에서 정하고 시험 연도에 한 번 적용했습니다.

| 시험 연도 (578장, 물체 114개) | 재현율 | 화면당 오경보 |
|---|---|---|
| YOLOv8n 단독 | 52.6% | 0.66 |
| + 물리 특징 재판정 | 51.8% | 0.41 |
| + 재판정, YOLO 오경보를 배경으로 추가 학습 | 39.5% | 0.34 |

- 재판정으로 시험 연도 오경보의 38%가 제거되었고, 맞게 찾은 물체 중 1.7%를 잃었습니다.
- YOLO 오경보를 배경 학습에 추가한 방식은 오경보를 48% 줄였지만 맞게 찾은 물체의 25%를 잃어 채택하지 않았습니다.
- 검증 장면에서 정한 탐지 임계값은 검증에서 재현율 83%(오경보 3.9건)였지만 시험 연도에서는 53%(0.66건)였습니다. 연도가 바뀌면 운영점 자체가 이동합니다.
- 맞게 찾은 물체 60개의 종류 구분(MILCO vs NOMBO) 균형정확도는 YOLO 0.62, 물리 특징 분류기 0.68이었습니다.
- 결과 수치: [`src/b4_cascade.json`](src/b4_cascade.json)

## Dataset

- Santos et al. 2024, *Data in Brief* (CC BY 4.0) — Gavia AUV · Marine Sonic 900/1800 kHz, JPEG 1,170장(고유 장면 753개), MILCO · NOMBO 라벨
- 원자료는 저장소에 포함하지 않으며 `src/download_data.sh`로 받습니다.

<details>
<summary>자료 점검 (중복 장면 · 연도별 구성)</summary>

<p align="center"><img src="docs/figures/data_diagnosis.png" width="95%" alt="Objects per year, duplicated scene exports and box sizes"></p>
<p align="center"><sub>연도별 고유 물체 수, 축소 복사본 예시, 표적 크기</sub></p>

</details>

## Getting Started

```bash
pip install -r requirements.txt
cd src
./download_data.sh      # 공개 자료 내려받기
python screen.py        # 중복 장면 점검
python b2.py            # 전처리 · 정합필터 탐지
python prep_tiles.py && python train_yolo.py && python b3_yolo_eval.py   # YOLOv8n 학습 · 평가
python b3_patch.py 2    # 패치 3분류
python b4_cascade.py    # 탐지 → 분류 연계
```

결과 JSON과 학습된 탐지기 가중치가 저장소에 있어 원자료 없이도 수치를 확인할 수 있습니다.

## Project Structure

| 폴더 | 내용 |
|---|---|
| `src` | 점검 · 전처리 · 탐지 · 분류 스크립트, 결과 JSON, YOLO 가중치 |
| `docs` | 그림, 결과 리포트(HTML) |

## Limitations

- CPU로 학습한 소형 모델(YOLOv8n, 25 epoch)이며, 입력은 원시 신호가 아닌 JPEG 영상입니다.
- 패치 분류 표는 정답 위치 패치 조건의 결과입니다. 연계 평가의 임계값은 검증 장면 26장으로 정해 표본이 작습니다.
- 한 자료(한 장비 · 해역 계열)의 결과입니다.

## License

코드는 MIT License를 따릅니다. 자료와 자료에서 파생된 그림은 Santos et al. 2024의 CC BY 4.0 조건을 따릅니다.
