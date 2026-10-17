# Side-Scan Sonar Mine-Like Object Detection (측면주사 소나 기뢰유사물체 탐지)

공개 측면주사 소나(Side-Scan Sonar, SSS) 영상으로 **기뢰유사물체(mine-like object) 탐지 · 분류 AI 레퍼런스 모델의 성능을 비교**하고, 무인잠수정(UUV · XLUUV) 탑재를 전제로 한 **2단 탐지 · 분류기**를 설계한 코드입니다.

데이터 진단 → 영상 신호처리 → AI 탐지 · 분류 → 고찰의 순서로 구성했으며, 모든 결과는 공개 데이터와 이 저장소의 코드로 CPU 환경에서 재현할 수 있습니다.

## 한눈에 보기

| 항목 | 내용 |
|---|---|
| 데이터 | Santos et al. 2024, *Data in Brief* 53:110132 (CC BY 4.0). Gavia AUV, Marine Sonic 900/1800 kHz SSS. 영상 1,170장(고유 장면 753개), 클래스 MILCO(기뢰유사물체) · NOMBO(비기뢰 물체) |
| 신호처리 | 중복 장면 검출(고역통과 영상 상관), 색상표 제거 · 축척 통일, 수주(nadir) 검출, 횡방향 이득 정규화, 하이라이트–그림자(highlight–shadow) 특징, 고전적 정합필터 탐지기 |
| 탐지 | YOLOv8n, 320 px 타일 학습 → 1024 px 전체 화면 탐지 (CPU, 25 epoch) |
| 분류 | 음향 특징 + 로지스틱 회귀, DINOv2 고정 특징 + 선형 분류기, DINOv2 미세조정 (3분류: MILCO · NOMBO · 해저 배경) |
| 평가 | 공식 연도 분할(학습 2015 · 2017 · 2021 → 시험 2010 · 2018), 장면 · 이미지 분할 비교, 중심 일치 기준 FROC, mAP@0.5 |

## 핵심 결과

- **탐지**: JPEG로 내보낸 영상만 쓰는 입력 한계(원시 신호 · 이득 정보 없음)에도, 처음 보는 연도에서 **재현율 약 80%** (화면당 오경보 4건 허용, 78.1%). 오경보 1건/화면 기준 59.6%로 고전적 정합필터(4.3%)보다 크게 높음.
- **분류**: 이미지 단위 분할에서 DINOv2 F1 **0.81** → 문헌 보고치(0.81–0.82) 수준. 처음 보는 연도에서는 물리 특징(하이라이트–그림자 · 크기 · 수주 거리) 모델이 가장 견고 (F1 0.68, 연도 이동 시 하락 −4점 vs 범용 영상 특징 −15점).
- **데이터 진단**: 416 px 영상 417장은 모두 1024 px 원본의 축소 복사본 → 이미지 단위 무작위 분할은 같은 장면이 학습 · 시험에 함께 들어가 점수를 부풀림. 공식 연도 분할을 주 평가로 사용.

<p align="center"><img src="docs/figures/detection_froc.png" width="95%" alt="FROC curves for YOLOv8n and the classical matched filter on test years, with example detections"></p>
<p align="center"><sub>시험 연도 578화면 · 114표적의 FROC와 탐지 예시</sub></p>

<p align="center"><img src="docs/figures/classification_splits.png" width="95%" alt="Three-class macro F1 by evaluation split and year-split confusion matrices"></p>
<p align="center"><sub>평가 분할별 3분류 macro F1과 연도 분할 혼동 행렬</sub></p>

<p align="center"><img src="docs/figures/highlight_shadow.png" width="95%" alt="Side-scan frame, across-track highlight-shadow profiles and oriented patches"></p>
<p align="center"><sub>영상 신호처리와 하이라이트–그림자 단면</sub></p>

<p align="center"><img src="docs/figures/data_diagnosis.png" width="95%" alt="Objects per year, duplicated scene exports and box sizes"></p>
<p align="center"><sub>데이터 진단: 연도별 객체 수, 축소 복사본, 표적 크기</sub></p>

전체 결과는 `docs/milco-report.html`(단일 HTML 리포트)에서 볼 수 있습니다.

## 탑재 AI 관점의 의미

- **2단 구조**: 후보는 탐지기로 넓게 잡고(재현율), 물리 특징으로 거름(오경보 억제) → 통신이 제한된 XLUUV는 후보 패치만 송신 (64 px 패치 5개 ≈ 화면의 2%).
- **시험평가**: 중복 장면 점검과 연도 · 해역 분할, FROC 기반 허용 오경보 설정이 운용 성능 예측의 전제.
- **개선 방향**: 원시 신호(정합필터 출력) · 합성개구 소나(SAS) 자료 · 해역별 라벨 확보, GPU 학습과 더 큰 모델로 성능 향상 여지.

## 실행

```bash
pip install -r requirements.txt
cd src
./download_data.sh          # 공개 데이터 내려받기 → raw/<year>/
python screen.py            # 중복 장면 · 연도별 통계 → screen.json
python b2.py                # 수주 검출 · 이득 정규화 · 정합필터 탐지기 FROC → b2_results.json
python b2_signature.py      # 하이라이트–그림자 단면 → b2_signature.json
python prep_tiles.py        # 축척 통일 화면 + 320 px 타일
python train_yolo.py        # YOLOv8n 타일 학습 (CPU 약 1시간, Apple MPS는 device='mps')
python b3_yolo_eval.py      # 전체 화면 mAP · FROC → b3_yolo.json
python b3_patch.py 2        # 3분류: 음향 특징 / DINOv2 고정 / DINOv2 미세조정 → b3_patch.json
python build_report.py      # → milco-report.html
```

- 모든 스크립트는 `src/`에서 실행합니다. 결과 JSON과 학습된 탐지기 가중치(`src/runs/detect/runs/yolo_tiles/weights/best.pt`)는 저장소에 포함되어 있어, 학습 없이 평가 · 리포트 단계부터 다시 돌릴 수 있습니다.
- 원자료 영상, 타일, 특징 캐시(`*.npz`)는 용량 때문에 포함하지 않았습니다.

## 한계

- CPU로 학습한 소형 모델(YOLOv8n, 25 epoch)입니다.
- 입력은 JPEG로 내보낸 영상이며 원시 신호 · 이득 정보가 없습니다.
- 라벨이 없는 유사 물체가 오경보로 집계되므로 실제 오경보는 수치보다 적을 수 있습니다.
- 선행연구(Mine-JEPA 2026)는 분할 방식이 공개되지 않아 수치를 직접 비교할 수 없습니다.

## 데이터와 인용

- Santos, N. P. et al. (2024) Side-scan sonar imaging data of underwater vehicles for mine detection. *Data in Brief* 53, 110132. (CC BY 4.0)
- Mine-JEPA (2026) In-domain self-supervised learning for mine-like object classification in side-scan sonar. arXiv:2604.00383.
- Oquab, M. et al. (2024) DINOv2: Learning robust visual features without supervision. *TMLR*.
- Ultralytics YOLOv8.

## 관련 저장소

- [marine-mammal-classifier](https://github.com/jongchan-joseph-lee/marine-mammal-classifier): 수동 소나(해양포유류 음향) 분류 레퍼런스 모델 비교 (Side A). 평가 분할이 성능을 부풀리는 같은 현상을 확인함.

## 라이선스

코드는 MIT License를 따릅니다 (`LICENSE`). 데이터는 원 저작자의 CC BY 4.0 조건을 따릅니다.
