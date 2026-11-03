# 측면주사 소나(SSS) 공개 영상의 기뢰유사물체 탐지 · 분류 평가

공개 측면주사 소나(Side-Scan Sonar, SSS) 영상 자료에서 **기뢰유사물체(mine-like object) 탐지기와 패치 분류기를 각각 평가**하고, 자료의 중복 장면과 촬영 연도 변화가 평가 결과에 미치는 영향을 검토한 개인 연구입니다. 영상 기반 물리 특징(하이라이트–그림자)과 AI 모델(YOLOv8n · DINOv2)을 같은 분할 규약에서 비교했습니다.

| | |
|---|---|
| **자료** | Santos et al. 2024 (*Data in Brief*, CC BY 4.0) — Gavia AUV · Marine Sonic 900/1800 kHz, JPEG 영상 1,170장(고유 장면 753개), MILCO · NOMBO 라벨 |
| **평가** | 공식 연도 분할(학습 2015 · 2017 · 2021 → 시험 2010 · 2018), 장면 · 이미지 분할 비교, FROC · mAP@0.5 · macro F1 |
| **실행 환경** | CPU (노트북 2코어 기준 전체 약 2시간, 대부분 YOLO 학습) |
| **검증 범위** | 공개 영상 자료 내부 정량 평가까지. 원시 신호 · 탐지→분류 연계 · 실해역 · 탑재 성능은 다루지 않음 |

## 1. 배경

측면주사 소나 영상에서 기뢰유사물체는 해저면보다 밝은 **하이라이트**와 그 뒤의 **음향 그림자**로 나타납니다. Santos et al.(2024)이 공개한 AUV 측면주사 영상 자료는 이 문제를 공개 자료로 다룰 수 있는 드문 자료이며, 선행연구(Mine-JEPA 2026)는 이 자료의 물체 분류에서 macro F1 0.81–0.82를 보고했습니다. 다만 분할 방식이 공개되지 않았습니다.

자료를 먼저 점검해 보니 평가에 영향을 줄 수 있는 구조가 있었습니다.
- 같은 장면이 크기만 다르게 두 번 내보내진 영상이 많습니다(축소 복사본).
- 촬영 연도별 물체 수와 구성이 크게 다르고, 물체가 매우 작습니다(1024 px 기준 폭 수십 px).

**연구 질문**
1. 중복 장면을 정리하고 학습에 없던 연도로 시험하면 탐지 · 분류 성능은 어떻게 달라지는가?
2. 고전 정합필터 탐지기와 학습 기반 탐지기(YOLOv8n)는 같은 기준에서 어느 정도 차이가 나는가?
3. 하이라이트–그림자 물리 특징은 연도가 바뀔 때 AI 특징(DINOv2)과 비교해 얼마나 유지되는가?

## 2. 방법론

### 단계별 수행 범위

| 단계 | 내용 | 수행 범위 | 근거 |
|---|---|---|---|
| 1. 자료 점검 | 고역통과 영상 상관으로 중복 장면 검출, 연도별 물체 수 · 크기 통계 | 수행 | [`src/screen.py`](src/screen.py), [`src/screen.json`](src/screen.json) |
| 2. 영상 전처리 | 의사색상 → 그레이스케일, 영상 크기 통일(1024 px), 수주(nadir) 검출, 횡방향 이득 정규화 | 수행 | [`src/b2.py`](src/b2.py), [`src/prep_tiles.py`](src/prep_tiles.py) |
| 3. 물리 특징 | 표적 중심 횡방향 단면의 하이라이트–그림자 대비, 단일 특징 판별력(AUROC) | 수행 | [`src/b2_signature.py`](src/b2_signature.py), [`src/b2_shadow_auc.py`](src/b2_shadow_auc.py) |
| 4. 탐지 | 고전 정합필터(하이라이트 뒤 그림자) 기준선, YOLOv8n (320 px 타일 학습 → 1024 px 전체 화면 탐지, 25 epoch) | 수행 | [`src/b2.py`](src/b2.py), [`src/train_yolo.py`](src/train_yolo.py), [`src/b3_yolo_eval.py`](src/b3_yolo_eval.py) |
| 5. 패치 분류 | 정답 위치 패치 3분류(MILCO · NOMBO · 해저 배경): 물리 특징 + 로지스틱 회귀, DINOv2 고정 특징 + 선형 분류기, DINOv2 미세조정 | 수행 (정답 위치 패치 조건) | [`src/b3_patch.py`](src/b3_patch.py) |
| 6. 평가 규약 | 이미지 · 장면 · 연도 분할 비교, 탐지는 중심 거리 기준 FROC와 mAP@0.5, 분류는 macro F1 | 수행 | 결과 JSON |
| 7. 탐지 → 분류 연계 | 실제 탐지 후보를 분류기에 넣은 2단 성능 | 미수행 | — |
| 8. 원시 신호 · 실해역 · 탑재 | 원시 신호 처리, 다른 해역 · 장비, 실시간 처리 | 미수행 | — |

### 평가 규약에서 정한 것
- **분할**: 기본은 자료의 공식 연도 분할입니다. 이미지 단위 분할은 축소 복사본 때문에 같은 장면이 학습 · 시험에 함께 들어갈 수 있어 **진단용 비교 조건**으로만 사용했습니다.
- **탐지 일치 기준**: FROC는 검출 중심과 정답 중심의 거리로, mAP는 박스 IoU로 판정하므로 서로 다른 기준입니다.
- **분류 입력**: 정답 박스 위치 · 크기로 물체 패치를 잘라내고, 장면당 해저 배경 패치 2개를 추가했습니다. **YOLO 탐지 후보는 사용하지 않았습니다.**

## 3. 실행 예시

```bash
pip install -r requirements.txt
cd src
./download_data.sh          # 공개 자료 내려받기 → raw/<year>/
python screen.py            # 1. 중복 장면 · 연도별 통계        → screen.json
python b2.py                # 2·4. 전처리 · 정합필터 탐지 FROC  → b2_results.json
python b2_signature.py      # 3. 하이라이트–그림자 단면          → b2_signature.json
python b2_shadow_auc.py     # 3. 단일 특징 판별력 AUROC          → b2_shadow_auc.json
python prep_tiles.py        # 2. 1024 px 화면 + 320 px 타일
python train_yolo.py        # 4. YOLOv8n 타일 학습 (CPU, Apple MPS는 device='mps')
python b3_yolo_eval.py      # 4. 전체 화면 mAP · FROC           → b3_yolo.json
python b3_patch.py 2        # 5. 정답 위치 패치 3분류            → b3_patch.json
python build_report.py      # 결과 열람용 HTML                   → milco-report.html
```

- 모든 스크립트는 `src/`에서 실행합니다. 결과 JSON과 [학습된 탐지기 가중치](src/runs/detect/runs/yolo_tiles/weights/best.pt)는 저장소에 포함되어 있어, 원자료 없이도 결과 수치와 [결과 리포트](docs/milco-report.html)를 확인할 수 있습니다.
- 원자료 영상, 타일, 특징 캐시(`*.npz`)는 용량 때문에 포함하지 않았습니다.

## 4. 주요 결과

| 항목 | 결과 | 조건 |
|---|---|---|
| YOLOv8n 탐지 | 객체 재현율 **59.6%** (화면당 오경보 1건), **78.1%** (4건), mAP@0.5 0.101 | 시험 연도 578화면 · 정답 114개 |
| 고전 정합필터 탐지 | 객체 재현율 **4.3%** (화면당 오경보 1건) | 같은 시험 연도, 정답 115개¹ |
| 패치 3분류, 연도 분할 | 물리 특징 macro F1 **0.679**, DINOv2 고정 0.606, DINOv2 미세조정 0.645 | 정답 위치 패치 1,270개 |
| 장면 → 연도 분할 하락폭 | 물리 특징 0.040, DINOv2 고정 특징 0.146 | 같은 패치 분류 실험 |
| 하이라이트–그림자 대비 | 물체 vs 해저 배경 AUROC **0.830** (시험 연도 0.775), MILCO vs NOMBO 0.513 | 단일 특징, 분류기 없음 |
| 중복 장면 | 1,170장 중 중복 417쌍 → 고유 장면 753개 | 고역통과 영상 상관 r > 0.95 |

¹ YOLO는 1024 px 기준 폭 또는 높이가 4 px 미만인 라벨 1개를 제외해 114개입니다.

<p align="center"><img src="docs/figures/detection_froc.png" width="95%" alt="FROC curves for YOLOv8n and the classical matched filter on test years, with example detections"></p>
<p align="center"><sub>시험 연도(2010 · 2018) FROC와 탐지 예시. 정합필터는 정답 115개, YOLOv8n은 114개 기준</sub></p>

**패치 분류의 분할별 macro F1**

| 모델 | 이미지 단위 5-fold (진단용) | 장면 단위 5-fold | 연도 분할 |
|---|---|---|---|
| 물리 특징 + 로지스틱 회귀 | 0.734 | 0.719 | 0.679 |
| DINOv2 고정 특징 + 선형 분류기 | 0.814 | 0.751 | 0.606 |
| DINOv2 마지막 4블록 미세조정 | — | — | 0.645 |

근거: [`src/b3_yolo.json`](src/b3_yolo.json) (`froc_operational`, `map_operational`), [`src/b2_results.json`](src/b2_results.json) (`mf.operational`), [`src/b3_patch.json`](src/b3_patch.json), [`src/b2_shadow_auc.json`](src/b2_shadow_auc.json), [`src/screen.json`](src/screen.json). 결과 파일의 `operational`은 자료의 시험 연도 구분을 뜻합니다.

<details>
<summary><b>그림 더 보기</b> — 패치 분류 · 하이라이트–그림자 · 자료 점검</summary>

<p align="center"><img src="docs/figures/classification_splits.png" width="95%" alt="Three-class macro F1 by evaluation split and year-split confusion matrices"></p>
<p align="center"><sub>분할별 3분류 macro F1과 연도 분할 혼동 행렬 (정답 위치 패치 조건)</sub></p>

<p align="center"><img src="docs/figures/highlight_shadow.png" width="95%" alt="Side-scan frame, across-track highlight-shadow profiles and oriented patches"></p>
<p align="center"><sub>전처리 영상, 횡방향 하이라이트–그림자 단면, 단일 특징 ROC</sub></p>

<p align="center"><img src="docs/figures/data_diagnosis.png" width="95%" alt="Objects per year, duplicated scene exports and box sizes"></p>
<p align="center"><sub>자료 점검: 연도별 고유 물체 수, 축소 복사본 예시, 표적 크기</sub></p>

</details>

## 5. 고찰

**자료 중복은 분류 점수를 높게 보이게 할 수 있습니다.** DINOv2 고정 특징의 macro F1은 이미지 단위 0.814 → 장면 단위 0.751 → 연도 분할 0.606으로 낮아졌습니다. 이미지 단위 0.814는 문헌 보고치(0.81–0.82)와 비슷하지만, 같은 분할 규약으로 재평가한 값이 아니므로 성능이 같다고 볼 근거는 아닙니다. 이 자료에서는 장면 · 연도를 분리한 평가를 기본으로 두는 것이 타당하다고 판단했습니다.

**연도가 바뀔 때 물리 특징 모델의 하락이 작았습니다.** 장면 → 연도 분할에서 물리 특징 모델은 0.040, DINOv2 고정 특징 모델은 0.146 하락했습니다. 다만 이는 한 번의 연도 분할에서 관찰한 결과이며, 물리 특징이 일반적으로 더 강건하다는 결론으로 일반화하지 않습니다.

**하이라이트–그림자 특징은 물체–배경 구분 단서이지만 물체 종류 구분력은 약했습니다.** 물체 vs 해저 AUROC는 0.830이었지만 MILCO vs NOMBO는 0.513으로 우연 수준에 가까웠습니다. 물체 종류 구분에는 모양 · 크기 · 질감 같은 다른 정보가 필요해 보입니다. 이 특징은 JPEG 밝기에서 계산한 값이며 보정된 음향 산란강도가 아닙니다.

**탐지 단계가 전체 성능의 출발점입니다.** 같은 시험 연도에서 YOLOv8n은 화면당 오경보 1건 기준 59.6%, 정합필터는 4.3%의 객체를 찾았습니다. 분류 성능은 정답 위치 패치에서 잰 값이므로, 탐지에서 놓친 물체는 분류 단계에서 되찾을 수 없습니다. 실제 탐지 후보를 분류기에 연결했을 때의 재현율 · 오경보율은 아직 측정하지 않았습니다.

**남은 질문**
- 탐지 후보를 분류기로 재판정하면 오경보가 얼마나 줄고, 탐지 누락은 얼마나 늘어나는가?
- 라벨에 없는 유사 물체가 오경보로 집계된 비율은 얼마인가? (추가 라벨 검토 필요)
- 원시 신호 또는 정합필터 출력에 접근할 수 있다면, 영상 이전 단계에서 표적을 더 잘 찾을 수 있는가?

## 6. 한계

- CPU로 학습한 소형 모델(YOLOv8n, 25 epoch)입니다. 더 큰 모델과 긴 학습으로 수치가 달라질 수 있습니다.
- 입력은 JPEG로 내보낸 영상이며 원시 신호 · 이득 정보가 없습니다. 물리적 해상도가 아니라 픽셀 축척을 통일했습니다.
- FROC 값은 시험 자료에서 신뢰도 임계값을 순회해 얻은 곡선 위의 값입니다. 실제 운용 임계값은 별도 검증 자료에서 정해야 합니다.
- 분류 성능은 정답 위치 · 크기를 사용한 패치 조건의 결과이며, 탐지 후보에 대한 분류와 2단 연계 성능은 미검증입니다.
- 한 자료(한 장비 · 해역 계열)의 연도 분할 결과입니다. 다른 해역 · 장비 일반화, 실제 기뢰 식별, 실해역 운용은 검증하지 않았습니다.
- 선행연구(Mine-JEPA 2026)는 분할 방식이 공개되지 않아 수치를 직접 비교할 수 없습니다.

## 데이터와 인용

- Santos, N. P. et al. (2024) Side-scan sonar imaging data of underwater vehicles for mine detection. *Data in Brief* 53, 110132. (CC BY 4.0)
- Mine-JEPA (2026) In-domain self-supervised learning for mine-like object classification in side-scan sonar. arXiv:2604.00383.
- Oquab, M. et al. (2024) DINOv2: Learning robust visual features without supervision. *TMLR*.
- Ultralytics YOLOv8.

## 관련 저장소

- [marine-mammal-classifier-benchmark](https://github.com/jongchan-joseph-lee/marine-mammal-classifier-benchmark) — Watkins 해양포유류 음향 DB의 레퍼런스 표현 평가 (원고 초안)
- [SAS_signal_processing](https://github.com/jongchan-joseph-lee/SAS_signal_processing) — 공개 합성 SAS 원시 신호의 영상 형성 · 검증

## 라이선스

코드는 MIT License를 따릅니다 ([`LICENSE`](LICENSE)). 자료와 자료에서 파생된 그림은 원 저작자의 CC BY 4.0 조건을 따릅니다.
