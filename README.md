# GE Stock Intelligence

> GE Aerospace 주가 데이터를 수집·전처리하고 **TensorFlow Keras 회귀 모델**로 다음 거래일 종가를 예측한 뒤,  
> **n8n 자동화**를 통해 매일 아침 **Discord**로 리포트를 발송하는 End-to-End ML 서비스입니다.

## 과제 미션

| 미션 | 내용 |
|---|---|
| **Mission 1** | `yfinance`로 GE 주가 수집 → 기술 지표 28개 생성 → MinMaxScaler 정규화 → 80/20 분리 |
| **Mission 2** | TensorFlow Keras — Dense Regression + LSTM Regression 학습 및 MAE/RMSE/MAPE 평가 |
| **Mission 3** | FastAPI 예측 서버 + n8n 스케줄러 → Discord Webhook 자동 발송 (평일 매일 오전 7시 KST) |

---

## 프로젝트 구조

```
GE/
├── src/
│   ├── collect_data.py     # 데이터 수집 (yfinance)
│   ├── preprocess.py       # 전처리 & 피처 생성
│   ├── indicators.py       # 기술적 지표 계산 (RSI, MACD, BB 등)
│   ├── train_model.py      # TF Dense + LSTM 모델 학습
│   ├── predict.py          # 단일 예측 실행 & JSON 저장
│   ├── compare.py          # 테마주 비교 분석 (GE/GEV/GEHC/RTX/HON/BA)
│   ├── api.py              # FastAPI 예측 서버
│   ├── run_pipeline.py     # 전체 파이프라인 순차 실행
│   └── config.py           # 경로 & 하이퍼파라미터 설정
├── app.py                  # Streamlit 대시보드
├── data/
│   ├── raw/                # 원본 주가 CSV
│   ├── processed/          # 전처리 데이터 (ge_train.csv / ge_test.csv)
│   └── predictions/        # 예측 결과 (latest_prediction.json / history.csv)
├── models/
│   ├── ge_dense_model.keras
│   ├── ge_lstm_model.keras
│   ├── feature_scaler.pkl
│   ├── target_scaler.pkl
│   └── feature_columns.json
├── reports/
│   ├── metrics.json                  # 모델 평가 결과
│   ├── figures/                      # 학습 loss 그래프, 예측 비교 그래프
│   ├── presentation_outline.md       # 발표 문서
│   ├── notebooklm_prompts.md         # NotebookLM PPT 생성 프롬프트
│   └── GE_Stock_Intelligence.pptx   # 발표 PPT (16장)
├── n8n/
│   ├── workflow_export.json          # n8n 워크플로우 (Discord 연동)
│   └── README.md                     # n8n 설정 가이드
└── requirements.txt
```

---

## 설치 (macOS / Python 3.12)

> TensorFlow 2.21은 Python 3.9~3.12를 지원합니다. **Python 3.13 사용 시 학습이 hang됩니다.**

```bash
# Python 3.12 설치 (Homebrew)
brew install python@3.12

# 가상환경 생성 및 패키지 설치
cd /path/to/GE
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

---

## 전체 파이프라인 실행

```bash
# 데이터 수집 → 전처리 → 학습 → 예측을 순서대로 실행
.venv/bin/python src/run_pipeline.py
```

단계별 개별 실행:

```bash
.venv/bin/python src/collect_data.py          # 1. 데이터 수집
.venv/bin/python src/preprocess.py            # 2. 전처리 & 피처 생성
TRAIN_EPOCHS=30 .venv/bin/python src/train_model.py   # 3. 모델 학습
.venv/bin/python src/predict.py               # 4. Dense 예측
.venv/bin/python src/predict.py --model lstm  # 4. LSTM 예측
```

LSTM을 건너뛰고 Dense만 학습:

```bash
.venv/bin/python src/train_model.py --skip-lstm
```

---

## 데이터 & 전처리 (Mission 1)

- **출처**: Yahoo Finance (`yfinance`)
- **티커**: `GE` (GE Aerospace)
- **기간**: 2014-01-01 ~ 현재 (일봉, 약 3,096행)

생성 피처 (총 28개):

| 카테고리 | 피처 |
|---|---|
| 수익률 | `daily_return`, `target_return_next_day_scaled` |
| 이동평균 | `ma_5`, `ma_10`, `ma_20`, `ma_60` |
| 갭 | `ma_20_gap`, `ma_60_gap`, `open_gap` |
| 변동성 | `volatility_5`, `volatility_20` |
| 거래량 | `volume_change`, `volume_ma_20_ratio` |
| RSI | `rsi_14` |
| MACD | `macd`, `macd_signal`, `macd_hist` |
| 볼린저밴드 | `bb_width`, `bb_position` |

정규화: `MinMaxScaler` / 학습·테스트 분리: **80% / 20%**

---

## 모델 (Mission 2)

### 알고리즘 선택 근거

| | Dense (MLP) | LSTM |
|---|---|---|
| 입력 | 당일 28개 피처 (스냅샷) | 과거 30일 × 28개 (시퀀스) |
| 시간 정보 | 고려 안 함 | 게이트(망각·입력·출력)로 기억 |
| 장점 | 빠름, 범용성 | 장기 시계열 패턴 학습 |

주가는 시계열 데이터이므로 LSTM이 과거 흐름을 게이트로 선별 기억해 더 정확한 예측이 가능합니다.  
두 모델을 모두 학습해 시계열 정보의 유효성을 직접 검증했습니다.

### Dense Regression 구조

```
Input(28) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(32, ReLU) → Output(1)
```

### LSTM Regression 구조

```
Input(30일×28) → LSTM(64, return_seq=True) → LSTM(32) → Dense(16, ReLU) → Output(1)
```

### 공통 하이퍼파라미터

| 항목 | 값 |
|---|---|
| Optimizer | Adam (lr=0.001) |
| Loss | MSE |
| Epochs | 30 (EarlyStopping patience=12) |
| Batch Size | 32 |
| 검증 비율 | 20% |

### 평가 결과

| 모델 | MAE ($) | RMSE ($) | MAPE (%) |
|---|---|---|---|
| Baseline (오늘=내일) | 3.02 | 4.44 | 1.40 |
| Dense Regression | 6.45 | 9.59 | 2.59 |
| **LSTM Regression** | **3.13** | **4.58** | **1.44** |

> LSTM이 Dense 대비 MAE 기준 약 52% 낮은 오차를 기록했습니다.  
> Baseline이 높은 것은 주가의 강한 자기상관(오늘 ≈ 내일) 때문입니다.

---

## 서비스 자동화 (Mission 3)

### FastAPI 예측 서버

```bash
.venv/bin/uvicorn src.api:app --reload --port 8000
```

| 엔드포인트 | 설명 |
|---|---|
| `GET /predict?model=dense` | Dense 모델로 다음 거래일 종가 예측 |
| `GET /predict?model=lstm` | LSTM 모델로 다음 거래일 종가 예측 |
| `GET /compare` | 테마주 6종 비교 리포트 (GE/GEV/GEHC/RTX/HON/BA) |
| `GET /health` | 서버 & 모델 파일 상태 확인 |

### n8n 워크플로우

1. `n8n/workflow_export.json`을 n8n에 Import
2. **Schedule Trigger** → 평일 오전 7시 KST 자동 실행
3. **HTTP Request** → `http://localhost:8000/predict` 호출
4. **Discord Webhook** → 예측 리포트 발송
5. **HTTP Request** → `http://localhost:8000/compare` 호출
6. **Discord Webhook** → 테마주 비교 리포트 발송

> Discord Webhook URL 설정: `n8n/README.md` 참고

### Discord 예측 메시지 예시

```
📊 GE Daily Prediction
─────────────────────
기준일: 2026-04-24
현재 종가: $284.60
예측 종가: $296.88 ▲
예상 수익률: +4.31% 📈
─────────────────────
RSI(14): 47.97 | 변동성(20일): 3.20%
모델: TensorFlow LSTM Regression
⚠️ 본 결과는 교육용 예측이며 투자 조언이 아닙니다.
```

---

## Streamlit 대시보드

```bash
.venv/bin/streamlit run app.py
# → http://localhost:8501
```

| 탭 | 내용 |
|---|---|
| 📈 가격 추세 | GE 최근 2년 종가 인터랙티브 차트 |
| 🔮 예측 히스토리 | 누적 예측 기록 테이블 |
| 🔀 테마주 비교 | 정규화 차트·수익률표·상관관계 히트맵 |
| 🏗️ 서비스 구조 | n8n 자동화 흐름 설명 |

---

## 기술 스택

| 분류 | 기술 |
|---|---|
| 언어 | Python 3.12 |
| 데이터 수집 | yfinance, pandas, numpy |
| ML 프레임워크 | TensorFlow 2.21 / Keras |
| 정규화 | scikit-learn MinMaxScaler |
| API 서버 | FastAPI + uvicorn |
| 대시보드 | Streamlit + Plotly |
| 자동화 | n8n (HTTP Request 노드) |
| 알림 | Discord Webhook |
| 저장 | JSON, CSV |

---

## macOS TensorFlow 실행 시 주의사항

Apple Silicon(M1/M2/M3) + macOS에서 TF 학습이 hang되는 경우, `train_model.py`에 아래 설정이 적용되어 있습니다:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
tf.config.set_visible_devices([], "GPU")
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
```

> `from tensorflow import keras`는 반드시 위 설정 **이후**에 import해야 적용됩니다.

---

## 한계점 & 개선 방향

- 주가는 뉴스·금리·실적·시장 심리 등 외부 요인에 크게 영향받음
- 기술 지표만으로는 Baseline(오늘≈내일) 대비 일관된 우위 달성 어려움
- 개선: S&P 500·VIX·금리 데이터 추가 / FinBERT 뉴스 감성 분석 / Transformer 모델 (PatchTST)

---

> ⚠️ **면책**: 본 프로젝트는 교육용 ML 실습이며 투자 조언이 아닙니다.
