# GE 주가 예측 자동 리포트 서비스 — 발표 문서

> **프로젝트명**: GE Stock Intelligence  
> **목적**: 교육용 ML 파이프라인 실습  
> **면책**: 본 결과는 학습용 예측이며 투자 조언이 아닙니다.

---

## 1. 프로젝트 개요

General Electric(`GE Aerospace`, 티커: GE)의 주가 데이터를 수집·전처리하고,  
머신러닝 회귀 모델로 **다음 거래일 종가를 예측**한 뒤,  
n8n 자동화를 통해 **매일 아침 Discord로 리포트를 발송**하는 End-to-End 서비스입니다.

### 핵심 키워드
`yfinance` · `scikit-learn` · `MLP Regression` · `FastAPI` · `Streamlit` · `n8n` · `Discord Webhook`

---

## 2. 문제 정의

| 항목 | 내용 |
|---|---|
| **예측 대상** | 다음 거래일 수익률 → 예상 종가로 환산 |
| **입력 피처** | 과거 OHLCV + 기술적 지표 28개 |
| **출력** | 다음 거래일 수익률 (회귀) |
| **모델 타입** | 지도학습 · 회귀(Regression) |

주가 예측은 분류(상승/하락)가 아닌 **연속값 회귀**로 접근합니다.  
예측 수익률을 현재 종가에 곱해 다음 거래일 예상 종가를 산출합니다.

---

## 3. 전체 아키텍처

```
Yahoo Finance (yfinance)
        │
        ▼
collect_data.py ──→ data/raw/ge_raw.csv
        │
        ▼
preprocess.py ──→ data/processed/ (train/test CSV)
        │
        ▼
train_model.py ──→ models/ (Dense .pkl, LSTM-window .pkl)
        │
        ▼
predict.py ──→ data/predictions/latest_prediction.json
        │
   ┌────┴────┐
   ▼         ▼
FastAPI     Streamlit
(포트 8000)  (포트 8501)
   │
   ▼
n8n 스케줄러 (평일 매일 오전 7시 KST)
   │
   ├──→ Discord: 예측 리포트
   ├──→ Discord: 테마주 비교 리포트 (GE·GEV·GEHC·RTX·HON·BA)
   └──→ Google Sheets: 누적 기록 저장
```

---

## 4. 데이터 수집 (미션 1)

- **출처**: Yahoo Finance (`yfinance` 라이브러리)
- **티커**: `GE` (GE Aerospace)
- **기간**: 2014-01-01 ~ 현재 (일봉)
- **행 수**: 약 3,096행
- **주요 컬럼**: `Open`, `High`, `Low`, `Close`, `Volume`

```python
# collect_data.py 핵심 코드
import yfinance as yf
df = yf.download("GE", start="2014-01-01", auto_adjust=True)
```

---

## 5. 전처리 및 피처 엔지니어링 (미션 1)

### 생성된 피처 (총 28개)

| 카테고리 | 피처 |
|---|---|
| **수익률** | `daily_return`, `target_return_next_day` |
| **이동평균** | `ma_5`, `ma_10`, `ma_20`, `ma_60` |
| **갭** | `ma_20_gap`, `ma_60_gap`, `open_gap` |
| **변동성** | `volatility_5`, `volatility_20` |
| **거래량** | `volume_change`, `volume_ma_20` |
| **RSI** | `rsi_14` |
| **MACD** | `macd`, `macd_signal`, `macd_hist` |
| **볼린저밴드** | `bb_width`, `bb_position` |

### 정규화
- 피처: `MinMaxScaler` (0~1 스케일링)
- 타깃: `MinMaxScaler` (수익률 범위)
- 학습/테스트 분리: **80% / 20%**

---

## 6. 모델 (미션 2)

> Python 3.12 + TensorFlow 2.21 / Keras 환경에서 학습합니다.  
> Apple Silicon 환경에서 XLA hang 방지를 위해 Eager 모드 강제 설정이 적용됩니다.

### Dense Regression (TensorFlow Keras)

```
입력층 (28개 피처)
   → Dense(128, ReLU)
   → Dense(64, ReLU)
   → Dense(32, ReLU)
   → 출력층(1) — 다음 거래일 수익률
```

| 하이퍼파라미터 | 값 |
|---|---|
| Optimizer | Adam (lr=0.001) |
| 최대 에폭 | 30 |
| 배치 크기 | 32 |
| Early Stopping | patience=12 |
| ReduceLROnPlateau | patience=6, factor=0.5 |
| 검증 비율 | 20% |

### LSTM Regression (TensorFlow Keras)

과거 **30거래일** 시퀀스를 입력으로 사용해 시계열 패턴을 학습합니다.

```
입력: 30일 × 28 피처 시퀀스
   → LSTM(64, return_sequences=True)
   → LSTM(32)
   → Dense(16, ReLU)
   → 출력층(1) — 다음 거래일 수익률
```

---

## 7. 모델 평가 결과 (미션 2)

### 평가 지표

| 지표 | 설명 |
|---|---|
| **MAE** | 평균 절대 오차 (달러) |
| **RMSE** | 평균 제곱근 오차 (달러) |
| **MAPE** | 평균 절대 퍼센트 오차 (%) |

### 결과 비교표

| 모델 | MAE ($) | RMSE ($) | MAPE (%) |
|---|---|---|---|
| **Baseline** (오늘=내일) | **3.02** | **4.44** | **1.40** |
| Dense Regression (TF Keras) | 6.45 | 9.59 | 2.59 |
| **LSTM Regression (TF Keras)** | **3.13** | **4.58** | **1.44** |
### 해석
- Baseline이 성능이 높은 것은 **주가의 자기상관(autocorrelation)** 때문입니다.
- 주가는 "오늘 가격 ≈ 내일 가격" 경향이 강해, 단순 Baseline을 이기기 어렵습니다.
- LSTM-window가 Dense보다 좋은 성능 → **시계열 순서 정보**가 유효함을 시사합니다.
- 모델 개선보다는 **외부 데이터(뉴스·VIX·금리) 추가**가 더 효과적일 것으로 판단됩니다.

---

## 8. 서비스 자동화 (미션 3)

### FastAPI 예측 서버

| 엔드포인트 | 메서드 | 설명 |
|---|---|---|
| `/predict?model=dense` | GET | 다음 거래일 종가 예측 |
| `/compare` | GET | 테마주 6종 비교 리포트 |
| `/health` | GET | 서버 및 모델 상태 확인 |

### n8n 자동화 워크플로우

```
Weekday Schedule (매일 오전 7시 KST)
  ├── /predict 호출 → 예측 메시지 → Discord 전송
  │                              → Google Sheets 기록
  └── /compare 호출 → 비교 리포트 → Discord 전송
```

### Discord 알림 예시

**① 예측 메시지**
```
📊 GE Daily Prediction
기준일: 2026-04-24
현재 종가: $284.60
예측 다음 거래일 종가: $299.76
예상 수익률: +5.33% 📈 상승 예상
RSI(14): 47.97 | 변동성(20일): 3.20%
```

**② 테마주 비교 리포트**
```
📊 GE 테마주 비교 리포트 (기준일: 2026-04-24)

종목   이름              현재가      1개월      1년
GE    GE Aerospace    $284.60   ▼ -0.22%  ▲+44.93%
GEV   GE Vernova    $1,149.19  ▲+31.62%  ▲+219.49%
GEHC  GE HealthCare   $68.83   ▼ -3.74%   ▲+2.85%
RTX   RTX(라이시온)   $174.26   ▼ -9.64%  ▲+45.28%
HON   Honeywell      $213.17   ▼ -5.33%  ▲+15.09%
BA    Boeing         $232.44  ▲+19.59%  ▲+31.87%
```

---

## 9. Streamlit 대시보드

**주소**: `http://localhost:8501`

| 탭 | 내용 |
|---|---|
| **가격 추세** | GE 최근 2년 종가 인터랙티브 차트 |
| **예측 히스토리** | 누적 예측 기록 테이블 |
| **테마주 비교** | 6종목 정규화 차트·수익률표·상관관계 히트맵 |
| **서비스 구조** | n8n 자동화 흐름 설명 |

---

## 10. 기술 스택 요약

| 분류 | 기술 |
|---|---|
| **언어** | Python 3.13 |
| **데이터** | yfinance, pandas, numpy |
| **ML** | scikit-learn (MLPRegressor) |
| **지표** | RSI, MACD, Bollinger Bands |
| **API 서버** | FastAPI + uvicorn |
| **대시보드** | Streamlit + Plotly |
| **자동화** | n8n (HTTP Request 기반) |
| **알림** | Discord Webhook |
| **저장** | JSON, CSV, Google Sheets |

---

## 11. 한계점 및 개선 방향

### 한계점
- 주가는 **뉴스·실적·금리·시장 심리** 등 외부 요인에 크게 영향받음
- 기술적 지표만으로는 Baseline(오늘=내일) 대비 우위를 가지기 어려움
- 과거 패턴이 미래에 반복된다는 가정에 의존

### 개선 방향

| 방향 | 구체적 방법 |
|---|---|
| **외부 데이터 추가** | S&P 500, VIX, 미국 국채 금리, 달러 인덱스 |
| **뉴스 감성분석** | FinBERT 모델로 뉴스 헤드라인 감성 점수화 |
| **앙상블** | MLP + Random Forest + XGBoost 앙상블 |
| **모델 고도화** | Transformer 기반 시계열 모델 (Informer, PatchTST) |
| **서비스 확장** | 다른 종목으로 확장 (S&P 500 구성 종목 전체) |

---

## 12. 프로젝트 파일 구조

```
GE/
├── src/
│   ├── collect_data.py     # 데이터 수집
│   ├── preprocess.py       # 전처리 및 피처 생성
│   ├── indicators.py       # 기술적 지표 계산
│   ├── train_model.py      # 모델 학습
│   ├── predict.py          # 단일 예측 실행
│   ├── compare.py          # 테마주 비교 분석
│   ├── api.py              # FastAPI 서버
│   ├── run_pipeline.py     # 전체 파이프라인 실행
│   └── config.py           # 경로/하이퍼파라미터 설정
├── app.py                  # Streamlit 대시보드
├── data/
│   ├── raw/                # 원본 주가 데이터
│   ├── processed/          # 전처리 데이터 (train/test)
│   └── predictions/        # 예측 결과 JSON/CSV
├── models/                 # 학습된 모델 .pkl
├── reports/
│   ├── metrics.json        # 모델 평가 결과
│   └── figures/            # 학습 loss, 예측 그래프
└── n8n/
    ├── workflow_export.json # n8n 워크플로우
    └── README.md           # n8n 설정 가이드
```
