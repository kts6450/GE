# GE 주가 예측 자동 리포트 서비스

General Electric 티커 `GE`의 주가 데이터를 수집하고, 기술적 지표 기반 전처리 후 TensorFlow 회귀 모델로 다음 거래일 수익률을 예측하는 프로젝트입니다. 예측 수익률은 현재 종가에 반영해 다음 거래일 예상 종가로 환산하며, 결과는 JSON/CSV로 저장되고 n8n 워크플로우를 통해 Google Sheets 기록과 Telegram 알림으로 서비스화할 수 있습니다.

## 과제 미션 매핑

- 미션 1: `yfinance`로 GE 주가 데이터를 수집하고 결측치 처리, 기술적 지표 생성, 정규화, 학습/테스트 분리를 수행합니다.
- 미션 2: TensorFlow Dense Regression과 LSTM Regression 모델을 학습하고 MAE, RMSE, MAPE로 평가합니다.
- 미션 3: n8n Schedule Trigger가 Python 예측 스크립트를 실행하고 결과를 Google Sheets 및 Telegram으로 전송합니다.

## 프로젝트 구조

```text
ge-stock-prediction-service/
  data/
    raw/
    processed/
    predictions/
  models/
  n8n/
    start-n8n.cmd
    start-n8n.ps1
    workflow_export.json
  start-n8n.cmd
  reports/
    figures/
    presentation_outline.md
  src/
    collect_data.py
    config.py
    indicators.py
    predict.py
    preprocess.py
    run_pipeline.py
    train_model.py
  app.py
  requirements.txt
```

## 설치

```powershell
cd C:\Users\kts64\Documents\ML\ge-stock-prediction-service
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

PowerShell 실행 정책 때문에 `Activate.ps1`이 막힐 수 있으므로, 이 프로젝트는 가상환경을 활성화하지 않고 `.venv\Scripts\python.exe`를 직접 호출하는 방식을 권장합니다.

## 전체 파이프라인 실행

```powershell
.\.venv\Scripts\python.exe src\run_pipeline.py
```

개별 실행도 가능합니다.

```powershell
.\.venv\Scripts\python.exe src\collect_data.py
.\.venv\Scripts\python.exe src\preprocess.py
.\.venv\Scripts\python.exe src\train_model.py
.\.venv\Scripts\python.exe src\predict.py --model dense
```

LSTM 학습까지 시간이 오래 걸리면 Dense 모델만 학습할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe src\train_model.py --skip-lstm
```

## Streamlit 대시보드 실행

파이프라인을 한 번 실행해 `latest_prediction.json`, `metrics.json`, 그래프 파일이 생성된 뒤 대시보드를 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

대시보드 화면 구성:

- 다크 히어로 + KPI 그리드: 티커 배지, 기준일, 마지막 실행 시각(UTC)
- 모델 성능 표는 HTML 스타일 테이블로 표시해 `st.dataframe`이 불러오는 `pyarrow` 경로를 피함
- 가격 추세는 Plotly 인터랙티브 라인차트(확대/이동)로 표시
- 실제값 vs 예측값은 학습 시 생성한 PNG, 사이드바에 빠른 가이드
- 누적 예측 히스토리, n8n 자동화 흐름 및 교육용 고지

## 데이터와 전처리

수집 데이터는 `data/raw/ge_raw.csv`에 저장됩니다. 전처리 후 데이터는 `data/processed/`에 저장됩니다.

생성되는 주요 피처:

- OHLCV: `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`
- 수익률: `daily_return`, `target_return_next_day`
- 이동평균: `ma_5`, `ma_10`, `ma_20`, `ma_60`
- 변동성: `volatility_5`, `volatility_20`
- 거래량 변화: `volume_change`, `volume_ma_20`
- 기술적 지표: `rsi_14`, `macd`, `bb_width`, `bb_position`

## 모델

기본 모델은 Dense Regression입니다. 입력은 정규화된 기술적 지표이고 출력은 다음 거래일 수익률입니다. 예측 수익률을 현재 종가에 곱해 다음 거래일 예상 종가를 계산합니다.

추가 모델로 LSTM Regression을 제공합니다. 최근 30거래일 시퀀스를 입력으로 사용해 다음 거래일 수익률을 예측합니다.

모델 평가는 아래 기준으로 저장됩니다.

- Baseline: 오늘 종가를 내일 종가로 가정
- Dense Regression
- LSTM Regression
- MAE, RMSE, MAPE

결과 파일:

- `models/ge_dense_model.keras`
- `models/ge_lstm_model.keras`
- `reports/metrics.json`
- `reports/figures/dense_training_loss.png`
- `reports/figures/lstm_training_loss.png`
- `reports/figures/actual_vs_predicted.png`

## n8n 연결

`n8n/workflow_export.json`을 n8n에 import합니다.

**실행:** 프로젝트 루트의 **`start-n8n.cmd`** 더블클릭, 또는 `n8n\start-n8n.cmd` / `n8n\start-n8n.ps1`. **여기서는 터미널에 `n8n start` 만 치지 마세요(항상 5678)** — Teruten TCube 때문에 실패합니다. **포트는 스크립트가 자동 선택**하고, 콘솔·`n8n/.n8n-ui-port`에 주소가 나옵니다. `n8n`이 PATH에 있어야 합니다(`npm i -g n8n`). 상세: **`n8n/README.md`**.

워크플로우 구성:

- Weekday Schedule: 평일 매일 실행
- Execute Command: `python src\predict.py --model dense`
- Code: Python 출력에서 JSON 예측 결과 파싱
- Google Sheets: 예측 결과 누적 저장
- Telegram: 예측 리포트 발송

Google Sheets와 Telegram 노드는 기본적으로 비활성화되어 있습니다. n8n에서 credential과 시트 ID, Telegram chat ID를 입력한 뒤 활성화하면 됩니다.

## 예측 메시지 예시

```text
[GE Daily Prediction]
기준일: 2026-04-26
현재 종가: $158.2
예측 다음 거래일 종가: $160.05
예상 수익률: 1.17%
모델: TensorFlow Dense Regression
최근 20일 변동성: 2.34%
주의: 본 결과는 학습용 예측이며 투자 조언이 아닙니다.
```

## 발표 시 강조점

이 프로젝트는 실제 투자 추천 서비스가 아니라 교육용 회귀 예측 실험입니다. 주가는 뉴스, 금리, 실적 발표, 시장 심리 등 외부 요인의 영향을 크게 받기 때문에 예측값은 참고용으로만 사용해야 합니다.

개선 방향:

- S&P 500, VIX, 금리 데이터 추가
- 뉴스 감성분석 추가
- Dense, LSTM, Random Forest 등 앙상블 비교
- Streamlit 대시보드에 기간 필터와 모델 선택 기능 추가
