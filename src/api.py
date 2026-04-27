"""
GE 주가 예측 API 서버
실행: .venv/bin/uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
n8n에서 GET http://localhost:8000/predict?model=dense 로 호출
n8n에서 GET http://localhost:8000/compare 로 테마주 비교 리포트 호출
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from collect_data import collect_stock_data
from compare import get_comparison_data, THEME_TICKERS
from config import DENSE_MODEL_PATH, LSTM_MODEL_PATH, START_DATE, TICKER, ensure_directories
from predict import (
    build_prediction_payload,
    load_feature_columns,
    predict_dense,
    predict_lstm,
    prepare_latest_features,
    save_prediction,
    format_message,
)

app = FastAPI(
    title="GE Stock Prediction API",
    description="GE 주가 다음 거래일 종가 예측 API (교육용)",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"status": "ok", "endpoints": ["/predict", "/compare", "/health"]}


@app.get("/health")
def health():
    dense_ready = DENSE_MODEL_PATH.exists()
    lstm_ready = LSTM_MODEL_PATH.exists()
    return {
        "status": "ok",
        "dense_model": dense_ready,
        "lstm_model": lstm_ready,
    }


@app.get("/predict")
def predict(model: str = Query(default="dense", pattern="^(dense|lstm)$")):
    ensure_directories()

    if model == "dense" and not DENSE_MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Dense 모델이 없습니다. train_model.py를 먼저 실행하세요.")
    if model == "lstm" and not LSTM_MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="LSTM 모델이 없습니다. train_model.py를 먼저 실행하세요.")

    try:
        feature_columns = load_feature_columns()
        raw_data = collect_stock_data(ticker=TICKER, start_date=START_DATE)
        prepared_data = prepare_latest_features(raw_data, feature_columns)

        if model == "lstm":
            predicted_return = predict_lstm(prepared_data, feature_columns)
            model_name = "sklearn MLP LSTM-window Regression"
        else:
            predicted_return = predict_dense(prepared_data, feature_columns)
            model_name = "sklearn MLP Dense Regression"

        payload = build_prediction_payload(prepared_data, predicted_return, model_name)
        save_prediction(payload)
        payload["message"] = format_message(payload)

        return JSONResponse(content=payload)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/compare")
def compare():
    """GE 관련 테마주 비교 리포트 (Discord 메시지 포함)."""
    try:
        data = get_comparison_data(start_date="2023-01-01")
        returns = data["returns_table"]
        volatility = data["volatility"]
        latest = data["latest_prices"]
        latest_date = data["latest_date"]
        ticker_labels = data["ticker_labels"]

        def ret_str(ticker: str, period: str) -> str:
            try:
                val = returns.loc[ticker, period]
                if val != val:
                    return "—"
                sign = "+" if val >= 0 else ""
                arrow = "▲" if val >= 0 else "▼"
                return f"{arrow} {sign}{val:.2f}%"
            except Exception:
                return "—"

        def strength_bar(val: float, max_val: float = 50.0) -> str:
            filled = min(int(abs(val) / max_val * 8), 8)
            bar = "█" * filled + "░" * (8 - filled)
            return bar

        lines = [
            "## 📊 GE 테마주 비교 리포트",
            f"> **기준일:** {latest_date}",
            "",
            "### 🔢 현재가 & 수익률",
            "```",
            f"{'종목':<6} {'이름':<18} {'현재가':>8}  {'1개월':>8}  {'3개월':>8}  {'1년':>8}  {'YTD':>8}",
            "─" * 70,
        ]

        for ticker, label in ticker_labels.items():
            price = latest.get(ticker, float("nan"))
            price_str = f"${price:,.2f}" if price == price else "—"
            lines.append(
                f"{ticker:<6} {label:<18} {price_str:>8}  "
                f"{ret_str(ticker, '1개월'):>10}  "
                f"{ret_str(ticker, '3개월'):>10}  "
                f"{ret_str(ticker, '1년'):>10}  "
                f"{ret_str(ticker, 'YTD'):>10}"
            )

        lines.append("```")
        lines.append("")
        lines.append("### 📉 연율화 변동성 (20일)")
        lines.append("```")
        max_vol = float(volatility.max()) if not volatility.empty else 50.0
        for ticker in volatility.index:
            vol = volatility[ticker]
            bar = strength_bar(vol, max_vol)
            lines.append(f"{ticker:<6} {bar}  {vol:.1f}%")
        lines.append("```")

        lines += [
            "",
            "### 🔗 상관관계 요약 (GE 기준, 최근 6개월)",
        ]
        corr = data["correlation"]
        if "GE" in corr.columns:
            ge_corr = corr["GE"].drop("GE").sort_values(ascending=False)
            lines.append("```")
            for ticker, val in ge_corr.items():
                bar = strength_bar(val, 1.0)
                lines.append(f"{ticker:<6} {bar}  {val:+.2f}")
            lines.append("```")

        lines.append("-# ⚠️ 본 데이터는 교육용이며 투자 권유가 아닙니다.")

        discord_message = "\n".join(lines)

        return JSONResponse(content={
            "latest_date": latest_date,
            "discord_message": discord_message,
            "returns": returns.to_dict(),
            "volatility": volatility.to_dict(),
            "latest_prices": latest.to_dict(),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
