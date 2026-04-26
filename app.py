import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
LATEST_PREDICTION_PATH = PROJECT_ROOT / "data" / "predictions" / "latest_prediction.json"
PREDICTION_HISTORY_PATH = PROJECT_ROOT / "data" / "predictions" / "prediction_history.csv"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
PREDICTION_FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "actual_vs_predicted.png"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ge_raw.csv"

ACCENT = "#6366f1"
SURFACE = "#f8fafc"


st.set_page_config(
    page_title="GE · Stock Intelligence",
    page_icon="GE",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def format_currency(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"${value:,.2f}"


def format_percent(value: float | int | None) -> str:
    if value is None:
        return "-"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.4f}%"


def render_global_css() -> None:
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"]  {{
            font-family: "Segoe UI", "Malgun Gothic", system-ui, sans-serif;
        }}
        .stApp {{
            background: linear-gradient(180deg, {SURFACE} 0%, #e2e8f0 100%);
        }}
        [data-testid="stHeader"] {{
            background: rgba(15, 23, 42, 0.92);
        }}
        [data-testid="stToolbar"] {{
            color: #f1f5f9;
        }}
        .block-container {{
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
            max-width: 1200px;
        }}
        .hero-wrap {{
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #312e81 100%);
            border-radius: 20px;
            padding: 1.75rem 1.9rem 1.5rem 1.9rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 16px 40px rgba(15, 23, 42, 0.35);
        }}
        .hero-top {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem 1rem;
        }}
        .ticker-pill {{
            display: inline-block;
            background: rgba(99, 102, 241, 0.35);
            color: #e0e7ff;
            font-weight: 700;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            border: 1px solid rgba(165, 180, 252, 0.45);
        }}
        .hero-title {{
            color: #f8fafc;
            font-size: 1.65rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            line-height: 1.2;
            margin: 0.4rem 0 0.35rem 0;
        }}
        .hero-sub {{
            color: #cbd5e1;
            font-size: 0.95rem;
            line-height: 1.5;
            max-width: 52rem;
        }}
        .hero-meta {{
            text-align: right;
            min-width: 10rem;
        }}
        .hero-meta-k {{
            color: #94a3b8;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        .hero-meta-v {{
            color: #f1f5f9;
            font-size: 0.95rem;
            font-weight: 600;
            margin-top: 0.15rem;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 1.1rem;
        }}
        @media (max-width: 900px) {{
            .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        }}
        @media (max-width: 520px) {{
            .kpi-grid {{ grid-template-columns: 1fr; }}
        }}
        .kpi-card {{
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            min-height: 100px;
            backdrop-filter: blur(8px);
        }}
        .kpi-label {{
            color: #94a3b8;
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.4rem;
        }}
        .kpi-value {{
            color: #f8fafc;
            font-size: 1.45rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }}
        .kpi-hint {{
            color: #94a3b8;
            font-size: 0.8rem;
            margin-top: 0.4rem;
            line-height: 1.35;
        }}
        .kpi-up {{ color: #34d399; }}
        .kpi-down {{ color: #fb7185; }}
        .section-title {{
            color: #0f172a;
            font-size: 1.05rem;
            font-weight: 700;
            margin: 0 0 0.85rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .section-title::before {{
            content: "";
            display: inline-block;
            width: 4px;
            height: 1.1rem;
            border-radius: 2px;
            background: {ACCENT};
        }}
        .insight-pill {{
            display: inline-block;
            background: #ecfeff;
            color: #0e7490;
            font-size: 0.8rem;
            font-weight: 600;
            padding: 0.4rem 0.75rem;
            border-radius: 999px;
            border: 1px solid #a5f3fc;
        }}
        .data-table-wrap {{
            overflow-x: auto;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
        }}
        .data-table-wrap table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        .data-table-wrap th {{
            background: #f1f5f9;
            color: #334155;
            font-weight: 700;
            text-align: left;
            padding: 0.65rem 0.75rem;
            border-bottom: 1px solid #e2e8f0;
        }}
        .data-table-wrap td {{
            padding: 0.6rem 0.75rem;
            border-bottom: 1px solid #f1f5f9;
            color: #0f172a;
        }}
        .data-table-wrap tr:last-child td {{ border-bottom: none; }}
        .notice-box {{
            border-left: 4px solid {ACCENT};
            background: #eef2ff;
            padding: 0.85rem 1rem;
            border-radius: 12px;
            color: #312e81;
            font-size: 0.9rem;
        }}
        div[data-baseweb="tab-list"] button {{
            font-weight: 600;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metrics_to_dataframe(metrics: dict) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame()

    labels = {
        "baseline_today_equals_tomorrow": "Baseline",
        "dense_regression": "Dense Regression",
        "lstm_regression": "LSTM Regression",
    }
    rows = []
    for key, values in metrics.items():
        rows.append(
            {
                "Model": labels.get(key, key),
                "MAE": values.get("mae"),
                "RMSE": values.get("rmse"),
                "MAPE (%)": values.get("mape"),
            }
        )
    return pd.DataFrame(rows)


def render_data_table(data: pd.DataFrame, float_format: str = "{:.4f}") -> None:
    if data.empty:
        st.info("표시할 데이터가 없습니다.")
        return

    formatted = data.copy()
    for column in formatted.select_dtypes(include=["float", "float64", "float32"]).columns:
        formatted[column] = formatted[column].map(lambda value: float_format.format(value))
    for column in formatted.columns:
        if formatted[column].dtype == object:
            formatted[column] = formatted[column].map(lambda v: "" if v is None else v)

    inner = formatted.to_html(index=False, escape=True)
    st.markdown(
        f'<div class="data-table-wrap">{inner}</div>',
        unsafe_allow_html=True,
    )


def render_hero(
    prediction: dict,
) -> None:
    generated = html.escape(str(prediction.get("generated_at_utc", "—")))
    last_date = html.escape(str(prediction.get("last_market_date", "—")))
    model_name = html.escape(str(prediction.get("model", "—")))

    predicted_return = prediction.get("predicted_next_return_pct", 0) or 0
    return_class = "kpi-up" if predicted_return >= 0 else "kpi-down"
    direction = "상승 예상" if predicted_return >= 0 else "하락 예상"
    up_icon = "▲" if predicted_return >= 0 else "▼"

    c1 = format_currency(prediction.get("current_close"))
    c2 = format_currency(prediction.get("predicted_next_close"))
    c3 = format_percent(predicted_return)
    c4 = format_percent(prediction.get("volatility_20_pct"))
    rsi = html.escape(str(prediction.get("rsi_14", "—")))

    st.markdown(
        f"""
        <div class="hero-wrap">
            <div class="hero-top">
                <div>
                    <span class="ticker-pill">NYSE · GE</span>
                    <h1 class="hero-title">GE Stock Intelligence</h1>
                    <p class="hero-sub">
                        TensorFlow 회귀 모델이 다음 거래일 수익률을 추정하고, 오늘 종가에 반영한 예상 종가를 보여줍니다.
                    </p>
                </div>
                <div class="hero-meta">
                    <div class="hero-meta-k">Last run (UTC)</div>
                    <div class="hero-meta-v">{generated}</div>
                    <div class="hero-meta-k" style="margin-top:0.65rem">기준일</div>
                    <div class="hero-meta-v">{last_date}</div>
                </div>
            </div>
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">현재 종가</div>
                    <div class="kpi-value">{html.escape(c1)}</div>
                    <div class="kpi-hint">기준 시장일 {last_date}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">예측 다음 거래일 종가</div>
                    <div class="kpi-value">{html.escape(c2)}</div>
                    <div class="kpi-hint">{model_name}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">예상 수익률 {up_icon}</div>
                    <div class="kpi-value {return_class}">{html.escape(c3)}</div>
                    <div class="kpi-hint">{html.escape(direction)}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">최근 20일 변동성</div>
                    <div class="kpi-value">{html.escape(c4)}</div>
                    <div class="kpi-hint">RSI(14): {rsi}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_price_figure(raw_data: pd.DataFrame) -> go.Figure:
    chart_data = raw_data.copy()
    chart_data["Date"] = pd.to_datetime(chart_data["Date"])
    chart_data = chart_data.tail(520)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_data["Date"],
            y=chart_data["Close"],
            mode="lines",
            name="Close",
            line=dict(color=ACCENT, width=2.2),
            hovertemplate="%{x|%Y-%m-%d}<br>Close: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=0, r=0, t=20, b=0),
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fafafa",
        font=dict(family="Segoe UI, sans-serif", color="#0f172a"),
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0", showline=False),
        yaxis=dict(
            showgrid=True,
            gridcolor="#e2e8f0",
            tickprefix="$",
            tickformat=",.0f",
        ),
        hovermode="x unified",
        showlegend=False,
    )
    return fig


def render_dashboard() -> None:
    render_global_css()

    prediction = load_json(LATEST_PREDICTION_PATH)
    metrics = load_json(METRICS_PATH)
    history = load_csv(PREDICTION_HISTORY_PATH)
    raw_data = load_csv(RAW_DATA_PATH)

    with st.sidebar:
        st.markdown("### GE Intelligence")
        st.caption("교육용 ML 파이프라인 대시보드. 투자 권유가 아닙니다.")
        st.markdown("---")
        st.markdown("**빠른 가이드**")
        st.markdown("1. `src\\run_pipeline.py` 로 데이터/학습/예측")
        st.markdown("2. 이 페이지 새로고침")
        st.markdown("---")
        st.markdown("**데이터**")
        st.text(f"raw exists: {RAW_DATA_PATH.exists()}")
        st.text(f"latest pred: {LATEST_PREDICTION_PATH.exists()}")

    if not prediction:
        st.markdown(
            """
            <div class="hero-wrap" style="margin-bottom:0.5rem">
                <span class="ticker-pill">NYSE · GE</span>
                <h1 class="hero-title">GE Stock Intelligence</h1>
                <p class="hero-sub">예측 JSON이 아직 없습니다. 파이프라인을 먼저 실행하세요.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.warning("예측 결과가 없습니다. `src\\run_pipeline.py`를 먼저 실행하세요.")
        return

    render_hero(prediction)

    col_left, col_right = st.columns([1.45, 1.0], gap="large")
    with col_left:
        with st.container(border=True):
            st.markdown('<h3 class="section-title">실제값 vs 예측값</h3>', unsafe_allow_html=True)
            if PREDICTION_FIGURE_PATH.exists():
                st.image(str(PREDICTION_FIGURE_PATH), width="stretch")
            else:
                st.info("그래프가 없습니다. `src\\train_model.py` 를 실행하세요.")

    with col_right:
        with st.container(border=True):
            st.markdown('<h3 class="section-title">모델 성능 비교</h3>', unsafe_allow_html=True)
            metrics_df = metrics_to_dataframe(metrics)
            if metrics_df.empty:
                st.info("지표가 없습니다.")
            else:
                best_model = metrics_df.sort_values("MAPE (%)").iloc[0]
                render_data_table(metrics_df)
                st.markdown(
                    f"""
                    <div style="margin-top:0.75rem">
                        <span class="insight-pill">MAPE 최적 모델 · {html.escape(str(best_model["Model"]))} · {float(best_model["MAPE (%)"]):.4f}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["가격 추세", "예측 히스토리", "서비스 구조"])
    with tab1:
        with st.container(border=True):
            st.markdown('<h3 class="section-title">GE 종가 추세</h3>', unsafe_allow_html=True)
            if raw_data.empty:
                st.info("원본 주가가 없습니다. `src\\collect_data.py` 또는 파이프라인을 실행하세요.")
            else:
                st.plotly_chart(
                    build_price_figure(raw_data),
                    width="stretch",
                    config={"displayModeBar": False},
                )

    with tab2:
        with st.container(border=True):
            st.markdown('<h3 class="section-title">누적 예측 기록</h3>', unsafe_allow_html=True)
            if history.empty:
                st.info("히스토리가 없습니다.")
            else:
                display_columns = [
                    "generated_at_utc",
                    "last_market_date",
                    "current_close",
                    "predicted_next_close",
                    "predicted_next_return_pct",
                    "model",
                ]
                existing = [c for c in display_columns if c in history.columns]
                render_data_table(history[existing].tail(20))

    with tab3:
        with st.container(border=True):
            st.markdown('<h3 class="section-title">자동화 흐름 (n8n)</h3>', unsafe_allow_html=True)
            st.code(
                (
                    "Yahoo Finance (GE) → collect_data\n"
                    "    → preprocess → train_model → predict\n"
                    "        → data/predictions/latest_prediction.json\n"
                    "            → n8n (스케줄) → Google Sheets + Telegram\n"
                    "            → Streamlit (이 대시보드)\n"
                ),
                language="text",
            )
            st.markdown(
                """
                <div class="notice-box">
                본 UI는 <strong>교육용</strong>입니다. 뉴스·거시경제·이벤트는 반영되지 않을 수 있으며, 표시 수치는 투자 권유가 아닙니다.
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    render_dashboard()
