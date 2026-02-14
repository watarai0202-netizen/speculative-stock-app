# app.py
from __future__ import annotations

import time
import urllib.request
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# =========================
# 1. アプリ設定 & 認証
# =========================
st.set_page_config(page_title="二段上げスキャナー", layout="wide")
MY_PASSWORD = "stock testa"  # ※要望により直書きのまま

if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.title("🔒 認証")
    pwd = st.text_input("パスワード", type="password")
    if pwd == MY_PASSWORD:
        st.session_state.auth = True
        st.rerun()
    st.stop()

# =========================
# 2. サイドバー設定（条件指定）
# =========================
st.sidebar.title("⚙️ スキャン条件")

GITHUB_CSV_URL = "https://raw.githubusercontent.com/watarai0202-netizen/stocktest-app-1/main/data_j.csv"

target_market = st.sidebar.radio("📊 市場選択", ("グロース", "スタンダード", "プライム"), index=0)

st.sidebar.subheader("🚫 不人気フィルター")
min_avg_value = st.sidebar.slider("最低売買代金(5日平均/億円)", 0.1, 10.0, 0.5, 0.1)

st.sidebar.subheader("📈 二段上げパラメータ（精度UP版）")
jump_days = st.sidebar.selectbox("1. 第一波の累積日数", [2, 3, 4, 5], index=1)
min_jump = st.sidebar.slider(f"2. 過去40日の最大{jump_days}日上昇率(%)", 10, 80, 20, 1)
vol_dry_limit = st.sidebar.slider("3. 出来高枯渇度（当日/20日中央値）上限", 0.05, 1.5, 0.55, 0.05)
ma_near_pct = st.sidebar.slider("4. 25日線との乖離(±%)", 0.5, 10.0, 2.0, 0.1)
atr_contract_limit = st.sidebar.slider("5. ATR収縮（ATR5/ATR20）上限", 0.3, 1.2, 0.75, 0.05)
dist_to_high_limit = st.sidebar.slider("6. 20日高値までの距離(%) 上限", 0.5, 10.0, 3.0, 0.1)
require_ma_up = st.sidebar.checkbox("7. 25MAが上向き（5日前比+）を必須", value=True)

st.sidebar.subheader("🧪 実行設定")
batch_size = st.sidebar.slider("バッチサイズ（yfinance一括取得）", 10, 100, 50, 5)
use_auto_adjust = st.sidebar.checkbox("価格を調整（auto_adjust=True）", value=True)
scan_period = st.sidebar.selectbox("スキャン用 取得期間", ["3mo", "6mo", "1y"], index=1)

st.sidebar.subheader("🧪 上位だけバックテスト")
enable_backtest = st.sidebar.checkbox("上位候補のみバックテストする", value=True)
top_n_bt = st.sidebar.slider("バックテスト対象（スコア上位N）", 1, 80, 20, 1)
bt_period = st.sidebar.selectbox("バックテスト期間", ["6mo", "1y", "2y", "5y"], index=2)
bt_horizon = st.sidebar.selectbox("将来の評価期間（k営業日）", [3, 5, 10, 15, 20], index=1)
bt_hit_threshold = st.sidebar.slider("命中判定（k日内 最大上昇が +何% 以上）", 3, 40, 10, 1)

# キャッシュクリア
if st.sidebar.button("🔄 キャッシュクリア"):
    st.cache_data.clear()
    st.rerun()

# =========================
# 3. データ読み込み
# =========================
@st.cache_data(ttl=3600)
def load_master_data(market_name: str) -> Tuple[List[str], Dict[str, str]]:
    with urllib.request.urlopen(GITHUB_CSV_URL) as resp:
        df = pd.read_csv(BytesIO(resp.read()))

    m_key = f"{market_name}（内国株式）"
    df_filtered = df[(df["市場・商品区分"] == m_key) & (df["33業種区分"] != "－")].copy()

    tickers = [f"{str(code).split('.')[0]}.T" for code in df_filtered["コード"]]
    info = {f"{str(row['コード']).split('.')[0]}.T": row["銘柄名"] for _, row in df_filtered.iterrows()}
    return tickers, info


def fetch_ohlcv(ticker: str, period: str, auto_adjust: bool) -> pd.DataFrame:
    """
    yfinanceがHigh/Lowを返さない・NaNが混ざるケースでも
    可能な限りバックテスト可能なOHLCVに整形する。
    """
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        progress=False,
        auto_adjust=auto_adjust,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Closeが無いと何もできない
    if "Close" not in df.columns:
        return pd.DataFrame()

    # dropna() を全列対象にすると全消しになりやすいので Close だけに限定
    df = df.dropna(subset=["Close"]).copy()
    if df.empty:
        return pd.DataFrame()

    # 欠けがちな列は Close で補完（検証不能を避ける）
    for col in ["Open", "High", "Low"]:
        if col not in df.columns:
            df[col] = df["Close"]
        else:
            df[col] = df[col].fillna(df["Close"])

    if "Volume" not in df.columns:
        df["Volume"] = 0
    else:
        df["Volume"] = df["Volume"].fillna(0)

    # 型の安定化
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"]).copy()

    df = df.sort_index()
    return df


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    return tr.rolling(period).mean()


def check_strategy(
    data: pd.DataFrame,
    *,
    min_avg_value_: float,
    jump_days_: int,
    min_jump_: float,
    vol_dry_limit_: float,
    ma_near_pct_: float,
    atr_contract_limit_: float,
    dist_to_high_limit_: float,
    require_ma_up_: bool,
) -> Tuple[bool, str, Dict[str, float]]:
    need_len = max(60, 25 + 5, 20 + jump_days_)
    if len(data) < need_len:
        return False, "データ不足", {}

    c = data["Close"].astype(float)
    v = data["Volume"].astype(float)

    # 売買代金（直近5日平均/億円）
    avg_val = (c * v).tail(5).mean() / 1e8
    if avg_val < min_avg_value_:
        return False, "売買代金不足", {"avg_val": float(avg_val)}

    # 第一波：過去40日で最大N日上昇率
    jump_series = (c / c.shift(jump_days_) - 1.0) * 100.0
    max_jump = jump_series.tail(40).max()
    if pd.isna(max_jump) or max_jump < min_jump_:
        return False, "第一波弱い", {"max_jump": float(max_jump) if pd.notna(max_jump) else 0.0}

    # 枯渇：中央値RVOL
    v_med20 = v.tail(20).median()
    rvol_med = (v.iloc[-1] / v_med20) if v_med20 > 0 else 9.9
    if rvol_med > vol_dry_limit_:
        return False, "枯渇してない", {"rvol": float(rvol_med)}

    # 25MA乖離
    ma25 = c.rolling(25).mean().iloc[-1]
    curr_p = c.iloc[-1]
    diff_ma25 = abs(curr_p - ma25) / ma25 * 100.0
    if diff_ma25 > ma_near_pct_:
        return False, "25MA乖離大", {"diff_ma25": float(diff_ma25)}

    # ATR収縮
    atr5 = compute_atr(data, 5).iloc[-1]
    atr20 = compute_atr(data, 20).iloc[-1]
    atr_ratio = (atr5 / atr20) if atr20 and atr20 > 0 else 9.9
    if atr_ratio > atr_contract_limit_:
        return False, "ボラ収縮弱い", {"atr_ratio": float(atr_ratio)}

    # 高値距離（20日）
    high20 = c.tail(20).max()
    dist_to_high = (high20 - curr_p) / curr_p * 100.0
    if dist_to_high > dist_to_high_limit_:
        return False, "高値まで遠い", {"dist_to_high": float(dist_to_high)}

    # MAの向き
    ma25_now = c.rolling(25).mean().iloc[-1]
    ma25_prev = c.rolling(25).mean().shift(5).iloc[-1]
    ma25_slope = ma25_now - ma25_prev
    if require_ma_up_ and not (ma25_slope > 0):
        return False, "MA下向き", {"ma25_slope": float(ma25_slope)}

    metrics = {
        "avg_val": float(avg_val),
        "max_jump": float(max_jump),
        "rvol": float(rvol_med),
        "diff_ma25": float(diff_ma25),
        "atr_ratio": float(atr_ratio),
        "dist_to_high": float(dist_to_high),
        "ma25_slope": float(ma25_slope),
        "price": float(curr_p),
    }
    return True, "OK", metrics


def score_metrics(m: Dict[str, float]) -> float:
    max_jump = max(0.0, min(m.get("max_jump", 0.0), 200.0))
    rvol = max(0.01, min(m.get("rvol", 9.9), 9.9))
    atr_ratio = max(0.01, min(m.get("atr_ratio", 9.9), 9.9))
    dist = max(0.0, min(m.get("dist_to_high", 99.0), 99.0))
    diff = max(0.0, min(m.get("diff_ma25", 99.0), 99.0))
    slope = m.get("ma25_slope", 0.0)

    s_jump = max_jump / 80.0
    s_rvol = 1.0 / rvol
    s_atr = 1.0 / atr_ratio
    s_dist = 1.0 / (1.0 + dist)
    s_diff = 1.0 / (1.0 + diff)
    s_slope = 0.15 if slope > 0 else 0.0

    return (
        1.30 * s_jump +
        1.10 * s_rvol +
        1.10 * s_atr +
        0.90 * s_dist +
        0.70 * s_diff +
        s_slope
    )


# =========================
# 4. バックテスト（上位だけ）
# =========================
def compute_signal_series(
    df: pd.DataFrame,
    *,
    min_avg_value_: float,
    jump_days_: int,
    min_jump_: float,
    vol_dry_limit_: float,
    ma_near_pct_: float,
    atr_contract_limit_: float,
    dist_to_high_limit_: float,
    require_ma_up_: bool,
) -> pd.Series:
    df = df.copy()
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float)

    avg_val = (c * v).rolling(5).mean() / 1e8

    jump = (c / c.shift(jump_days_) - 1.0) * 100.0
    max_jump_40 = jump.rolling(40).max()

    v_med20 = v.rolling(20).median()
    rvol_med = v / v_med20

    ma25 = c.rolling(25).mean()
    diff_ma25 = (c - ma25).abs() / ma25 * 100.0

    atr5 = compute_atr(df, 5)
    atr20 = compute_atr(df, 20)
    atr_ratio = atr5 / atr20

    high20 = c.rolling(20).max()
    dist_to_high = (high20 - c) / c * 100.0

    ma25_slope = ma25 - ma25.shift(5)

    cond = (
        (avg_val >= min_avg_value_) &
        (max_jump_40 >= min_jump_) &
        (rvol_med <= vol_dry_limit_) &
        (diff_ma25 <= ma_near_pct_) &
        (atr_ratio <= atr_contract_limit_) &
        (dist_to_high <= dist_to_high_limit_)
    )
    if require_ma_up_:
        cond = cond & (ma25_slope > 0)

    return cond.fillna(False)


def backtest_one(df: pd.DataFrame, signal: pd.Series, horizon: int) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[signal.index]

    c = df["Close"].astype(float).to_numpy()
    h = df["High"].astype(float).to_numpy()
    l = df["Low"].astype(float).to_numpy()
    sig = signal.to_numpy()

    idxs = np.where(sig)[0]
    rows = []
    n = len(df)

    for i in idxs:
        if i + 1 >= n:
            continue
        end = min(n, i + 1 + horizon)

        base = c[i]
        if not np.isfinite(base) or base <= 0:
            continue

        max_high = np.nanmax(h[i + 1:end])
        min_low = np.nanmin(l[i + 1:end])

        max_up = (max_high / base - 1.0) * 100.0 if np.isfinite(max_high) else np.nan
        max_dd = (min_low / base - 1.0) * 100.0 if np.isfinite(min_low) else np.nan

        rows.append(
            {
                "date": df.index[i],
                "base_close": base,
                "max_up_%": max_up,
                "max_dd_%": max_dd,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def run_backtest_top(
    tickers: List[str],
    *,
    period: str,
    auto_adjust: bool,
    min_avg_value_: float,
    jump_days_: int,
    min_jump_: float,
    vol_dry_limit_: float,
    ma_near_pct_: float,
    atr_contract_limit_: float,
    dist_to_high_limit_: float,
    require_ma_up_: bool,
    horizon: int,
    hit_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    all_trades = []

    for t in tickers:
        df = fetch_ohlcv(t, period=period, auto_adjust=auto_adjust)
        if df.empty:
            summaries.append(
                {
                    "ticker": t,
                    "signals": 0,
                    "hit_rate_%": np.nan,
                    f"avg_max_up_{horizon}d_%": np.nan,
                    f"med_max_up_{horizon}d_%": np.nan,
                    f"worst_dd_{horizon}d_%": np.nan,
                }
            )
            continue

        sig = compute_signal_series(
            df,
            min_avg_value_=min_avg_value_,
            jump_days_=jump_days_,
            min_jump_=min_jump_,
            vol_dry_limit_=vol_dry_limit_,
            ma_near_pct_=ma_near_pct_,
            atr_contract_limit_=atr_contract_limit_,
            dist_to_high_limit_=dist_to_high_limit_,
            require_ma_up_=require_ma_up_,
        )

        trades = backtest_one(df, sig, horizon=horizon)
        if trades.empty:
            summaries.append(
                {
                    "ticker": t,
                    "signals": 0,
                    "hit_rate_%": np.nan,
                    f"avg_max_up_{horizon}d_%": np.nan,
                    f"med_max_up_{horizon}d_%": np.nan,
                    f"worst_dd_{horizon}d_%": np.nan,
                }
            )
            continue

        hit = (trades["max_up_%"] >= hit_threshold).mean() * 100.0
        avg_up = float(np.nanmean(trades["max_up_%"]))
        med_up = float(np.nanmedian(trades["max_up_%"]))
        worst_dd = float(np.nanmin(trades["max_dd_%"]))

        summaries.append(
            {
                "ticker": t,
                "signals": int(len(trades)),
                "hit_rate_%": float(hit),
                f"avg_max_up_{horizon}d_%": avg_up,
                f"med_max_up_{horizon}d_%": med_up,
                f"worst_dd_{horizon}d_%": worst_dd,
            }
        )

        trades_out = trades.copy()
        trades_out["ticker"] =_
