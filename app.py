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
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        progress=False,
        auto_adjust=auto_adjust,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # まれに MultiIndex で返るケース対策
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    df = df.copy()

    # Closeが無い場合、Adj CloseをCloseとして使う
    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        else:
            return pd.DataFrame()

    # dropna全列は危険なのでCloseだけ
    df = df.dropna(subset=["Close"])
    if df.empty:
        return pd.DataFrame()

    # 欠けやすい列を必ず作る（Closeで補完）
    for col in ["Open", "High", "Low"]:
        if col not in df.columns:
            df[col] = df["Close"]
        else:
            df[col] = df[col].fillna(df["Close"])

    if "Volume" not in df.columns:
        df["Volume"] = 0
    else:
        df["Volume"] = df["Volume"].fillna(0)

    # 型を安定化
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close"]).sort_index()
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
    
def score_series(
    df: pd.DataFrame,
    *,
    jump_days_: int,
    min_avg_value_floor_: float,
) -> pd.Series:
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float)

    jump = (c / c.shift(jump_days_) - 1.0) * 100.0
    max_jump_40 = jump.rolling(40).max().clip(lower=0)

    v_med20 = v.rolling(20).median()
    rvol_med = (v / v_med20).replace([np.inf, -np.inf], np.nan)

    ma25 = c.rolling(25).mean()
    diff_ma25 = ((c - ma25).abs() / ma25 * 100.0).replace([np.inf, -np.inf], np.nan)

    atr5 = compute_atr(df, 5)
    atr20 = compute_atr(df, 20)
    atr_ratio = (atr5 / atr20).replace([np.inf, -np.inf], np.nan)

    high20 = c.rolling(20).max()
    dist_to_high = ((high20 - c) / c * 100.0).replace([np.inf, -np.inf], np.nan)

    avg_val = (c * v).rolling(5).mean() / 1e8

    s_jump = (max_jump_40 / 80.0).clip(upper=3.0)
    s_rvol = (1.0 / rvol_med.clip(lower=0.05)).clip(upper=10.0)
    s_atr  = (1.0 / atr_ratio.clip(lower=0.20)).clip(upper=10.0)
    s_dist = (1.0 / (1.0 + dist_to_high.clip(lower=0.0))).clip(upper=1.0)
    s_diff = (1.0 / (1.0 + diff_ma25.clip(lower=0.0))).clip(upper=1.0)

    ma25_slope = ma25 - ma25.shift(5)
    s_slope = (ma25_slope > 0).astype(float) * 0.15

    score = (
        1.30 * s_jump +
        1.10 * s_rvol +
        1.10 * s_atr +
        0.90 * s_dist +
        0.70 * s_diff +
        s_slope
    )

    # ★バックテストでは “床” を別にする（ここが重要）
    score = score.where(avg_val >= float(min_avg_value_floor_))
    return score



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

    # signal を df の index に合わせる（ここがズレると KeyError の温床）
    signal = signal.reindex(df.index).fillna(False)

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

        rows.append({"date": df.index[i], "base_close": base, "max_up_%": max_up, "max_dd_%": max_dd})

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

     # 旧: 条件ANDの完全一致シグナル
# sig = compute_signal_series(...)

# 新: スコア上位5%の日をシグナル扱い
score = score_series(df)          # 各日のスコア（連続値）
score_valid = score.dropna()      # 有効な日だけで閾値計算

if score_valid.empty:
    # データ不足などでスコアが作れない場合
    sig = pd.Series(False, index=df.index)
else:
    thr = score_valid.quantile(0.95)   # 上位5% = 95パーセンタイル
    sig = (score >= thr).fillna(False) # シグナル（True/False）


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
        trades_out["ticker"] = t
        all_trades.append(trades_out)

    sum_df = pd.DataFrame(summaries)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    return sum_df, trades_df


# =========================
# 5. メイン画面
# =========================
st.title(f"🚀 {target_market}・二段上げ狙い（精度UP版）")
st.caption("第一波（複数日上昇）→枯渇（中央値RVOL）→25MA付近→ATR収縮→高値が近い、で“短期再噴火”候補を優先。")

colA, colB, colC = st.columns([1.1, 1.1, 1.6])
with colA:
    st.write("**スキャン対象**")
    st.write(f"- 市場: {target_market}")
    st.write(f"- 期間: {scan_period} / 1d")

with colB:
    st.write("**主要条件**")
    st.write(f"- 売買代金: {min_avg_value:.2f}億/日以上")
    st.write(f"- 第一波: {jump_days}日で{min_jump:.0f}%以上")
    st.write(f"- 枯渇: RVOL≤{vol_dry_limit:.2f}")

with colC:
    st.write("**トリガー寄せ**")
    st.write(f"- 25MA乖離≤{ma_near_pct:.1f}% / ATR5/20≤{atr_contract_limit:.2f} / 高値距離≤{dist_to_high_limit:.1f}%")
    st.write(f"- 25MA上向き必須: {'ON' if require_ma_up else 'OFF'}")

if st.button("📡 スキャン開始", type="primary"):
    tickers, info_db = load_master_data(target_market)
    if not tickers:
        st.warning("対象銘柄が見つかりませんでした。")
        st.stop()

    results: List[Dict[str, object]] = []
    fail_reasons: Dict[str, int] = {}
    fetch_fail: List[str] = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    total = len(tickers)
    t0 = time.time()

    for i in range(0, total, batch_size):
        batch = tickers[i: i + batch_size]
        status_text.text(f"スキャン中... {i}/{total}")
        progress_bar.progress(min(1.0, i / total))

        try:
            df_batch = yf.download(
                batch,
                period=scan_period,
                interval="1d",
                progress=False,
                group_by="ticker",
                threads=True,
                auto_adjust=use_auto_adjust,
            )
            if not isinstance(df_batch.columns, pd.MultiIndex):
                df_batch = pd.concat({batch[0]: df_batch}, axis=1)

            # MultiIndex: level0=ticker
            tickers_in_batch = set(df_batch.columns.get_level_values(0))

            for t in batch:
                if t not in tickers_in_batch:
                    fetch_fail.append(t)
                    continue

                stock_data = df_batch[t].dropna()
                # 念のため最低列チェック
                need_cols = {"Open", "High", "Low", "Close", "Volume"}
                if stock_data.empty or not need_cols.issubset(set(stock_data.columns)):
                    fetch_fail.append(t)
                    continue

                ok, reason, m = check_strategy(
                    stock_data,
                    min_avg_value_=min_avg_value,
                    jump_days_=jump_days,
                    min_jump_=min_jump,
                    vol_dry_limit_=vol_dry_limit,
                    ma_near_pct_=ma_near_pct,
                    atr_contract_limit_=atr_contract_limit,
                    dist_to_high_limit_=dist_to_high_limit,
                    require_ma_up_=require_ma_up,
                )
                if not ok:
                    fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                    continue

                sc = score_metrics(m)
                results.append(
                    {
                        "ticker": t,
                        "コード": t.replace(".T", ""),
                        "銘柄名": info_db.get(t, "不明"),
                        "スコア": float(sc),
                        "現在値": float(m["price"]),
                        f"第一波({jump_days}日)%": float(m["max_jump"]),
                        "枯渇RVOL(中央値)": float(m["rvol"]),
                        "25MA乖離%": float(m["diff_ma25"]),
                        "ATR5/20": float(m["atr_ratio"]),
                        "高値距離%": float(m["dist_to_high"]),
                        "代金(億円)": float(m["avg_val"]),
                    }
                )
        except Exception:
            fetch_fail.extend(batch)
            continue

    progress_bar.progress(1.0)
    status_text.empty()
    elapsed = time.time() - t0

    # サマリー
    st.subheader("結果サマリー")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ヒット銘柄数", f"{len(results)}")
    c2.metric("対象銘柄数", f"{total}")
    c3.metric("取得失敗数", f"{len(fetch_fail)}")
    c4.metric("処理時間(秒)", f"{elapsed:.1f}")

    if fail_reasons:
        with st.expander("落選理由の内訳（チューニング用）", expanded=False):
            reason_df = pd.DataFrame(
                [{"理由": k, "件数": v} for k, v in sorted(fail_reasons.items(), key=lambda x: -x[1])]
            )
            st.dataframe(reason_df, use_container_width=True, hide_index=True)

    if fetch_fail:
        with st.expander("取得失敗ティッカー（yfinance欠損など）", expanded=False):
            st.write(", ".join(fetch_fail[:300]) + (" ..." if len(fetch_fail) > 300 else ""))

    if not results:
        st.warning("該当銘柄なし。パラメータを緩めてみてください。")
        st.stop()

    # 表示
    st.success(f"🎯 {len(results)} 銘柄が条件に合致しました（スコア順）")
    res_df = pd.DataFrame(results).sort_values("スコア", ascending=False).reset_index(drop=True)

    show_df = res_df.drop(columns=["ticker"]).copy()
    show_df["スコア"] = show_df["スコア"].map(lambda x: f"{x:.3f}")
    show_df["現在値"] = show_df["現在値"].map(lambda x: f"{x:,.1f}")
    show_df[f"第一波({jump_days}日)%"] = show_df[f"第一波({jump_days}日)%"].map(lambda x: f"{x:.1f}%")
    show_df["枯渇RVOL(中央値)"] = show_df["枯渇RVOL(中央値)"].map(lambda x: f"{x:.2f}倍")
    show_df["25MA乖離%"] = show_df["25MA乖離%"].map(lambda x: f"{x:.1f}%")
    show_df["ATR5/20"] = show_df["ATR5/20"].map(lambda x: f"{x:.2f}")
    show_df["高値距離%"] = show_df["高値距離%"].map(lambda x: f"{x:.1f}%")
    show_df["代金(億円)"] = show_df["代金(億円)"].map(lambda x: f"{x:.2f}億円")

    st.dataframe(show_df, use_container_width=True, hide_index=True)

    # =========================
    # 上位だけバックテスト
    # =========================
    if enable_backtest:
        st.subheader("🧪 上位候補のみバックテスト（同条件シグナル→将来k日）")

        top_n = min(int(top_n_bt), len(res_df))
        top_tickers = res_df.head(top_n)["ticker"].tolist()

        st.write(
            f"- 対象：スコア上位 **{top_n}** 銘柄  / 期間：**{bt_period}**  / 先読み：**{bt_horizon}** 日  / 命中：**+{bt_hit_threshold}%** 以上"
        )

        with st.spinner("バックテスト計算中（上位だけ）..."):
            sum_df, trades_df = run_backtest_top(
                top_tickers,
                period=bt_period,
                auto_adjust=use_auto_adjust,
                min_avg_value_=min_avg_value,
                jump_days_=jump_days,
                min_jump_=min_jump,
                vol_dry_limit_=vol_dry_limit,
                ma_near_pct_=ma_near_pct,
                atr_contract_limit_=atr_contract_limit,
                dist_to_high_limit_=dist_to_high_limit,
                require_ma_up_=require_ma_up,
                horizon=int(bt_horizon),
                hit_threshold=float(bt_hit_threshold),
            )

        if sum_df.empty:
            st.warning("バックテストできるデータがありませんでした（取得失敗/データ不足の可能性）。")
        else:
            sum_df = sum_df.copy()
            sum_df["コード"] = sum_df["ticker"].str.replace(".T", "", regex=False)
            sum_df["銘柄名"] = sum_df["ticker"].map(lambda x: info_db.get(x, "不明"))

            hit_col = "hit_rate_%"
            med_col = f"med_max_up_{bt_horizon}d_%"
            sig_col = "signals"
            sum_df = sum_df.sort_values([hit_col, med_col, sig_col], ascending=[False, False, False])

            show_bt = sum_df[
                ["コード", "銘柄名", "signals", "hit_rate_%",
                 f"avg_max_up_{bt_horizon}d_%", f"med_max_up_{bt_horizon}d_%", f"worst_dd_{bt_horizon}d_%"]
            ].copy()

            show_bt["hit_rate_%"] = show_bt["hit_rate_%"].map(lambda x: "-" if pd.isna(x) else f"{x:.1f}%")
            show_bt[f"avg_max_up_{bt_horizon}d_%"] = show_bt[f"avg_max_up_{bt_horizon}d_%"].map(lambda x: "-" if pd.isna(x) else f"{x:.1f}%")
            show_bt[f"med_max_up_{bt_horizon}d_%"] = show_bt[f"med_max_up_{bt_horizon}d_%"].map(lambda x: "-" if pd.isna(x) else f"{x:.1f}%")
            show_bt[f"worst_dd_{bt_horizon}d_%"] = show_bt[f"worst_dd_{bt_horizon}d_%"].map(lambda x: "-" if pd.isna(x) else f"{x:.1f}%")

            st.write("**銘柄別サマリー（条件が出た日の、その後k日内の成績）**")
            st.dataframe(show_bt, use_container_width=True, hide_index=True)

            if not trades_df.empty:
                all_hit = (trades_df["max_up_%"] >= float(bt_hit_threshold)).mean() * 100.0
                all_med = float(np.nanmedian(trades_df["max_up_%"]))
                all_worst = float(np.nanmin(trades_df["max_dd_%"]))

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("全シグナル数", f"{len(trades_df)}")
                m2.metric("全体 命中率", f"{all_hit:.1f}%")
                m3.metric("全体 中央値(MaxUp)", f"{all_med:.1f}%")
                m4.metric("全体 ワーストDD", f"{all_worst:.1f}%")

                with st.expander("全シグナル明細（必要なら）", expanded=False):
                    td = trades_df.copy()
                    td["コード"] = td["ticker"].str.replace(".T", "", regex=False)
                    td["銘柄名"] = td["ticker"].map(lambda x: info_db.get(x, "不明"))
                    td = td[["date", "コード", "銘柄名", "base_close", "max_up_%", "max_dd_%"]]
                    td["base_close"] = td["base_close"].map(lambda x: f"{x:,.1f}")
                    td["max_up_%"] = td["max_up_%"].map(lambda x: f"{x:.1f}%")
                    td["max_dd_%"] = td["max_dd_%"].map(lambda x: f"{x:.1f}%")
                    st.dataframe(td, use_container_width=True, hide_index=True)

    # チャート確認導線
    st.subheader("候補チャート（ワンクリック確認）")
    pick_code = st.selectbox("銘柄を選択", options=res_df["コード"].tolist(), index=0)
    pick_ticker = f"{pick_code}.T"

    try:
        df_one = yf.download(
            pick_ticker,
            period=scan_period,
            interval="1d",
            progress=False,
            auto_adjust=use_auto_adjust,
        ).dropna()

        if len(df_one) >= 10 and "Close" in df_one.columns:
            st.write(f"**{pick_code}：{info_db.get(pick_ticker, '不明')}**")
            st.line_chart(df_one["Close"], height=260)
            if "Volume" in df_one.columns:
                st.bar_chart(df_one["Volume"], height=180)
        else:
            st.info("チャート表示に十分なデータがありません。")
    except Exception as e:
        st.warning(f"チャート取得に失敗: {e}")

else:
    st.info("左の条件を調整して「📡 スキャン開始」を押してください。")
