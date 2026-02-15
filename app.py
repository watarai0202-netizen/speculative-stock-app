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
# 2. サイドバー設定
# =========================
st.sidebar.title("⚙️ スキャン条件")

GITHUB_CSV_URL = "https://raw.githubusercontent.com/watarai0202-netizen/stocktest-app-1/main/data_j.csv"
target_market = st.sidebar.radio("📊 市場選択", ("グロース", "スタンダード", "プライム"), index=0)

st.sidebar.subheader("🚫 足切り（ここだけは守る）")
min_avg_value = st.sidebar.slider("最低売買代金(5日平均/億円)", 0.1, 10.0, 0.5, 0.1)
vol_dry_limit = st.sidebar.slider("出来高枯渇度（当日/20日中央値）上限", 0.05, 2.0, 0.70, 0.05)
ma_near_pct = st.sidebar.slider("25日線との乖離(±%) 上限", 0.5, 15.0, 4.0, 0.1)

st.sidebar.subheader("⭐ スコア加点（落とさない・順位付け用）")
jump_days = st.sidebar.selectbox("第一波の累積日数（加点）", [2, 3, 4, 5], index=1)
min_jump = st.sidebar.slider(f"過去40日の最大{jump_days}日上昇率(%)（加点）", 5, 80, 15, 1)
atr_contract_limit = st.sidebar.slider("ATR収縮（ATR5/ATR20）目安（加点）", 0.3, 1.5, 0.85, 0.05)
dist_to_high_limit = st.sidebar.slider("20日高値までの距離(%) 目安（加点）", 0.5, 20.0, 6.0, 0.1)
require_ma_up = st.sidebar.checkbox("25MAが上向き（5日前比+）を加点", value=True)

st.sidebar.subheader("🧪 実行設定")
batch_size = st.sidebar.slider("バッチサイズ（yfinance一括取得）", 10, 100, 50, 5)
use_auto_adjust = st.sidebar.checkbox("価格を調整（auto_adjust=True）", value=True)

scan_period = st.sidebar.selectbox("スキャン用 取得期間（指標に必要）", ["3mo", "6mo", "1y"], index=1)
top_k = st.sidebar.slider("表示件数（上位）", 10, 200, 50, 5)

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


def safe_float(x) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return np.nan
        return v
    except Exception:
        return np.nan


def calc_metrics(data: pd.DataFrame, *, jump_days_: int) -> Dict[str, float]:
    """
    指標計算（最後の1日分）
    """
    c = data["Close"].astype(float)
    v = data["Volume"].astype(float)

    # 流動性
    avg_val = (c * v).tail(5).mean() / 1e8

    # 第一波（過去40日で最大N日上昇率）
    jump_series = (c / c.shift(jump_days_) - 1.0) * 100.0
    max_jump = jump_series.tail(40).max()

    # 枯渇（中央値）
    v_med20 = v.tail(20).median()
    rvol_med = (v.iloc[-1] / v_med20) if v_med20 > 0 else np.nan

    # 25MA乖離
    ma25 = c.rolling(25).mean().iloc[-1]
    curr_p = c.iloc[-1]
    diff_ma25 = abs(curr_p - ma25) / ma25 * 100.0 if ma25 and ma25 > 0 else np.nan

    # ATR収縮
    atr5 = compute_atr(data, 5).iloc[-1]
    atr20 = compute_atr(data, 20).iloc[-1]
    atr_ratio = (atr5 / atr20) if atr20 and atr20 > 0 else np.nan

    # 高値距離（20日）
    high20 = c.tail(20).max()
    dist_to_high = (high20 - curr_p) / curr_p * 100.0 if curr_p and curr_p > 0 else np.nan

    # MA向き
    ma25_now = c.rolling(25).mean().iloc[-1]
    ma25_prev = c.rolling(25).mean().shift(5).iloc[-1]
    ma25_slope = ma25_now - ma25_prev

    return {
        "price": safe_float(curr_p),
        "avg_val": safe_float(avg_val),
        "max_jump": safe_float(max_jump),
        "rvol": safe_float(rvol_med),
        "diff_ma25": safe_float(diff_ma25),
        "atr_ratio": safe_float(atr_ratio),
        "dist_to_high": safe_float(dist_to_high),
        "ma25_slope": safe_float(ma25_slope),
    }


def pass_filter(m: Dict[str, float], *, min_avg_value_: float, vol_dry_limit_: float, ma_near_pct_: float) -> bool:
    """
    足切りは3つだけ（0件問題を潰すため）
    """
    if not np.isfinite(m.get("avg_val", np.nan)) or m["avg_val"] < min_avg_value_:
        return False
    if not np.isfinite(m.get("rvol", np.nan)) or m["rvol"] > vol_dry_limit_:
        return False
    if not np.isfinite(m.get("diff_ma25", np.nan)) or m["diff_ma25"] > ma_near_pct_:
        return False
    return True


def score_metrics(
    m: Dict[str, float],
    *,
    min_jump_: float,
    atr_contract_limit_: float,
    dist_to_high_limit_: float,
    require_ma_up_: bool,
) -> float:
    """
    落とさず順位付けに使うスコア
    - 第一波が強いほど↑
    - 枯渇（rvol）が小さいほど↑
    - ATR収縮（atr_ratio）が小さいほど↑
    - 高値が近いほど↑
    - MAが上向きならボーナス
    """
    max_jump = m.get("max_jump", np.nan)
    rvol = m.get("rvol", np.nan)
    atr_ratio = m.get("atr_ratio", np.nan)
    dist = m.get("dist_to_high", np.nan)
    diff = m.get("diff_ma25", np.nan)
    slope = m.get("ma25_slope", 0.0)

    # 欠損は弱め評価
    max_jump = 0.0 if not np.isfinite(max_jump) else max(0.0, min(max_jump, 200.0))
    rvol = 9.9 if not np.isfinite(rvol) else max(0.05, min(rvol, 9.9))
    atr_ratio = 9.9 if not np.isfinite(atr_ratio) else max(0.20, min(atr_ratio, 9.9))
    dist = 99.0 if not np.isfinite(dist) else max(0.0, min(dist, 99.0))
    diff = 99.0 if not np.isfinite(diff) else max(0.0, min(diff, 99.0))

    # 加点：第一波（min_jumpを基準に「超えた度合い」）
    # 例: min_jump=15なら、15%超えから効く
    s_jump = max(0.0, (max_jump - min_jump_) / 50.0)  # だいたい0〜2
    s_rvol = 1.0 / rvol                               # 枯渇ほど強い
    s_atr = 1.0 / atr_ratio                           # 収縮ほど強い
    s_dist = 1.0 / (1.0 + max(0.0, dist - dist_to_high_limit_))  # 高値が遠いほど減点
    s_diff = 1.0 / (1.0 + diff)                       # MAから離れるほど減点
    s_ma = 0.25 if (require_ma_up_ and slope > 0) else 0.0

    # 直感に寄せた重み
    return (
        1.40 * s_jump +
        1.10 * s_rvol +
        1.00 * s_atr +
        0.80 * s_dist +
        0.60 * s_diff +
        s_ma
    )


# =========================
# 4. メイン画面
# =========================
st.title(f"🚀 {target_market}・二段上げ狙い（実用版）")
st.caption("0件問題を潰して「毎日使える候補リスト」に寄せた版。足切りは3つだけ、あとはスコアで並べる。")

colA, colB = st.columns([1.2, 1.8])
with colA:
    st.write("**足切り（必須）**")
    st.write(f"- 売買代金(5日平均) ≥ {min_avg_value:.2f} 億円")
    st.write(f"- 枯渇RVOL(当日/20日中央値) ≤ {vol_dry_limit:.2f}")
    st.write(f"- 25MA乖離 ≤ {ma_near_pct:.1f}%")

with colB:
    st.write("**スコア加点（順位付け）**")
    st.write(f"- 第一波: {jump_days}日上昇（過去40日max） / 目安 {min_jump:.0f}%")
    st.write(f"- ATR収縮目安: ATR5/ATR20 ≤ {atr_contract_limit:.2f}")
    st.write(f"- 高値距離目安: 20日高値まで ≤ {dist_to_high_limit:.1f}%")
    st.write(f"- MA上向き加点: {'ON' if require_ma_up else 'OFF'}")

if st.button("📡 スキャン開始", type="primary"):
    tickers, info_db = load_master_data(target_market)
    if not tickers:
        st.warning("対象銘柄が見つかりませんでした。")
        st.stop()

    strict_results: List[Dict[str, object]] = []
    all_candidates: List[Dict[str, object]] = []
    fail_counts = {"売買代金": 0, "枯渇": 0, "25MA乖離": 0, "データ不足": 0, "取得失敗": 0}

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
            if df_batch is None or df_batch.empty:
                fail_counts["取得失敗"] += len(batch)
                continue

            if not isinstance(df_batch.columns, pd.MultiIndex):
                # 1銘柄だけのとき
                df_batch = pd.concat({batch[0]: df_batch}, axis=1)

            tickers_in_batch = set(df_batch.columns.get_level_values(0))

            for t in batch:
                if t not in tickers_in_batch:
                    fail_counts["取得失敗"] += 1
                    continue

                stock_data = df_batch[t].dropna()
                need_cols = {"Open", "High", "Low", "Close", "Volume"}
                if stock_data.empty or not need_cols.issubset(set(stock_data.columns)):
                    fail_counts["取得失敗"] += 1
                    continue

                # 指標計算に必要な長さ
                if len(stock_data) < 80:
                    fail_counts["データ不足"] += 1
                    continue

                m = calc_metrics(stock_data, jump_days_=jump_days)

                # 足切り失敗内訳（チューニング用）
                if not (np.isfinite(m["avg_val"]) and m["avg_val"] >= min_avg_value):
                    fail_counts["売買代金"] += 1
                elif not (np.isfinite(m["rvol"]) and m["rvol"] <= vol_dry_limit):
                    fail_counts["枯渇"] += 1
                elif not (np.isfinite(m["diff_ma25"]) and m["diff_ma25"] <= ma_near_pct):
                    fail_counts["25MA乖離"] += 1

                sc = score_metrics(
                    m,
                    min_jump_=min_jump,
                    atr_contract_limit_=atr_contract_limit,
                    dist_to_high_limit_=dist_to_high_limit,
                    require_ma_up_=require_ma_up,
                )

                row = {
                    "ticker": t,
                    "コード": t.replace(".T", ""),
                    "銘柄名": info_db.get(t, "不明"),
                    "スコア": float(sc),
                    "現在値": float(m["price"]) if np.isfinite(m["price"]) else np.nan,
                    f"第一波({jump_days}日)%": float(m["max_jump"]) if np.isfinite(m["max_jump"]) else np.nan,
                    "枯渇RVOL(中央値)": float(m["rvol"]) if np.isfinite(m["rvol"]) else np.nan,
                    "25MA乖離%": float(m["diff_ma25"]) if np.isfinite(m["diff_ma25"]) else np.nan,
                    "ATR5/20": float(m["atr_ratio"]) if np.isfinite(m["atr_ratio"]) else np.nan,
                    "高値距離%": float(m["dist_to_high"]) if np.isfinite(m["dist_to_high"]) else np.nan,
                    "代金(億円)": float(m["avg_val"]) if np.isfinite(m["avg_val"]) else np.nan,
                    "MA傾き(参考)": float(m["ma25_slope"]) if np.isfinite(m["ma25_slope"]) else np.nan,
                }
                all_candidates.append(row)

                if pass_filter(m, min_avg_value_=min_avg_value, vol_dry_limit_=vol_dry_limit, ma_near_pct_=ma_near_pct):
                    strict_results.append(row)

        except Exception:
            fail_counts["取得失敗"] += len(batch)
            continue

    progress_bar.progress(1.0)
    status_text.empty()
    elapsed = time.time() - t0

    # サマリー
    st.subheader("結果サマリー")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ヒット銘柄数（足切り通過）", f"{len(strict_results)}")
    c2.metric("対象銘柄数", f"{total}")
    c3.metric("処理時間(秒)", f"{elapsed:.1f}")
    c4.metric("取得失敗（概算）", f"{fail_counts['取得失敗']}")

    with st.expander("落選内訳（足切り3条件）", expanded=False):
        st.write(pd.DataFrame([{"理由": k, "件数": v} for k, v in fail_counts.items()]))

    # 表示データ作成
    def format_table(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["スコア"] = out["スコア"].map(lambda x: f"{x:.3f}")
        out["現在値"] = out["現在値"].map(lambda x: "-" if pd.isna(x) else f"{x:,.1f}")
        out[f"第一波({jump_days}日)%"] = out[f"第一波({jump_days}日)%"].map(lambda x: "-" if pd.isna(x) else f"{x:.1f}%")
        out["枯渇RVOL(中央値)"] = out["枯渇RVOL(中央値)"].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}倍")
        out["25MA乖離%"] = out["25MA乖離%"].map(lambda x: "-" if pd.isna(x) else f"{x:.1f}%")
        out["ATR5/20"] = out["ATR5/20"].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}")
        out["高値距離%"] = out["高値距離%"].map(lambda x: "-" if pd.isna(x) else f"{x:.1f}%")
        out["代金(億円)"] = out["代金(億円)"].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}億円")
        out["MA傾き(参考)"] = out["MA傾き(参考)"].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}")
        return out

    # まず足切り通過があるならそれを上位表示
    if strict_results:
        st.success("✅ 足切り通過銘柄をスコア順で表示（まずここを見る）")
        res_df = pd.DataFrame(strict_results).sort_values("スコア", ascending=False).head(top_k).reset_index(drop=True)
        show_df = format_table(res_df.drop(columns=["ticker"]))
        st.dataframe(show_df, use_container_width=True, hide_index=True)
        candidate_df = res_df
    else:
        # 0件のとき：自動で緩めて候補を出す（同じ取得データのまま）
        st.warning("⚠️ 足切り通過が0件。自動的に条件を“段階的に”緩めて候補を出します。")

        if not all_candidates:
            st.error("候補生成に必要なデータが取れていません。取得期間を6mo/1yにして再実行してください。")
            st.stop()

        base = pd.DataFrame(all_candidates).dropna(subset=["代金(億円)", "枯渇RVOL(中央値)", "25MA乖離%"], how="any").copy()
        if base.empty:
            st.error("指標計算が成立する銘柄がありませんでした。取得期間を1yにして再実行してください。")
            st.stop()

        # 段階的に緩める（3段）
        relax_steps = [
            (min_avg_value, vol_dry_limit, ma_near_pct, "元の条件"),
            (min_avg_value * 0.8, min(vol_dry_limit * 1.3, 2.0), min(ma_near_pct * 1.3, 15.0), "少し緩め"),
            (min_avg_value * 0.6, min(vol_dry_limit * 1.6, 2.0), min(ma_near_pct * 1.6, 15.0), "さらに緩め"),
        ]

        picked = None
        picked_label = ""
        for mv, vd, mp, label in relax_steps:
            cond = (base["代金(億円)"] >= mv) & (base["枯渇RVOL(中央値)"] <= vd) & (base["25MA乖離%"] <= mp)
            df_try = base.loc[cond].copy()
            if len(df_try) >= 10:
                picked = df_try
                picked_label = f"{label}（売買代金≥{mv:.2f} / RVOL≤{vd:.2f} / 乖離≤{mp:.1f}%）"
                break

        if picked is None:
            # それでも少なければ：流動性だけは守って、スコア上位を出す
            mv = min_avg_value * 0.6
            picked = base.loc[base["代金(億円)"] >= mv].copy()
            picked_label = f"最終救済（売買代金≥{mv:.2f}のみで抽出 → スコア上位）"

        st.info(f"表示ルール：{picked_label}")
        candidate_df = picked.sort_values("スコア", ascending=False).head(top_k).reset_index(drop=True)
        show_df = format_table(candidate_df.drop(columns=["ticker"]))
        st.dataframe(show_df, use_container_width=True, hide_index=True)

   
