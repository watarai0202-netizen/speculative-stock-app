# app.py
from __future__ import annotations

import time
import urllib.request
from io import BytesIO
from typing import Dict, List, Tuple

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
# ① 第一波（複数日）上昇：3日累積最大上昇率
jump_days = st.sidebar.selectbox("1. 第一波の累積日数", [2, 3, 4, 5], index=1)
min_jump = st.sidebar.slider(f"2. 過去40日の最大{jump_days}日上昇率(%)", 10, 80, 20, 1)

# ② 枯渇：中央値ベースRVOL
vol_dry_limit = st.sidebar.slider("3. 出来高枯渇度（当日/20日中央値）上限", 0.05, 1.5, 0.55, 0.05)

# ③ 25MA 乖離
ma_near_pct = st.sidebar.slider("4. 25日線との乖離(±%)", 0.5, 10.0, 2.0, 0.1)

# ④ 溜め：ATR収縮
atr_contract_limit = st.sidebar.slider("5. ATR収縮（ATR5/ATR20）上限", 0.3, 1.2, 0.75, 0.05)

# ⑤ 仕掛けが近い：20日高値までの距離
dist_to_high_limit = st.sidebar.slider("6. 20日高値までの距離(%) 上限", 0.5, 10.0, 3.0, 0.1)

# ⑥ MAの向き
require_ma_up = st.sidebar.checkbox("7. 25MAが上向き（5日前比+）を必須", value=True)

st.sidebar.subheader("🧪 実行設定")
batch_size = st.sidebar.slider("バッチサイズ（yfinance一括取得）", 10, 100, 50, 5)
use_auto_adjust = st.sidebar.checkbox("価格を調整（auto_adjust=True）", value=True)
scan_period = st.sidebar.selectbox("取得期間", ["3mo", "6mo", "1y"], index=1)

# キャッシュクリア（Streamlit側のcache_data）
if st.sidebar.button("🔄 キャッシュクリア"):
    st.cache_data.clear()
    st.rerun()

# =========================
# 3. データ読み込み
# =========================
@st.cache_data(ttl=3600)
def load_master_data(market_name: str) -> Tuple[List[str], Dict[str, str]]:
    """市場ごとに銘柄リストを読み込む（TSE CSV）"""
    with urllib.request.urlopen(GITHUB_CSV_URL) as resp:
        df = pd.read_csv(BytesIO(resp.read()))

    m_key = f"{market_name}（内国株式）"
    df_filtered = df[(df["市場・商品区分"] == m_key) & (df["33業種区分"] != "－")].copy()

    tickers = [f"{str(code).split('.')[0]}.T" for code in df_filtered["コード"]]
    info = {f"{str(row['コード']).split('.')[0]}.T": row["銘柄名"] for _, row in df_filtered.iterrows()}
    return tickers, info


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """ATR（単純移動平均）"""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
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
    """
    精度UP版「二段上げ」判定 + スコア用メトリクス返却
    """
    need_len = max(60, 25 + 5, 20 + jump_days_)
    if len(data) < need_len:
        return False, "データ不足", {}

    c = data["Close"].astype(float)
    v = data["Volume"].astype(float)

    # 0) 売買代金（直近5日平均/億円）
    avg_val = (c * v).tail(5).mean() / 1e8
    if avg_val < min_avg_value_:
        return False, "売買代金不足", {"avg_val": float(avg_val)}

    # 1) 第一波：過去40日で最大「jump_days日」上昇率
    #    例）3日上昇率 = Close / Close.shift(3) - 1
    jump_series = (c / c.shift(jump_days_) - 1.0) * 100.0
    max_jump = jump_series.tail(40).max()
    if pd.isna(max_jump) or max_jump < min_jump_:
        return False, "第一波弱い", {"max_jump": float(max_jump) if pd.notna(max_jump) else 0.0}

    # 2) 出来高枯渇：当日 / 20日中央値（異常日耐性）
    v_med20 = v.tail(20).median()
    rvol_med = (v.iloc[-1] / v_med20) if v_med20 > 0 else 9.9
    if rvol_med > vol_dry_limit_:
        return False, "枯渇してない", {"rvol": float(rvol_med)}

    # 3) 25MA 乖離
    ma25 = c.rolling(25).mean().iloc[-1]
    curr_p = c.iloc[-1]
    diff_ma25 = abs(curr_p - ma25) / ma25 * 100.0
    if diff_ma25 > ma_near_pct_:
        return False, "25MA乖離大", {"diff_ma25": float(diff_ma25)}

    # 4) 溜め：ATR収縮（ATR5/ATR20）
    atr5 = compute_atr(data, 5).iloc[-1]
    atr20 = compute_atr(data, 20).iloc[-1]
    atr_ratio = (atr5 / atr20) if atr20 and atr20 > 0 else 9.9
    if atr_ratio > atr_contract_limit_:
        return False, "ボラ収縮弱い", {"atr_ratio": float(atr_ratio)}

    # 5) 仕掛けが近い：20日高値までの距離
    high20 = c.tail(20).max()
    dist_to_high = (high20 - curr_p) / curr_p * 100.0
    if dist_to_high > dist_to_high_limit_:
        return False, "高値まで遠い", {"dist_to_high": float(dist_to_high)}

    # 6) 25MAの向き
    ma25_slope = (c.rolling(25).mean().iloc[-1]) - (c.rolling(25).mean().shift(5).iloc[-1])
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
    """
    スコア（上位候補を並べるため）
    - 第一波強いほど +（max_jump）
    - 枯渇してるほど +（rvolが低い）
    - ボラ収縮ほど +（atr_ratioが低い）
    - 高値が近いほど +（dist_to_highが低い）
    - 25MA乖離は小さいほど +
    - MA上向きは少し加点
    """
    # 乱暴にクリップして安定化（極端値の影響を抑える）
    max_jump = max(0.0, min(m.get("max_jump", 0.0), 200.0))
    rvol = max(0.01, min(m.get("rvol", 9.9), 9.9))
    atr_ratio = max(0.01, min(m.get("atr_ratio", 9.9), 9.9))
    dist = max(0.0, min(m.get("dist_to_high", 99.0), 99.0))
    diff = max(0.0, min(m.get("diff_ma25", 99.0), 99.0))
    slope = m.get("ma25_slope", 0.0)

    # 0〜1っぽい指標に寄せる（簡易）
    s_jump = max_jump / 80.0  # 80%で1付近
    s_rvol = 1.0 / rvol       # 小さいほど高得点
    s_atr = 1.0 / atr_ratio
    s_dist = 1.0 / (1.0 + dist)
    s_diff = 1.0 / (1.0 + diff)
    s_slope = 0.15 if slope > 0 else 0.0

    # ウェイト（好みに合わせて調整してOK）
    return (
        1.30 * s_jump +
        1.10 * s_rvol +
        1.10 * s_atr +
        0.90 * s_dist +
        0.70 * s_diff +
        s_slope
    )


# =========================
# 4. メイン画面
# =========================
st.title(f"🚀 {target_market}・二段上げ狙い（精度UP版）")
st.caption("第一波（複数日上昇）→枯渇（中央値RVOL）→25MA付近→ATR収縮→高値が近い、で“明日〜数日”寄せ。")

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

# 実行
if st.button("📡 スキャン開始", type="primary"):
    try:
        tickers, info_db = load_master_data(target_market)
    except Exception as e:
        st.error(f"マスター読み込み失敗: {e}")
        st.stop()

    if not tickers:
        st.warning("対象銘柄が見つかりませんでした。")
        st.stop()

    results: List[Dict[str, object]] = []
    fail_reasons: Dict[str, int] = {}
    fetch_fail: List[str] = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    # yfinanceで一括取得
    total = len(tickers)
    t0 = time.time()

    for i in range(0, total, batch_size):
        batch = tickers[i : i + batch_size]
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

            # 1銘柄のみの構造補正
            if not isinstance(df_batch.columns, pd.MultiIndex):
                df_batch = pd.concat({batch[0]: df_batch}, axis=1)

            for t in batch:
                # 取得漏れ
                if t not in df_batch.columns.levels[0]:
                    fetch_fail.append(t)
                    continue

                stock_data = df_batch[t].dropna()
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
            # バッチ単位で落ちた場合は全部取得失敗扱い
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

    # メイン結果
    if not results:
        st.warning("該当銘柄なし。パラメータを緩めてみてください。")
        st.stop()

    st.success(f"🎯 {len(results)} 銘柄が条件に合致しました（スコア順）")

    res_df = pd.DataFrame(results).sort_values("スコア", ascending=False).reset_index(drop=True)

    # 表示用フォーマット（表示は綺麗に、ソートは数値のまま）
    show_df = res_df.copy()
    show_df["スコア"] = show_df["スコア"].map(lambda x: f"{x:.3f}")
    show_df["現在値"] = show_df["現在値"].map(lambda x: f"{x:,.1f}")
    show_df[f"第一波({jump_days}日)%"] = show_df[f"第一波({jump_days}日)%"].map(lambda x: f"{x:.1f}%")
    show_df["枯渇RVOL(中央値)"] = show_df["枯渇RVOL(中央値)"].map(lambda x: f"{x:.2f}倍")
    show_df["25MA乖離%"] = show_df["25MA乖離%"].map(lambda x: f"{x:.1f}%")
    show_df["ATR5/20"] = show_df["ATR5/20"].map(lambda x: f"{x:.2f}")
    show_df["高値距離%"] = show_df["高値距離%"].map(lambda x: f"{x:.1f}%")
    show_df["代金(億円)"] = show_df["代金(億円)"].map(lambda x: f"{x:.2f}億円")

    st.dataframe(show_df, use_container_width=True, hide_index=True)

    # チャート確認導線
    st.subheader("候補チャート（ワンクリック確認）")
    pick_code = st.selectbox(
        "銘柄を選択",
        options=res_df["コード"].tolist(),
        index=0,
    )
    pick_ticker = f"{pick_code}.T"

    # 選択銘柄を取得して表示（軽量に直近6mo固定でもOKだが、ここはscan_periodに合わせる）
    try:
        df_one = yf.download(
            pick_ticker,
            period=scan_period,
            interval="1d",
            progress=False,
            auto_adjust=use_auto_adjust,
        ).dropna()

        if len(df_one) >= 10:
            st.write(f"**{pick_code}：{info_db.get(pick_ticker, '不明')}**")
            st.line_chart(df_one["Close"], height=260)
            st.bar_chart(df_one["Volume"], height=180)
        else:
            st.info("チャート表示に十分なデータがありません。")
    except Exception as e:
        st.warning(f"チャート取得に失敗: {e}")

    # 参考リンク（TradingView / Kabutan等は必要に応じて好みで）
    with st.expander("外部リンク（確認用）", expanded=False):
        st.write(f"- TradingView: https://www.tradingview.com/symbols/TSE-{pick_code}/")
        st.write(f"- 株探: https://kabutan.jp/stock/?code={pick_code}")

else:
    st.info("左の条件を調整して「📡 スキャン開始」を押してください。")
