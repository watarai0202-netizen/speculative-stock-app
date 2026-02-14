import streamlit as st
import pandas as pd
import yfinance as yf
import os
import time
from io import BytesIO
import urllib.request

# =========================
# 1. アプリ設定 & 認証
# =========================
st.set_page_config(page_title="二段上げ狙い・枯渇スキャナー", layout="wide")
MY_PASSWORD = "stock testa"

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
# 2. 設定（サイドバー）
# =========================
st.sidebar.title("⚙️ スクリーニング条件")

GITHUB_CSV_RAW_URL = "https://raw.githubusercontent.com/watarai0202-netizen/stocktest-app-1/main/data_j.csv"

# 不人気除外フィルター
st.sidebar.subheader("🚫 不人気除外設定")
min_avg_value = st.sidebar.slider("最低売買代金(直近5日平均/億円)", 0.1, 10.0, 0.5, step=0.1)

# 戦略フィルター
st.sidebar.subheader("📈 二段上げ・枯渇戦略")
lookback_days = 20 # 1ヶ月の営業日数目安
min_jump_pct = st.sidebar.slider("1. 過去20日の最大上昇率(%)", 10, 30, 15)
vol_dry_ratio = st.sidebar.slider("2. 出来高枯渇度(平均の何倍以下か)", 0.1, 1.0, 0.5)
ma_diff_pct = st.sidebar.slider("3. MA(25日)からの乖離率(±%)", 0.1, 5.0, 2.0)

target_market = st.sidebar.radio("📊 市場", ("グロース", "スタンダード", "プライム"), index=0)

# =========================
# 3. データ処理エンジン
# =========================

@st.cache_data(ttl=3600)
def load_master():
    with urllib.request.urlopen(GITHUB_CSV_RAW_URL) as resp:
        df = pd.read_csv(BytesIO(resp.read()))
    
    # 市場絞り込み
    m_key = f"{target_market}（内国株式）"
    df = df[(df["市場・商品区分"] == m_key) & (df["33業種区分"] != "－")]
    
    tickers = [f"{str(code).split('.')[0]}.T" for code in df["コード"]]
    info = {f"{str(row['コード']).split('.')[0]}.T": row['銘柄名'] for _, row in df.iterrows()}
    return tickers, info

@st.cache_data(ttl=300)
def fetch_data_batch(batch):
    return yf.download(batch, period="3mo", interval="1d", progress=False, group_by="ticker", threads=True)

def check_strategy(data):
    """
    戦略ロジック:
    1. 過去20日以内に15%以上の急騰があるか
    2. 今日の出来高が20日平均の50%以下（枯渇）か
    3. 25日線に近いか
    """
    if len(data) < 25: return False, {}
    
    close = data['Close']
    high = data['High']
    volume = data['Volume']
    
    # A. 急騰履歴の確認 (直近20日の最大1日上昇率)
    daily_ret = close.pct_change()
    max_jump = daily_ret.tail(lookback_days).max() * 100
    
    # B. 出来高の枯渇 (今日の出来高 vs 20日平均)
    avg_vol20 = volume.rolling(20).mean().iloc[-1]
    curr_vol = volume.iloc[-1]
    rvol = curr_vol / avg_vol20 if avg_vol20 > 0 else 99
    
    # C. 25日線との距離
    ma25 = close.rolling(25).mean().iloc[-1]
    curr_price = close.iloc[-1]
    ma_dist = abs(curr_price - ma25) / ma25 * 100
    
    # D. 売買代金 (直近5日平均)
    avg_value = (close * volume).tail(5).mean() / 1e8 # 億円
    
    # 判定
    is_jumped = max_jump >= min_jump_pct
    is_dried = rvol <= vol_dry_ratio
    is_near_ma = ma_dist <= ma_diff_pct
    is_liquid = avg_value >= min_avg_value
    
    details = {
        "最大上昇": max_jump,
        "出来高倍率": rvol,
        "MA乖離": ma_dist,
        "売買代金": avg_value
    }
    
    if is_jumped and is_dried and is_near_ma and is_liquid:
        return True, details
    return False, details

# =========================
# 4. メイン実行
# =========================
st.title(f"🔭 {target_market}・二段上げ候補スキャナー")

if st.button("📡 スキャン開始", type="primary"):
    tickers, info_db = load_master()
    results = []
    
    bar = st.progress(0)
    status = st.empty()
    
    batch_size = 40
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i+batch_size]
        status.text(f"分析中... {i}/{len(tickers)}")
        bar.progress(i / len(tickers))
        
        try:
            df_all = fetch_data_batch(batch)
            if not isinstance(df_all.columns, pd.MultiIndex):
                df_all = pd.concat({batch[0]: df_all}, axis=1)
                
            for t in batch:
                if t not in df_all.columns.levels[0]: continue
                data = df_all[t].dropna()
                
                match, d = check_strategy(data)
                if match:
                    results.append({
                        "コード": t.replace(".T", ""),
                        "銘柄名": info_db.get(t, "不明"),
                        "現在値": f"{data['Close'].iloc[-1]:,.0f}",
                        "最大上昇率": f"{d['最大上昇']:.1f}%",
                        "出来高倍率": f"{d['出来高倍率']:.2f}倍",
                        "MA乖離": f"{d['MA乖離']:.1f}%",
                        "平均代金": f"{d['売買代金']:.2f}億円",
                    })
        except:
            continue

    bar.progress(1.0)
    status.empty()

    if results:
        st.success(f"🎯 期待銘柄が {len(results)} 件見つかりました")
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
    else:
        st.warning("条件に合致する銘柄はありません。条件を少し緩めてみてください。")
