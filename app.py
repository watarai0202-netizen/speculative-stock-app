import streamlit as st
import pandas as pd
import yfinance as yf
import urllib.request
from io import BytesIO
import time

# =========================
# 1. アプリ設定 & 認証
# =========================
st.set_page_config(page_title="二段上げスキャナー", layout="wide")
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
# 2. サイドバー設定（条件指定）
# =========================
st.sidebar.title("⚙️ スキャン条件")

GITHUB_CSV_URL = "https://raw.githubusercontent.com/watarai0202-netizen/stocktest-app-1/main/data_j.csv"

target_market = st.sidebar.radio("📊 市場選択", ("グロース", "スタンダード", "プライム"), index=0)

st.sidebar.subheader("🚫 不人気フィルター")
min_avg_value = st.sidebar.slider("最低売買代金(5日平均/億円)", 0.1, 5.0, 0.5)

st.sidebar.subheader("📈 戦略パラメータ")
min_jump = st.sidebar.slider("1. 過去20日の最大上昇率(%)", 10, 30, 15)
vol_dry_limit = st.sidebar.slider("2. 出来高枯渇度(平均の何倍か)", 0.1, 1.0, 0.5)
ma_near_pct = st.sidebar.slider("3. 25日線との乖離(±%)", 0.5, 5.0, 2.0)

# =========================
# 3. ロジック関数
# =========================

@st.cache_data(ttl=3600)
def load_master_data(market_name):
    """市場ごとにキャッシュを分けて銘柄リストを読み込む"""
    try:
        with urllib.request.urlopen(GITHUB_CSV_URL) as resp:
            df = pd.read_csv(BytesIO(resp.read()))
        
        # 市場・商品区分でフィルタリング
        m_key = f"{market_name}（内国株式）"
        df_filtered = df[(df["市場・商品区分"] == m_key) & (df["33業種区分"] != "－")]
        
        tickers = [f"{str(code).split('.')[0]}.T" for code in df_filtered["コード"]]
        info = {f"{str(row['コード']).split('.')[0]}.T": row['銘柄名'] for _, row in df_filtered.iterrows()}
        return tickers, info
    except Exception as e:
        st.error(f"マスター読み込み失敗: {e}")
        return [], {}

def check_strategy(data):
    """
    【戦略】
    - 過去20日以内に爆上がり(min_jump以上)がある
    - 今日の出来高が20日平均の vol_dry_limit 以下
    - 価格が25日移動平均線の ma_near_pct 以内
    - 売買代金が min_avg_value 以上
    """
    if len(data) < 25:
        return False, {}

    c = data['Close']
    v = data['Volume']
    
    # 売買代金（直近5日平均/億円）
    avg_val = (c * v).tail(5).mean() / 1e8
    if avg_val < min_avg_value:
        return False, {}

    # 1. 過去20日の最大上昇率
    max_jump_found = c.pct_change().tail(20).max() * 100
    if max_jump_found < min_jump:
        return False, {}

    # 2. 出来高枯渇
    avg_v20 = v.rolling(20).mean().iloc[-1]
    rvol = v.iloc[-1] / avg_v20 if avg_v20 > 0 else 9.9
    if rvol > vol_dry_limit:
        return False, {}

    # 3. MA乖離
    ma25 = c.rolling(25).mean().iloc[-1]
    curr_p = c.iloc[-1]
    diff = abs(curr_p - ma25) / ma25 * 100
    if diff > ma_near_pct:
        return False, {}

    return True, {
        "最大上昇": max_jump_found,
        "枯渇度": rvol,
        "乖離率": diff,
        "代金": avg_val
    }

# =========================
# 4. メイン画面・実行
# =========================
st.title(f"🚀 {target_market}・二段上げ狙い")

if st.button("📡 スキャン開始", type="primary"):
    tickers, info_db = load_master_data(target_market)
    
    if not tickers:
        st.warning("対象銘柄が見つかりませんでした。")
        st.stop()

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # yfinanceで一括取得（3ヶ月分）
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        status_text.text(f"スキャン中... {i}/{len(tickers)}")
        progress_bar.progress(i / len(tickers))
        
        try:
            df_batch = yf.download(batch, period="3mo", interval="1d", progress=False, group_by="ticker", threads=True)
            
            # 1銘柄のみの場合のデータ構造補正
            if not isinstance(df_batch.columns, pd.MultiIndex):
                df_batch = pd.concat({batch[0]: df_batch}, axis=1)

            for t in batch:
                if t not in df_batch.columns.levels[0]:
                    continue
                
                stock_data = df_batch[t].dropna()
                is_match, d = check_strategy(stock_data)
                
                if is_match:
                    results.append({
                        "コード": t.replace(".T", ""),
                        "銘柄名": info_db.get(t, "不明"),
                        "現在値": f"{stock_data['Close'].iloc[-1]:,.1f}",
                        "最大上昇": f"{d['最大上昇']:.1f}%",
                        "出来高枯渇": f"{d['枯渇度']:.2f}倍",
                        "25MA乖離": f"{d['乖離率']:.1f}%",
                        "売買代金": f"{d['代金']:.2f}億円"
                    })
        except Exception:
            continue
            
    progress_bar.progress(1.0)
    status_text.empty()

    if results:
        st.success(f"🎯 {len(results)} 銘柄が条件に合致しました")
        # 出来高が枯れている順（枯渇度が低い順）に表示
        res_df = pd.DataFrame(results).sort_values("出来高枯渇")
        st.dataframe(res_df, use_container_width=True, hide_index=True)
    else:
        st.warning("該当銘柄なし。パラメータを緩めてみてください。")
