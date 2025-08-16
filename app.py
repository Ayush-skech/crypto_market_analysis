import requests
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np

# Coingecko API URL
COINGECKO_API = "https://api.coingecko.com/api/v3"

# -------API helpers-------
def api_headers():
    return {"accept": "application/json"}

@st.cache_data(ttl=180)
def get_markets(vs="usd", per_page=100, page=1):
    """Fetch top market data."""
    url = f"{COINGECKO_API}/coins/markets"
    params = {
        "vs_currency": vs,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": page,
        "sparkline": "false",
        "price_change_percentage": "1h,24h,7d",
    }
    r = requests.get(url, params=params, headers=api_headers(), timeout=20)
    r.raise_for_status()
    data = pd.DataFrame(r.json())
    keep = [
        "id", "symbol", "name", "current_price", "market_cap", "total_volume",
        "price_change_percentage_1h_in_currency",
        "price_change_percentage_24h_in_currency",
        "price_change_percentage_7d_in_currency",
    ]
    data = data[keep].rename(columns={
        "current_price": "price",
        "total_volume": "volume_24h",
        "price_change_percentage_1h_in_currency": "chg_1h_%",
        "price_change_percentage_24h_in_currency": "chg_24h_%",
        "price_change_percentage_7d_in_currency": "chg_7d_%",
    })
    return data

@st.cache_data(ttl=180)
def get_history(coin_id, vs="usd", days=30):
    """Fetch daily historical prices."""
    url = f"{COINGECKO_API}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs, "days": days, "interval": "daily"}
    r = requests.get(url, params=params, headers=api_headers(), timeout=20)
    r.raise_for_status()
    prices = r.json().get("prices", [])
    if not prices:
        return pd.DataFrame(columns=["date", "price"])
    df = pd.DataFrame(prices, columns=["ts", "price"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    return df[["date", "price"]]

def format_compact(x):
    try:
        x = float(x)
    except Exception:
        return x
    if x >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f}B"
    if x >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    if x >= 1_000:
        return f"{x/1_000:.2f}K"
    return f"{x:.2f}"

def safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except requests.RequestException as e:
        st.warning(f"Network/API error: {e}")
    except Exception as e:
        st.warning(f"Unexpected error: {e}")
    return None

# ---------UI---------
st.set_page_config(page_title="Crypto Market Analysis", layout="wide")
st.title("Crypto Market Analysis Dashboard")
st.caption("Real-time and historical market data • Source: CoinGecko")

with st.sidebar:
    st.subheader("Controls")
    vs = st.selectbox("Currency", ["usd", "inr", "eur"], index=0)
    days = st.selectbox("Timeframe (days)", [7, 30, 90, 180, 365], index=1)
    topn = st.slider("Number of coins", 10, 200, 100, 10)
    refresh = st.button("Refresh Data")
    st.markdown("---")
    st.write("Tip: Use refresh for latest prices.")

if refresh:
    st.cache_data.clear()

markets = safe(get_markets, vs=vs, per_page=min(topn, 250), page=1)
if markets is None or markets.empty:
    st.error("Could not load market data right now.")
    st.stop()

# KPIs
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Loaded Coins", len(markets))
with c2:
    st.metric("Median Price", format_compact(markets["price"].median()))
with c3:
    med_24h = markets["chg_24h_%"].median()
    st.metric("Median 24h %", f"{med_24h:.2f}%")
with c4:
    st.metric("Total Volume", format_compact(markets["volume_24h"].sum()))

# Search + table
query = st.text_input("Search by name or symbol")
table = markets.copy()
if query:
    q = query.strip().lower()
    table = table[table["name"].str.lower().str.contains(q) | table["symbol"].str.lower().str.contains(q)]

st.subheader("Market Snapshot")
st.dataframe(
    table.style.format({
        "price": lambda x: f"{x:,.4f}",
        "market_cap": lambda x: f"{x:,.0f}",
        "volume_24h": lambda x: f"{x:,.0f}",
        "chg_1h_%": lambda x: f"{x:.2f}%",
        "chg_24h_%": lambda x: f"{x:.2f}%",
        "chg_7d_%": lambda x: f"{x:.2f}%",
    }),
    use_container_width=True,
    height=420,
)

# CSV download
st.download_button("Download CSV", table.to_csv(index=False).encode("utf-8"), "markets.csv", "text/csv")

# Gainers / Losers
left, right = st.columns(2)
with left:
    st.markdown("### Top Gainers (24h)")
    gainers = table.sort_values("chg_24h_%", ascending=False).head(10)
    st.dataframe(gainers[["name", "symbol", "price", "chg_24h_%"]], use_container_width=True, height=300)

with right:
    st.markdown("### Top Losers (24h)")
    losers = table.sort_values("chg_24h_%", ascending=True).head(10)
    st.dataframe(losers[["name", "symbol", "price", "chg_24h_%"]], use_container_width=True, height=300)

# Trend chart + ML prediction
st.subheader("Trend + ML Prediction")
coin_list = table["name"].tolist()
selected = st.selectbox("Select a coin", coin_list, index=0)
coin_id = table.loc[table["name"] == selected, "id"].iloc[0]

hist = safe(get_history, coin_id=coin_id, vs=vs, days=days)
if hist is not None and not hist.empty:
    # Plot historical trend
    fig = px.line(hist, x="date", y="price", title=f"{selected} Price • {vs.upper()}")

    # ML: Linear Regression
    hist = hist.reset_index(drop=True)
    hist["day_num"] = np.arange(len(hist))
    X = hist[["day_num"]].values
    y = hist["price"].values

    model = LinearRegression()
    model.fit(X, y)

    next_day_num = np.array([[len(hist)]])
    predicted_price = model.predict(next_day_num)[0]
    next_day_date = hist["date"].iloc[-1] + pd.Timedelta(days=1)

    # Add prediction point
    fig.add_scatter(
        x=[next_day_date], y=[predicted_price],
        mode="markers+text",
        name="Predicted Price",
        text=[f"{predicted_price:.2f}"],
        textposition="top center",
        marker=dict(color="red", size=10)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.success(f"Predicted {selected} price for {next_day_date.date()} = {predicted_price:.2f} {vs.upper()}")
else:
    st.info("No historical data for this selection/timeframe.")
