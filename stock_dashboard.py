from __future__ import annotations

from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from stock_tools import analyze_prices, evaluate_forward_return_model, fetch_fundamentals, fetch_prices, normalize_symbol, screen_fundamentals

st.set_page_config(page_title="Indian Stock Intelligence", page_icon="📊", layout="wide")
DEFAULT_WATCHLIST = "RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, ITC, LT, SBIN"

@st.cache_data(ttl=900, show_spinner=False)
def cached_prices(symbol: str, period: str) -> pd.DataFrame:
    return fetch_prices(symbol, period)

@st.cache_data(ttl=21_600, show_spinner=False)
def cached_fundamentals(symbol: str) -> dict:
    return fetch_fundamentals(symbol)

def show_price_chart(data: pd.DataFrame, symbol: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(data.index, data["Close"], label="Adjusted close", linewidth=1.5)
    ax.plot(data.index, data["SMA50"], label="50-session average", linewidth=1)
    ax.plot(data.index, data["SMA200"], label="200-session average", linewidth=1)
    ax.set_title(f"{symbol}: price and trend context")
    ax.set_ylabel("Price (INR, adjusted)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.title("Indian Stock Intelligence")
st.caption("Research and learning dashboard for NSE/BSE shares · Data is not live exchange data")
st.warning("This tool is for education and research, not personalized investment advice. A dip can reflect a real business problem. Scores and historical model results do not guarantee future returns. Check official filings and consult a qualified adviser.")

with st.sidebar:
    st.header("Choose shares")
    raw_symbols = st.text_area("Tickers (comma separated)", DEFAULT_WATCHLIST, height=115, help="For NSE, type RELIANCE or RELIANCE.NS. .NS is added automatically.")
    period = st.selectbox("Price history", ["2y", "5y", "10y"], index=1)
    min_market_cap_cr = st.number_input("Minimum market value (₹ crore)", min_value=0, value=5_000, step=1_000)
    if st.button("Refresh data now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

symbols: list[str] = []
for item in raw_symbols.split(","):
    item = item.strip()
    if not item:
        continue
    try:
        symbol = normalize_symbol(item)
        if symbol not in symbols:
            symbols.append(symbol)
    except ValueError as exc:
        st.sidebar.error(f"{item}: {exc}")
if not symbols:
    st.info("Add at least one ticker in the box on the left to begin.")
    st.stop()

st.caption(f"Price cache: up to 15 minutes · Fundamental cache: up to 6 hours · Page checked {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
rows, price_data, fundamentals = [], {}, {}
with st.spinner("Getting data. The free data source may be slow or temporarily unavailable…"):
    for symbol in symbols:
        try:
            prices = analyze_prices(cached_prices(symbol, period))
            price_data[symbol] = prices
            facts = cached_fundamentals(symbol)
            fundamentals[symbol] = facts
            screen = screen_fundamentals(facts, min_market_cap_cr)
            latest = prices.iloc[-1]
            rows.append({
                "Ticker": symbol,
                "Last adjusted close (₹)": latest["Close"],
                "RSI (14)": latest["RSI"],
                "Below 50-session average": bool(latest["BelowSMA50"]),
                "Fall from 6-month high (%)": latest["Drawdown126Pct"],
                "Dip context": "Possible pullback" if latest["DipContext"] else "No dip flags",
                "Quality/value checks": screen["summary"],
                "Data as of": str(prices.index[-1].date()),
            })
        except Exception as exc:
            st.error(f"Could not analyze {symbol}: {exc}")
if not rows:
    st.error("No ticker could be loaded. Check spelling and internet access, then try again later.")
    st.stop()

overview = pd.DataFrame(rows).set_index("Ticker")
st.subheader("Watchlist overview")
st.dataframe(overview.style.format({"Last adjusted close (₹)": "₹{:,.2f}", "RSI (14)": "{:.1f}", "Fall from 6-month high (%)": "{:.1f}%"}), use_container_width=True)
st.caption("A 'possible pullback' is only a price-pattern flag (below 50-session average and at least 10% below its recent 6-month high). It is not a valuation judgment or a buy signal.")

chosen = st.selectbox("Explore a share", list(price_data))
prices = price_data[chosen]
latest = prices.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Adjusted close", f"₹{latest['Close']:,.2f}")
c2.metric("RSI (14)", f"{latest['RSI']:.1f}")
c3.metric("From 6-month high", f"{latest['Drawdown126Pct']:.1f}%")
c4.metric("20-session volatility (annualized)", f"{latest['Volatility20Pct']:.1f}%")
show_price_chart(prices, chosen)
with st.expander("What do the dip and trend measures mean?"):
    st.markdown("- **RSI** summarizes recent price momentum; low RSI does not prove a share is cheap.\n- **50/200-session averages** compare recent prices with longer trends.\n- **Fall from 6-month high** describes the size of a recent decline; it does not identify why it happened.\n- **Volatility** describes how widely daily returns have moved. It is not a complete measure of risk.")

st.subheader("Fundamental checks")
facts = fundamentals[chosen]
if facts.get("error"):
    st.info(f"Fundamental data could not be retrieved: {facts['error']}")
else:
    fundamental_table = pd.DataFrame([
        ("Market value", facts.get("market_cap_cr"), "₹ crore"),
        ("Trailing P/E", facts.get("trailing_pe"), "times; not comparable across all industries"),
        ("Return on equity", facts.get("return_on_equity_pct"), "%"),
        ("Profit margin", facts.get("profit_margin_pct"), "%"),
        ("Debt/equity", facts.get("debt_to_equity"), "ratio; data provider's units"),
        ("Current ratio", facts.get("current_ratio"), "ratio"),
    ], columns=["Measure", "Value", "Units / note"])
    fundamental_table["Value"] = fundamental_table["Value"].map(lambda x: "Not provided" if pd.isna(x) else f"{x:,.2f}")
    st.dataframe(fundamental_table, hide_index=True, use_container_width=True)
    st.markdown(f"**Rule-based check:** {screen_fundamentals(facts, min_market_cap_cr)['summary']}")
    st.caption("Provider fundamentals can be missing, stale, differently defined, or reported in a different currency/period. Missing values are shown as unavailable; this screen is not a full valuation or quality assessment.")

st.subheader("Historical ML evaluation (research only)")
st.write("The model uses only past price/volume features to classify whether adjusted close was more than 5% higher after 20 trading sessions. Four expanding walk-forward test folds keep dates in order, with a 20-session gap before each test fold to avoid training on labels whose outcome period reaches into that test. This is a made-up research threshold, not a target price.")
try:
    evaluation = evaluate_forward_return_model(prices)
    if evaluation["status"] == "ok":
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Holdout dates", str(evaluation["test_rows"]))
        m2.metric("Positive-class share", f"{evaluation['positive_rate']:.1%}")
        m3.metric("Balanced accuracy", f"{evaluation['balanced_accuracy']:.2f}")
        m4.metric("ROC AUC", "Unavailable" if evaluation["roc_auc"] is None else f"{evaluation['roc_auc']:.2f}")
        st.caption(f"Walk-forward folds: {evaluation['folds']}; average training rows per fold: {evaluation['train_rows']}. Baseline (always choose the more common class in each training fold) accuracy: {evaluation['baseline_accuracy']:.2f}. Results cover this ticker and history only; they do not predict the next move.")
    else:
        st.info(evaluation["message"])
except Exception as exc:
    st.info(f"Could not evaluate the historical model: {exc}")

st.divider()
st.caption("Data source: Yahoo Finance through yfinance. Prices may be delayed, corrected, incomplete, or temporarily unavailable; this is not an official NSE feed. Use exchange and company sources to verify material information.")

