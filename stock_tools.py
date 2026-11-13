from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


def normalize_symbol(value: str) -> str:
    symbol = value.strip().upper().replace(" ", "")
    if not symbol:
        raise ValueError("Enter a ticker symbol.")
    if "." not in symbol:
        symbol += ".NS"
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9&_-]{0,24}\.(NS|BO)", symbol):
        raise ValueError("Use an NSE ticker such as RELIANCE or RELIANCE.NS, or a BSE ticker ending in .BO.")
    return symbol


def fetch_prices(symbol: str, period: str = "5y") -> pd.DataFrame:
    try:
        data = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False, threads=False, timeout=20, group_by="column")
    except Exception as exc:
        raise RuntimeError(f"Price download failed ({type(exc).__name__}). Yahoo Finance may be rate-limiting requests; retry later.") from exc
    if data is None or data.empty:
        raise ValueError("No daily prices returned. Check the symbol (NSE symbols usually end in .NS) or retry later.")
    if isinstance(data.columns, pd.MultiIndex):
        matching = [level for level in range(data.columns.nlevels) if symbol in data.columns.get_level_values(level)]
        if matching:
            data = data.xs(symbol, axis=1, level=matching[-1], drop_level=True)
        else:
            data.columns = data.columns.get_level_values(0)
    data.columns = [str(col).title() for col in data.columns]
    if "Close" not in data.columns:
        raise ValueError("The downloaded data did not contain a closing price.")
    data = data.loc[~data.index.duplicated(keep="last")].sort_index()
    for col in ("Close", "Volume"):
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["Close"])
    if len(data) < 220:
        raise ValueError(f"Only {len(data)} daily rows were returned. At least 220 are needed for long-term indicators.")
    return data


def analyze_prices(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    close = out["Close"].astype(float)
    delta = close.diff()
    gains, losses = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI"] = 100 - 100 / (1 + rs)
    out.loc[(avg_loss == 0) & (avg_gain > 0), "RSI"] = 100
    out["SMA50"] = close.rolling(50, min_periods=50).mean()
    out["SMA200"] = close.rolling(200, min_periods=200).mean()
    out["High126"] = close.rolling(126, min_periods=1).max()
    out["Drawdown126Pct"] = (close / out["High126"] - 1) * 100
    out["BelowSMA50"] = close < out["SMA50"]
    out["DipContext"] = out["BelowSMA50"] & (out["Drawdown126Pct"] <= -10)
    out["Volatility20Pct"] = out["Close"].pct_change().rolling(20).std() * np.sqrt(252) * 100
    return out.dropna(subset=["RSI", "SMA50", "SMA200"])


def fetch_fundamentals(symbol: str) -> dict[str, Any]:
    fields = {"market_cap_cr": "marketCap", "trailing_pe": "trailingPE", "return_on_equity_pct": "returnOnEquity", "profit_margin_pct": "profitMargins", "debt_to_equity": "debtToEquity", "current_ratio": "currentRatio"}
    try:
        info = yf.Ticker(symbol).get_info()
        facts: dict[str, Any] = {}
        for output, source in fields.items():
            try:
                raw = float(info.get(source))
                facts[output] = raw / 10_000_000 if output == "market_cap_cr" else raw * 100 if output in ("return_on_equity_pct", "profit_margin_pct") else raw
            except (TypeError, ValueError):
                facts[output] = None
        facts["currency"] = info.get("currency")
        return facts
    except Exception as exc:
        return {"error": f"{type(exc).__name__}; try again later"}


def screen_fundamentals(facts: dict[str, Any], minimum_market_cap_cr: float = 0) -> dict[str, Any]:
    if facts.get("error"):
        return {"summary": "Unavailable", "checks": {}}
    checks: dict[str, bool | None] = {}
    for name, key, test in [
        ("Market value floor", "market_cap_cr", lambda x: x >= minimum_market_cap_cr),
        ("Positive return on equity", "return_on_equity_pct", lambda x: x > 8),
        ("Positive profit margin", "profit_margin_pct", lambda x: x > 0),
        ("Debt/equity below 150", "debt_to_equity", lambda x: x < 150),
        ("Positive trailing earnings", "trailing_pe", lambda x: x > 0),
    ]:
        value = facts.get(key)
        checks[name] = None if value is None else test(value)
    available = [value for value in checks.values() if value is not None]
    if not available:
        summary = "Insufficient fundamental data"
    elif any(value is False for value in available):
        summary = f"{sum(available)}/{len(available)} available checks pass"
    elif len(available) < len(checks):
        summary = f"Available checks pass ({len(available)}/{len(checks)}; some missing)"
    else:
        summary = "All listed checks pass; review valuation and business context"
    return {"summary": summary, "checks": checks}


def _model_frame(data: pd.DataFrame, horizon: int = 20, threshold: float = 0.05) -> tuple[pd.DataFrame, pd.Series]:
    close = data["Close"].astype(float)
    returns = close.pct_change()
    features = pd.DataFrame(index=data.index)
    features["return_1"], features["return_5"], features["return_20"] = returns, close.pct_change(5), close.pct_change(20)
    features["rsi"] = analyze_prices(data)["RSI"]
    features["price_vs_sma50"] = close / close.rolling(50).mean() - 1
    features["price_vs_sma200"] = close / close.rolling(200).mean() - 1
    features["volatility_20"] = returns.rolling(20).std()
    features["volume_vs_20d"] = data["Volume"] / data["Volume"].rolling(20).mean() - 1 if "Volume" in data else 0.0
    future_return = close.shift(-horizon) / close - 1
    target = (future_return > threshold).astype(float)
    target.loc[future_return.isna()] = np.nan
    valid = features.replace([np.inf, -np.inf], np.nan).notna().all(axis=1) & target.notna()
    return features.loc[valid], target.loc[valid].astype(int)


def evaluate_forward_return_model(data: pd.DataFrame) -> dict[str, Any]:
    horizon, threshold = 20, 0.05
    features, target = _model_frame(data, horizon, threshold)
    if len(features) < 350:
        return {"status": "insufficient", "message": f"Only {len(features)} usable labeled dates. At least 350 are required for this simple chronological holdout."}
    splitter = TimeSeriesSplit(n_splits=4, gap=horizon)
    actual, predicted, probability, baseline = [], [], [], []
    train_sizes = []
    for train_indices, test_indices in splitter.split(features):
        x_train, y_train = features.iloc[train_indices], target.iloc[train_indices]
        x_test, y_test = features.iloc[test_indices], target.iloc[test_indices]
        if y_train.nunique() < 2:
            continue
        model = RandomForestClassifier(n_estimators=250, min_samples_leaf=8, max_features="sqrt", class_weight="balanced_subsample", random_state=42, n_jobs=1)
        model.fit(x_train, y_train)
        actual.extend(y_test.tolist())
        predicted.extend(model.predict(x_test).tolist())
        probability.extend(model.predict_proba(x_test)[:, list(model.classes_).index(1)].tolist())
        majority = int(y_train.mean() >= 0.5)
        baseline.extend([majority] * len(y_test))
        train_sizes.append(len(x_train))
    if not actual or len(set(actual)) < 2:
        return {"status": "insufficient", "message": "The walk-forward test folds do not contain enough examples of both outcomes to calculate a useful score."}
    return {
        "status": "ok", "train_rows": int(round(float(np.mean(train_sizes)))), "test_rows": len(actual),
        "positive_rate": float(np.mean(actual)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "baseline_accuracy": float(np.mean(np.asarray(actual) == np.asarray(baseline))),
        "roc_auc": float(roc_auc_score(actual, probability)),
        "folds": len(train_sizes),
    }

