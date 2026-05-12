"""8-regime market state classifier based on VIX × 10Y × DXY."""

import pandas as pd

REGIME_LABELS = {
    0: "GOLDILOCKS",
    1: "STRONG_USD",
    2: "RATE_PRESSURE",
    3: "DOLLAR_DOMINANCE",
    4: "RISK_OFF",
    5: "FLIGHT_TO_SAFETY",
    6: "STAGFLATION",
    7: "TIGHTENING_CRISIS",
}

# (vix_hi, rate_hi, dxy_up) → regime_id
_REGIME_MAP = {
    (0, 0, 0): 0,
    (0, 0, 1): 1,
    (0, 1, 0): 2,
    (0, 1, 1): 3,
    (1, 0, 0): 4,
    (1, 0, 1): 5,
    (1, 1, 0): 6,
    (1, 1, 1): 7,
}


def classify_regime(vix: float, us10y: float, dxy_5d_chg: float) -> int:
    vix_hi  = int(vix > 20)
    rate_hi = int(us10y > 4.2)
    dxy_up  = int(dxy_5d_chg > 0)
    return _REGIME_MAP.get((vix_hi, rate_hi, dxy_up), 0)


def classify_regime_label(vix: float, us10y: float, dxy_5d_chg: float) -> str:
    return REGIME_LABELS[classify_regime(vix, us10y, dxy_5d_chg)]


def add_regime_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add regime_id column to a DataFrame that has vix, us10y, dxy_5d_chg_pct."""
    df = df.copy()

    def _row_regime(row):
        return classify_regime(
            float(row.get("vix", 20) or 20),
            float(row.get("us10y", 4.0) or 4.0),
            float(row.get("dxy_5d_chg_pct", 0) or 0),
        )

    df["regime_id"] = df.apply(_row_regime, axis=1)
    return df
