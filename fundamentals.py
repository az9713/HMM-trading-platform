"""
fundamentals.py — Fundamental analysis via yfinance.

Provides company overview, financial ratios, statements,
analyst data, and a composite financial health score.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Any, Optional


class FundamentalAnalyzer:
    """Fetches and processes fundamental data from Yahoo Finance."""

    # Cache ticker objects to avoid redundant API calls within a session
    _ticker_cache: Dict[str, yf.Ticker] = {}

    def _get_ticker(self, symbol: str) -> yf.Ticker:
        if symbol not in self._ticker_cache:
            self._ticker_cache[symbol] = yf.Ticker(symbol)
        return self._ticker_cache[symbol]

    @staticmethod
    def is_crypto(ticker: str) -> bool:
        """Return True if the ticker looks like a crypto asset."""
        t = ticker.upper()
        return "-USD" in t or "-BTC" in t or "-ETH" in t

    # ── Formatting helpers ───────────────────────────────────────────────

    @staticmethod
    def format_large_number(n) -> str:
        """Format a number into human-readable form (e.g. 1.2B, 340M)."""
        if n is None or (isinstance(n, float) and np.isnan(n)):
            return "N/A"
        n = float(n)
        abs_n = abs(n)
        sign = "-" if n < 0 else ""
        if abs_n >= 1e12:
            return f"{sign}{abs_n / 1e12:.2f}T"
        if abs_n >= 1e9:
            return f"{sign}{abs_n / 1e9:.2f}B"
        if abs_n >= 1e6:
            return f"{sign}{abs_n / 1e6:.1f}M"
        if abs_n >= 1e3:
            return f"{sign}{abs_n / 1e3:.1f}K"
        return f"{sign}{abs_n:.2f}"

    @staticmethod
    def _safe_get(info: dict, key: str, default=None):
        """Safely retrieve a value from the info dict."""
        val = info.get(key, default)
        if val is None:
            return default
        return val

    # ── Company overview ─────────────────────────────────────────────────

    def get_company_overview(self, ticker: str) -> Dict[str, Any]:
        """Return company-level information."""
        t = self._get_ticker(ticker)
        try:
            info = t.info or {}
        except Exception:
            info = {}

        return {
            "name": info.get("longName") or info.get("shortName") or ticker,
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap"),
            "market_cap_fmt": self.format_large_number(info.get("marketCap")),
            "description": info.get("longBusinessSummary", "No description available."),
            "website": info.get("website", ""),
            "employees": info.get("fullTimeEmployees"),
            "country": info.get("country", "N/A"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "N/A"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        }

    # ── Financial ratios ─────────────────────────────────────────────────

    def get_financial_ratios(self, ticker: str) -> Dict[str, Any]:
        """Return key valuation and profitability ratios."""
        t = self._get_ticker(ticker)
        try:
            info = t.info or {}
        except Exception:
            info = {}

        ratios = {
            "pe_trailing": info.get("trailingPE"),
            "pe_forward": info.get("forwardPE"),
            "pb": info.get("priceToBook"),
            "ps": info.get("priceToSalesTrailing12Months"),
            "peg": info.get("pegRatio"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "gross_margin": info.get("grossMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "dividend_yield": info.get("dividendYield"),
            "payout_ratio": info.get("payoutRatio"),
            "beta": info.get("beta"),
        }
        return ratios

    # ── Financial statements ─────────────────────────────────────────────

    def get_financial_statements(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """Return income statement, balance sheet, and cash flow DataFrames."""
        t = self._get_ticker(ticker)

        def _safe_df(attr: str) -> pd.DataFrame:
            try:
                df = getattr(t, attr)
                if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                    return pd.DataFrame()
                return df
            except Exception:
                return pd.DataFrame()

        income = _safe_df("income_stmt")
        balance = _safe_df("balance_sheet")
        cashflow = _safe_df("cashflow")

        # Build summary DataFrames with key rows
        income_summary = self._extract_rows(income, [
            "Total Revenue", "Gross Profit", "Operating Income",
            "Net Income", "EBITDA", "Basic EPS",
        ])
        balance_summary = self._extract_rows(balance, [
            "Total Assets", "Total Liabilities Net Minority Interest",
            "Total Debt", "Cash And Cash Equivalents",
            "Stockholders Equity", "Common Stock Equity",
        ])
        cashflow_summary = self._extract_rows(cashflow, [
            "Operating Cash Flow", "Investing Cash Flow",
            "Financing Cash Flow", "Free Cash Flow",
            "Capital Expenditure",
        ])

        return {
            "income": income,
            "balance": balance,
            "cashflow": cashflow,
            "income_summary": income_summary,
            "balance_summary": balance_summary,
            "cashflow_summary": cashflow_summary,
        }

    @staticmethod
    def _extract_rows(df: pd.DataFrame, rows: list) -> pd.DataFrame:
        """Extract specific rows from a financial statement DataFrame."""
        if df.empty:
            return pd.DataFrame()
        available = [r for r in rows if r in df.index]
        if not available:
            return pd.DataFrame()
        return df.loc[available]

    # ── Analyst data ─────────────────────────────────────────────────────

    def get_analyst_data(self, ticker: str) -> Dict[str, Any]:
        """Return analyst recommendations and price targets."""
        t = self._get_ticker(ticker)
        try:
            info = t.info or {}
        except Exception:
            info = {}

        # Recommendations
        try:
            recs = t.recommendations
            if recs is not None and not recs.empty:
                recs_df = recs.tail(20)
            else:
                recs_df = pd.DataFrame()
        except Exception:
            recs_df = pd.DataFrame()

        # Earnings dates
        try:
            earnings_dates = t.earnings_dates
            if earnings_dates is not None and not earnings_dates.empty:
                earnings_dates = earnings_dates.head(4)
            else:
                earnings_dates = pd.DataFrame()
        except Exception:
            earnings_dates = pd.DataFrame()

        return {
            "recommendation_key": info.get("recommendationKey", "N/A"),
            "recommendation_mean": info.get("recommendationMean"),
            "number_of_analysts": info.get("numberOfAnalystOpinions"),
            "target_high": info.get("targetHighPrice"),
            "target_low": info.get("targetLowPrice"),
            "target_mean": info.get("targetMeanPrice"),
            "target_median": info.get("targetMedianPrice"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "recommendations": recs_df,
            "earnings_dates": earnings_dates,
        }

    # ── Financial health score ───────────────────────────────────────────

    def health_score(self, ratios: Dict[str, Any]) -> int:
        """
        Compute a simple 0-100 financial health score based on key ratios.

        Scoring breakdown (each category 0-20 points):
        1. Profitability  (ROE, profit margin)
        2. Valuation      (P/E, PEG)
        3. Liquidity      (current ratio, quick ratio)
        4. Leverage        (debt-to-equity)
        5. Growth          (revenue growth, earnings growth)
        """
        score = 0

        # 1. Profitability (0-20)
        roe = ratios.get("roe")
        pm = ratios.get("profit_margin")
        prof_pts = 0
        if roe is not None:
            if roe > 0.20:
                prof_pts += 10
            elif roe > 0.10:
                prof_pts += 7
            elif roe > 0:
                prof_pts += 3
        if pm is not None:
            if pm > 0.20:
                prof_pts += 10
            elif pm > 0.10:
                prof_pts += 7
            elif pm > 0:
                prof_pts += 3
        score += min(prof_pts, 20)

        # 2. Valuation (0-20)
        pe = ratios.get("pe_trailing")
        peg = ratios.get("peg")
        val_pts = 0
        if pe is not None:
            if 0 < pe < 15:
                val_pts += 10
            elif 0 < pe < 25:
                val_pts += 7
            elif 0 < pe < 40:
                val_pts += 4
        if peg is not None:
            if 0 < peg < 1:
                val_pts += 10
            elif 0 < peg < 2:
                val_pts += 7
            elif 0 < peg < 3:
                val_pts += 4
        score += min(val_pts, 20)

        # 3. Liquidity (0-20)
        cr = ratios.get("current_ratio")
        qr = ratios.get("quick_ratio")
        liq_pts = 0
        if cr is not None:
            if cr > 2.0:
                liq_pts += 10
            elif cr > 1.5:
                liq_pts += 7
            elif cr > 1.0:
                liq_pts += 4
        if qr is not None:
            if qr > 1.5:
                liq_pts += 10
            elif qr > 1.0:
                liq_pts += 7
            elif qr > 0.5:
                liq_pts += 4
        score += min(liq_pts, 20)

        # 4. Leverage (0-20)
        dte = ratios.get("debt_to_equity")
        lev_pts = 0
        if dte is not None:
            if dte < 30:
                lev_pts = 20
            elif dte < 50:
                lev_pts = 15
            elif dte < 100:
                lev_pts = 10
            elif dte < 200:
                lev_pts = 5
        else:
            lev_pts = 10  # neutral if unavailable
        score += lev_pts

        # 5. Growth (0-20)
        rg = ratios.get("revenue_growth")
        eg = ratios.get("earnings_growth")
        grow_pts = 0
        if rg is not None:
            if rg > 0.20:
                grow_pts += 10
            elif rg > 0.05:
                grow_pts += 7
            elif rg > 0:
                grow_pts += 3
        if eg is not None:
            if eg > 0.20:
                grow_pts += 10
            elif eg > 0.05:
                grow_pts += 7
            elif eg > 0:
                grow_pts += 3
        score += min(grow_pts, 20)

        return min(max(score, 0), 100)

    @staticmethod
    def health_color(score: int) -> str:
        """Return a color for the health score."""
        if score >= 70:
            return "#388e3c"  # green
        if score >= 40:
            return "#f57c00"  # orange
        return "#d32f2f"  # red

    @staticmethod
    def ratio_color(name: str, value) -> str:
        """Return green/yellow/red for a given ratio value."""
        if value is None:
            return "#9e9e9e"

        thresholds = {
            "pe_trailing":    [(15, "green"), (25, "orange"), (999, "red")],
            "pe_forward":     [(15, "green"), (25, "orange"), (999, "red")],
            "pb":             [(1.5, "green"), (3, "orange"), (999, "red")],
            "ps":             [(2, "green"), (5, "orange"), (999, "red")],
            "peg":            [(1, "green"), (2, "orange"), (999, "red")],
            "ev_ebitda":      [(10, "green"), (15, "orange"), (999, "red")],
            "debt_to_equity": [(50, "green"), (100, "orange"), (999, "red")],
            "current_ratio":  [(999, "red"), (1.5, "orange"), (2, "green")],
            "roe":            [(0.05, "red"), (0.15, "orange"), (999, "green")],
            "roa":            [(0.03, "red"), (0.08, "orange"), (999, "green")],
            "profit_margin":  [(0.05, "red"), (0.15, "orange"), (999, "green")],
        }

        color_map = {"green": "#388e3c", "orange": "#f57c00", "red": "#d32f2f"}

        if name in thresholds:
            rules = thresholds[name]
            # For "higher is better" ratios like ROE, order is ascending
            if name in ("current_ratio", "roe", "roa", "profit_margin"):
                if value < rules[0][0]:
                    return color_map[rules[0][1]]
                elif value < rules[1][0]:
                    return color_map[rules[1][1]]
                else:
                    return color_map[rules[2][1]]
            else:
                # Lower is better (PE, PB, etc.)
                for threshold, color in rules:
                    if value <= threshold:
                        return color_map[color]

        return "#9e9e9e"
