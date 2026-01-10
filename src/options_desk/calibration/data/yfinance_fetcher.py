"""
yfinance data fetcher for options and spot data.

author: Yunian Pan
email: yp1170@nyu.edu
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, datetime
from typing import Optional, List
import logging

from .data_provider import DataProvider, OptionChain, OptionQuote, MarketData


logger = logging.getLogger(__name__)


class YFinanceFetcher(DataProvider):
    """
    Data provider using yfinance (Yahoo Finance).
    
    Example:
        fetcher = YFinanceFetcher()
        chain = fetcher.get_option_chain('SPY')
        filtered = chain.filter(min_volume=100, min_open_interest=50)
        smile = filtered.get_slice(filtered.get_expiries()[0])
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
    ):
        """
        Initialize fetcher.
        
        Args:
            risk_free_rate: Default risk-free rate for calculations
            dividend_yield: Default dividend yield
        """
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self._cache = {}
    
    def get_spot(self, ticker: str) -> float:
        """
        Get current spot price.
        
        Args:
            ticker: Stock/ETF ticker symbol
            
        Returns:
            Current price
        """
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Try different price fields
        for field in ['regularMarketPrice', 'currentPrice', 'previousClose']:
            if field in info and info[field] is not None:
                return float(info[field])
        
        # Fallback: get from history
        hist = stock.history(period='1d')
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
            
        raise ValueError(f"Could not get spot price for {ticker}")
    
    def get_history(
        self,
        ticker: str,
        start: date,
        end: date,
        include_actions: bool = True,
        auto_adjust: bool = True,
        interval: str = '1d',
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data with dividends and splits.

        Args:
            ticker: Stock/ETF ticker
            start: Start date (date object or string 'YYYY-MM-DD')
            end: End date (date object or string 'YYYY-MM-DD')
            include_actions: Include dividends and stock splits (default True)
            auto_adjust: Use split-adjusted prices (default True)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume,
            and optionally Dividends, Stock Splits, Capital Gains

        Note:
            - Date is returned as a regular column (not index)
            - Prices are adjusted for splits by default
            - Dividends column shows dividend payments on ex-dividend dates
            - Stock Splits column shows split ratios (e.g., 2.0 for 2-for-1 split)
        """
        stock = yf.Ticker(ticker)

        # Convert strings to date objects if needed
        if isinstance(start, str):
            start = datetime.strptime(start, '%Y-%m-%d').date()
        if isinstance(end, str):
            end = datetime.strptime(end, '%Y-%m-%d').date()

        # Fetch historical data
        df = stock.history(start=start, end=end, auto_adjust=auto_adjust, interval=interval)

        if df.empty:
            raise ValueError(f"No history data for {ticker}")

        # Ensure actions data is included
        if include_actions:
            # Ensure Dividends column exists
            if 'Dividends' not in df.columns:
                df['Dividends'] = 0.0

            # Ensure Stock Splits column exists
            if 'Stock Splits' not in df.columns:
                df['Stock Splits'] = 0.0

            # Ensure Capital Gains exists (for ETFs)
            if 'Capital Gains' not in df.columns:
                df['Capital Gains'] = 0.0

            # If all zeros, try fetching actions separately and merge
            if df['Dividends'].sum() == 0 or df['Stock Splits'].sum() == 0:
                try:
                    actions = stock.actions
                    if not actions.empty:
                        # Merge all action columns
                        for col in ['Dividends', 'Stock Splits', 'Capital Gains']:
                            if col in actions.columns:
                                df = df.drop(columns=[col], errors='ignore')
                                df = df.join(actions[[col]], how='left')
                                df[col] = df[col].fillna(0)
                except Exception as e:
                    logger.debug(f"Could not fetch actions separately: {e}")
                    # Ensure columns exist even if fetch failed
                    for col in ['Dividends', 'Stock Splits', 'Capital Gains']:
                        if col not in df.columns:
                            df[col] = 0.0

        # Ensure standard columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Convert Date from index to regular column
        df = df.reset_index()

        # Standardize column order
        base_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if include_actions:
            base_cols.extend(['Dividends', 'Stock Splits', 'Capital Gains'])

        # Keep only relevant columns in standard order
        available_cols = [col for col in base_cols if col in df.columns]
        df = df[available_cols]

        return df
    
    def get_option_chain(
        self,
        ticker: str,
        reference_date: Optional[date] = None,
        expiry: Optional[str] = None,
    ) -> OptionChain:
        """
        Get option chain for a ticker.
        
        Args:
            ticker: Stock/ETF ticker
            reference_date: Reference date (defaults to today)
            expiry: Specific expiry date string (YYYY-MM-DD) or None for all
            
        Returns:
            OptionChain object with all available options
        """
        if reference_date is None:
            reference_date = date.today()
            
        stock = yf.Ticker(ticker)
        spot = self.get_spot(ticker)
        
        # Get available expiries
        try:
            available_expiries = stock.options
        except Exception as e:
            raise ValueError(f"Could not get options for {ticker}: {e}")
        
        if not available_expiries:
            raise ValueError(f"No options available for {ticker}")
        
        options = []
        
        # Process expiries
        expiries_to_fetch = [expiry] if expiry else available_expiries
        
        for exp_str in expiries_to_fetch:
            if exp_str not in available_expiries:
                logger.warning(f"Expiry {exp_str} not available for {ticker}")
                continue
                
            try:
                chain = stock.option_chain(exp_str)
                expiry_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                
                # Process calls
                for _, row in chain.calls.iterrows():
                    opt = self._row_to_option(row, expiry_date, 'call')
                    if opt:
                        options.append(opt)
                
                # Process puts
                for _, row in chain.puts.iterrows():
                    opt = self._row_to_option(row, expiry_date, 'put')
                    if opt:
                        options.append(opt)
                        
            except Exception as e:
                logger.warning(f"Error fetching {exp_str}: {e}")
                continue
        
        return OptionChain(
            underlying=ticker,
            spot_price=spot,
            reference_date=reference_date,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            options=options,
        )
    
    def _row_to_option(
        self,
        row: pd.Series,
        expiry: date,
        option_type: str,
    ) -> Optional[OptionQuote]:
        """Convert yfinance row to OptionQuote."""
        try:
            bid = float(row.get('bid', 0) or 0)
            ask = float(row.get('ask', 0) or 0)
            
            # Skip invalid quotes
            if bid <= 0 or ask <= 0 or ask < bid:
                mid = float(row.get('lastPrice', 0) or 0)
                if mid <= 0:
                    return None
            else:
                mid = (bid + ask) / 2
            
            # Get implied vol (Yahoo sometimes provides this)
            iv = row.get('impliedVolatility')
            if iv is not None and not np.isnan(iv):
                iv = float(iv)
            else:
                iv = None
            
            return OptionQuote(
                strike=float(row['strike']),
                expiry=expiry,
                option_type=option_type,
                bid=bid,
                ask=ask,
                mid=mid,
                last=float(row.get('lastPrice', mid) or mid),
                volume=int(row.get('volume', 0) or 0),
                open_interest=int(row.get('openInterest', 0) or 0),
                implied_volatility=iv,
            )
        except Exception as e:
            logger.debug(f"Error parsing option row: {e}")
            return None
    
    def get_option_expiries(self, ticker: str) -> List[str]:
        """Get available option expiry dates."""
        stock = yf.Ticker(ticker)
        return list(stock.options)

    def get_dividend_yield(
        self,
        ticker: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
        lookback_days: int = 252,
    ) -> float:
        """
        Calculate annualized dividend yield from historical data.

        Args:
            ticker: Stock/ETF ticker
            start: Start date (optional)
            end: End date (optional, defaults to today)
            lookback_days: Days to look back for dividend calculation (default 252 = 1 year)

        Returns:
            Annualized dividend yield as decimal (e.g., 0.015 for 1.5%)

        Example:
            >>> fetcher = YFinanceFetcher()
            >>> spy_div_yield = fetcher.get_dividend_yield('SPY')
            >>> print(f"SPY dividend yield: {spy_div_yield*100:.2f}%")
            SPY dividend yield: 1.32%
        """
        if end is None:
            end = date.today()
        if start is None:
            start = end - pd.Timedelta(days=lookback_days)

        # Fetch historical data with dividends
        df = self.get_history(ticker, start=start, end=end, include_actions=True)

        if 'Dividends' not in df.columns or df['Dividends'].sum() == 0:
            logger.warning(f"No dividend data found for {ticker}")
            return 0.0

        # Sum dividends over period
        total_dividends = df['Dividends'].sum()

        # Get average price over period
        avg_price = df['Close'].mean()

        # Annualize based on actual period length (Date is now a column)
        df['Date'] = pd.to_datetime(df['Date'])
        days_in_period = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days
        if days_in_period == 0:
            return 0.0

        annualization_factor = 365.25 / days_in_period

        # Calculate annualized yield
        dividend_yield = (total_dividends / avg_price) * annualization_factor

        return float(dividend_yield)


# Convenience function
def fetch_spy_data(days: int = 252) -> MarketData:
    """
    Quick helper to fetch SPY data for testing.
    
    Args:
        days: Number of days of history
        
    Returns:
        MarketData for SPY
    """
    fetcher = YFinanceFetcher()
    return fetcher.get_market_data('SPY', history_days=days)
