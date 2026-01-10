"""
Cryptocurrency Data Fetcher using Yahoo Finance

Works globally without API restrictions or rate limits.
Yahoo Finance aggregates crypto data from multiple exchanges.

Simpler and more reliable than exchange-specific APIs.

author: Yunian Pan
email: yp1170@nyu.edu
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict
import warnings
from pathlib import Path


class CryptoFetcher:
    """
    Fetch cryptocurrency data using Yahoo Finance.

    Advantages:
    - NO API key or authentication required
    - NO geo-blocking (works worldwide)
    - NO rate limits for reasonable usage
    - Reliable aggregated data
    - Simple to use

    Example:
        >>> fetcher = CryptoFetcher()
        >>> btc = fetcher.get_ohlcv('BTC-USD', days=365)
        >>> eth = fetcher.get_ohlcv('ETH-USD', days=365)
    """

    # Popular crypto tickers on Yahoo Finance
    POPULAR_TICKERS = {
        'BTC-USD': 'Bitcoin',
        'ETH-USD': 'Ethereum',
        'BNB-USD': 'Binance Coin',
        'SOL-USD': 'Solana',
        'XRP-USD': 'Ripple',
        'ADA-USD': 'Cardano',
        'DOGE-USD': 'Dogecoin',
        'MATIC-USD': 'Polygon',
        'DOT-USD': 'Polkadot',
        'AVAX-USD': 'Avalanche',
        'LINK-USD': 'Chainlink',
        'ATOM-USD': 'Cosmos',
        'LTC-USD': 'Litecoin',
        'UNI-USD': 'Uniswap',
    }

    def __init__(self, verbose: bool = True):
        """
        Initialize crypto fetcher.

        Args:
            verbose: Print progress messages
        """
        self.verbose = verbose

    def get_ohlcv(
        self,
        ticker: str,
        days: Optional[int] = 365,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        interval: str = '1d',
    ) -> pd.DataFrame:
        """
        Get OHLCV data for a cryptocurrency.

        Args:
            ticker: Yahoo Finance ticker (e.g., 'BTC-USD', 'ETH-USD')
                   Use uppercase with -USD suffix
            days: Number of days of history (ignored if start_date provided)
            start_date: Start date (optional, overrides days)
            end_date: End date (optional, defaults to today)
            interval: Data interval
                     Options: '1m', '5m', '15m', '1h', '1d', '1wk', '1mo'
                     Note: Minute data only available for last 7 days

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume

        Example:
            >>> # Last 365 days of daily data
            >>> btc = fetcher.get_ohlcv('BTC-USD', days=365)
            >>>
            >>> # Specific date range
            >>> btc = fetcher.get_ohlcv(
            ...     'BTC-USD',
            ...     start_date=date(2020, 1, 1),
            ...     end_date=date(2023, 12, 31)
            ... )
            >>>
            >>> # Hourly data for last 7 days
            >>> btc = fetcher.get_ohlcv('BTC-USD', days=7, interval='1h')
        """
        if self.verbose:
            print(f"Fetching {ticker} data ({interval} interval)...")

        # Determine date range
        if start_date is not None:
            start = start_date
        else:
            if end_date is not None:
                start = end_date - timedelta(days=days)
            else:
                start = datetime.now() - timedelta(days=days)

        end = end_date if end_date is not None else datetime.now()

        # Download data
        crypto = yf.Ticker(ticker)
        df = crypto.history(start=start, end=end, interval=interval)

        if df.empty:
            raise ValueError(
                f"No data returned for {ticker}. "
                "Check ticker symbol (should be like 'BTC-USD')"
            )

        # Reset index to get Date as column
        df = df.reset_index()

        # Rename columns to match standard format
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'Date'})

        # Select and reorder columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)

        if self.verbose:
            print(f"  Retrieved {len(df)} data points")
            print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")

        return df

    def get_current_price(self, tickers: List[str]) -> Dict[str, float]:
        """
        Get current price for cryptocurrencies.

        Args:
            tickers: List of Yahoo Finance tickers

        Returns:
            Dictionary mapping ticker to current price

        Example:
            >>> prices = fetcher.get_current_price(['BTC-USD', 'ETH-USD'])
            >>> print(f"BTC: ${prices['BTC-USD']:,.2f}")
        """
        prices = {}

        for ticker in tickers:
            crypto = yf.Ticker(ticker)
            info = crypto.info
            prices[ticker] = info.get('regularMarketPrice', np.nan)

        return prices

    def get_info(self, ticker: str) -> Dict:
        """
        Get cryptocurrency information.

        Args:
            ticker: Yahoo Finance ticker

        Returns:
            Dictionary with crypto info

        Example:
            >>> info = fetcher.get_info('BTC-USD')
            >>> print(f"Market Cap: ${info.get('marketCap', 0)/1e9:.2f}B")
        """
        crypto = yf.Ticker(ticker)
        return crypto.info

    def get_multiple(
        self,
        tickers: List[str],
        days: int = 365,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        interval: str = '1d',
    ) -> Dict[str, pd.DataFrame]:
        """
        Get OHLCV data for multiple cryptocurrencies.

        Args:
            tickers: List of Yahoo Finance tickers
            days: Number of days of history
            start_date: Start date (optional)
            end_date: End date (optional)
            interval: Data interval

        Returns:
            Dictionary mapping ticker to DataFrame

        Example:
            >>> cryptos = fetcher.get_multiple(
            ...     ['BTC-USD', 'ETH-USD', 'SOL-USD'],
            ...     days=180
            ... )
            >>> btc_data = cryptos['BTC-USD']
        """
        results = {}

        if self.verbose:
            print(f"Fetching data for {len(tickers)} cryptocurrencies...")

        for ticker in tickers:
            try:
                df = self.get_ohlcv(
                    ticker,
                    days=days,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                )
                results[ticker] = df
            except Exception as e:
                warnings.warn(f"Failed to fetch {ticker}: {e}")
                results[ticker] = None

        return results

    def list_popular(self) -> pd.DataFrame:
        """
        List popular cryptocurrency tickers.

        Returns:
            DataFrame with ticker and name

        Example:
            >>> cryptos = fetcher.list_popular()
            >>> print(cryptos)
        """
        data = []
        for ticker, name in self.POPULAR_TICKERS.items():
            data.append({'ticker': ticker, 'name': name})

        return pd.DataFrame(data)


def download_bitcoin(
    days: int = 365,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    interval: str = '1d',
    save: bool = False,
    filepath: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience function to download Bitcoin data.

    Args:
        days: Number of days (default 365)
        start_date: Optional start date
        end_date: Optional end date
        interval: Data interval (default '1d')
        save: If True, write the data to a CSV in the current working directory (or filepath)
        filepath: Optional custom path for the CSV; defaults to ./btc_usd.csv

    Returns:
        DataFrame with Bitcoin OHLCV data

    Example:
        >>> from calibration.data import download_bitcoin
        >>> btc = download_bitcoin(days=730)  # 2 years
        >>> btc = download_bitcoin(start_date=date(2020, 1, 1))
        >>> btc = download_bitcoin(save=True)  # also writes ./btc_usd.csv
    """
    fetcher = CryptoFetcher()
    df = fetcher.get_ohlcv(
        'BTC-USD',
        days=days,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
    )

    if save:
        path = Path(filepath) if filepath is not None else Path.cwd() / "btc_usd.csv"
        df.to_csv(path, index=False)
        if fetcher.verbose:
            print(f"  Saved BTC data to {path}")

    return df


def download_crypto_basket(
    tickers: List[str] = None,
    days: int = 365,
    interval: str = '1d',
) -> Dict[str, pd.DataFrame]:
    """
    Download data for a basket of cryptocurrencies.

    Args:
        tickers: List of tickers (default: BTC, ETH, BNB, SOL)
        days: Number of days
        interval: Data interval

    Returns:
        Dictionary mapping ticker to DataFrame

    Example:
        >>> basket = download_crypto_basket(
        ...     tickers=['BTC-USD', 'ETH-USD', 'SOL-USD'],
        ...     days=180
        ... )
        >>> btc = basket['BTC-USD']
    """
    if tickers is None:
        tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD']

    fetcher = CryptoFetcher()
    return fetcher.get_multiple(tickers, days=days, interval=interval)
