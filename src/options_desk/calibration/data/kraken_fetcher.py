"""
Kraken Public API Fetcher for OHLCV Data

Fetches historical OHLCV data from Kraken's public API.

API Documentation: https://docs.kraken.com/rest/#tag/Market-Data/operation/getOHLCData

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import requests
import pandas as pd
import time
from typing import Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class KrakenFetcher:
    """
    Fetch OHLCV data from Kraken exchange.

    API limits:
    - Public endpoints: No strict rate limit, but recommended 1 req/sec
    - OHLC data: Returns up to 720 candles per request
    - Available intervals: 1, 5, 15, 30, 60, 240, 1440, 10080, 21600 (minutes)

    Example:
        fetcher = KrakenFetcher()
        df = fetcher.get_ohlcv('XXBTZUSD', interval=5, days=7)
        print(df.head())
    """

    BASE_URL = "https://api.kraken.com/0/public"

    def __init__(self, rate_limit_delay: float = 1.0):
        """
        Initialize Kraken fetcher.

        Args:
            rate_limit_delay: Delay between requests (seconds)
        """
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()

    def get_ohlcv(
        self,
        pair: str = 'XXBTZUSD',
        interval: int = 5,  # Minutes
        days: Optional[int] = None,
        since: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data.

        Args:
            pair: Trading pair (e.g., 'XXBTZUSD' for BTC/USD, 'XETHZUSD' for ETH/USD)
            interval: Candle interval in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
            days: Fetch last N days of data
            since: Unix timestamp to fetch from

        Returns:
            DataFrame with columns:
            - Date: Datetime index
            - Open, High, Low, Close: OHLC prices
            - Volume: Trading volume
            - count: Number of trades

        Note:
            Kraken pair names use 'X' prefix for crypto (XXBTZUSD = BTC/USD)
            Common pairs: XXBTZUSD, XETHZUSD, XXRPZUSD, etc.
        """
        all_candles = []

        # Calculate since from days if provided
        if days is not None and since is None:
            since = int((datetime.now() - timedelta(days=days)).timestamp())

        params = {
            'pair': pair,
            'interval': interval,
        }

        if since is not None:
            params['since'] = since

        logger.info(f"Fetching Kraken {pair} OHLCV data (interval={interval}m)...")

        iterations = 0
        max_iterations = 100  # Safety limit

        while iterations < max_iterations:
            try:
                # Make request
                url = f"{self.BASE_URL}/OHLC"
                response = self.session.get(url, params=params)
                response.raise_for_status()

                data = response.json()

                if 'error' in data and data['error']:
                    raise ValueError(f"Kraken API error: {data['error']}")

                # Extract candles for the pair
                # Kraken returns data['result'][pair_name] and data['result']['last']
                result_key = list(data['result'].keys())[0]  # Get first key (pair name)
                candles = data['result'][result_key]
                last_timestamp = data['result']['last']

                if not candles:
                    logger.info("No more candles available")
                    break

                all_candles.extend(candles)
                logger.info(f"Fetched {len(all_candles):,} candles (iteration {iterations+1})")

                # Check if we've fetched all available data
                if len(candles) < 720:  # Kraken returns up to 720 candles
                    logger.info("Reached end of available data")
                    break

                # Update params for next request
                params['since'] = last_timestamp

                # Rate limiting
                time.sleep(self.rate_limit_delay)

                iterations += 1

            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching OHLCV: {e}")
                break

        if not all_candles:
            raise ValueError(f"No OHLCV data fetched for {pair}")

        # Convert to DataFrame
        # Kraken OHLC format: [time, open, high, low, close, vwap, volume, count]
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'vwap', 'Volume', 'count']
        )

        # Convert timestamp to datetime
        df['Date'] = pd.to_datetime(df['timestamp'], unit='s')

        # Convert numeric columns
        for col in ['Open', 'High', 'Low', 'Close', 'vwap', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['count'] = pd.to_numeric(df['count'], errors='coerce').astype(int)

        # Set Date as index and sort
        df = df.set_index('Date').sort_index()

        # Drop duplicate indices
        df = df[~df.index.duplicated(keep='first')]

        # Keep only essential columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'count']]

        logger.info(f"Total candles fetched: {len(df):,}")
        logger.info(f"Time span: {df.index[0]} to {df.index[-1]}")

        return df

    def get_asset_pairs(self) -> pd.DataFrame:
        """
        Get list of available trading pairs.

        Returns:
            DataFrame with pair information
        """
        url = f"{self.BASE_URL}/AssetPairs"
        response = self.session.get(url)
        response.raise_for_status()

        data = response.json()

        if 'error' in data and data['error']:
            raise ValueError(f"Kraken API error: {data['error']}")

        pairs = data['result']
        return pd.DataFrame.from_dict(pairs, orient='index')


def quick_fetch_kraken(
    pair: str = 'XXBTZUSD',
    interval: int = 5,
    days: int = 7,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Quick helper to fetch Kraken OHLCV data.

    Args:
        pair: Trading pair (XXBTZUSD for BTC/USD)
        interval: Candle interval in minutes
        days: Days of history to fetch
        save_path: Optional path to save CSV

    Returns:
        DataFrame with OHLCV data

    Example:
        df = quick_fetch_kraken('XXBTZUSD', interval=5, days=7)
        print(df.head())
    """
    fetcher = KrakenFetcher()
    df = fetcher.get_ohlcv(pair=pair, interval=interval, days=days)

    if save_path is not None:
        df.to_csv(save_path)
        logger.info(f"Saved to {save_path}")

    return df
