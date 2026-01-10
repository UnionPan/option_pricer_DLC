"""
Gemini Public API Fetcher for Tick Data

Fetches historical trade data from Gemini's public API.
Gemini provides free access to recent trade history.

API Documentation: https://docs.gemini.com/rest-api/#trades

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import requests
import pandas as pd
import time
from typing import Optional, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class GeminiFetcher:
    """
    Fetch historical trade data from Gemini exchange.

    API limits:
    - Public endpoints: 120 requests/minute
    - Trade history: Last 500 trades per request
    - Can paginate using timestamp and limit_trades parameters

    Example:
        fetcher = GeminiFetcher()
        df = fetcher.get_recent_trades('btcusd', hours=24)
        df.to_parquet('gemini_btcusd_24h.parquet')
    """

    BASE_URL = "https://api.gemini.com/v1"

    def __init__(self, rate_limit_delay: float = 0.5):
        """
        Initialize Gemini fetcher.

        Args:
            rate_limit_delay: Delay between requests (seconds) to avoid rate limiting
        """
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()

    def get_recent_trades(
        self,
        symbol: str = 'btcusd',
        hours: Optional[int] = None,
        max_trades: Optional[int] = None,
        since_timestamp: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch recent trade history.

        Args:
            symbol: Trading pair (e.g., 'btcusd', 'ethusd')
            hours: Fetch trades from last N hours (None = all available)
            max_trades: Maximum number of trades to fetch
            since_timestamp: Fetch trades after this Unix timestamp

        Returns:
            DataFrame with columns:
            - timestamp: Unix timestamp (seconds)
            - timestampms: Unix timestamp (milliseconds)
            - tid: Trade ID
            - price: Trade price
            - amount: Trade amount
            - exchange: 'gemini'
            - type: 'buy' or 'sell'

        Note:
            Gemini's API returns up to 500 trades per request.
            This function paginates to fetch more historical data.
        """
        all_trades = []

        # Calculate since_timestamp from hours if provided
        if hours is not None and since_timestamp is None:
            since_timestamp = int((datetime.now() - timedelta(hours=hours)).timestamp())

        # Start from most recent
        params = {
            'limit_trades': 500,  # Max per request
        }

        if since_timestamp is not None:
            params['since'] = since_timestamp

        logger.info(f"Fetching Gemini {symbol} trades...")

        total_fetched = 0
        iterations = 0
        max_iterations = 1000  # Safety limit

        while iterations < max_iterations:
            try:
                # Make request
                url = f"{self.BASE_URL}/trades/{symbol}"
                response = self.session.get(url, params=params)
                response.raise_for_status()

                trades = response.json()

                if not trades:
                    logger.info("No more trades available")
                    break

                all_trades.extend(trades)
                total_fetched += len(trades)

                logger.info(f"Fetched {total_fetched:,} trades (iteration {iterations+1})")

                # Check if we've fetched enough
                if max_trades is not None and total_fetched >= max_trades:
                    logger.info(f"Reached max_trades limit ({max_trades:,})")
                    break

                # Check if we've gone past the since_timestamp
                oldest_trade_ts = min(t['timestamp'] for t in trades)
                if since_timestamp is not None and oldest_trade_ts < since_timestamp:
                    logger.info(f"Reached since_timestamp ({since_timestamp})")
                    # Filter out trades before since_timestamp
                    all_trades = [t for t in all_trades if t['timestamp'] >= since_timestamp]
                    break

                # Gemini returns trades newest first, so get timestamp of oldest trade
                # and use it for next pagination
                oldest_ts = min(t['timestamp'] for t in trades)

                # Update params for next request (fetch older trades)
                params['since'] = oldest_ts - 1  # Subtract 1 to avoid duplicates

                # Rate limiting
                time.sleep(self.rate_limit_delay)

                iterations += 1

            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching trades: {e}")
                break

        if not all_trades:
            raise ValueError(f"No trades fetched for {symbol}")

        # Convert to DataFrame
        df = pd.DataFrame(all_trades)

        # Add exchange column
        df['exchange'] = 'gemini'

        # Ensure numeric types
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df['timestampms'] = pd.to_numeric(df['timestampms'], errors='coerce')
        df['tid'] = pd.to_numeric(df['tid'], errors='coerce')

        # Sort by timestamp (oldest first)
        df = df.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"Total trades fetched: {len(df):,}")
        logger.info(f"Time span: {pd.to_datetime(df['timestamp'].min(), unit='s')} to {pd.to_datetime(df['timestamp'].max(), unit='s')}")

        return df

    def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading pairs.

        Returns:
            List of symbol names (e.g., ['btcusd', 'ethusd', ...])
        """
        url = f"{self.BASE_URL}/symbols"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()


def quick_fetch_gemini(
    symbol: str = 'btcusd',
    hours: int = 24,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Quick helper to fetch Gemini trades.

    Args:
        symbol: Trading pair
        hours: Hours of history to fetch
        save_path: Optional path to save parquet file

    Returns:
        DataFrame of trades

    Example:
        df = quick_fetch_gemini('btcusd', hours=168)  # 1 week
        print(f"Fetched {len(df):,} trades")
    """
    fetcher = GeminiFetcher()
    df = fetcher.get_recent_trades(symbol=symbol, hours=hours)

    if save_path is not None:
        df.to_parquet(save_path, index=False)
        logger.info(f"Saved to {save_path}")

    return df
