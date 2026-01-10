"""
CoinGecko Data Fetcher for Cryptocurrency Historical Data

Fetch OHLCV data for cryptocurrencies using CoinGecko's free API.

Features:
- Historical OHLCV data (Open, High, Low, Close, Volume)
- No API key required for basic usage
- Rate limiting handling
- Integration with calibration modules

CoinGecko API Docs: https://www.coingecko.com/en/api/documentation

author: Yunian Pan
email: yp1170@nyu.edu
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Optional, Dict, List, Tuple
import time
import warnings


class CoinGeckoFetcher:
    """
    Fetch cryptocurrency historical data from CoinGecko.

    Free tier limits:
    - 10-50 calls/minute (demo plan)
    - OHLCV data available for most major cryptocurrencies
    - Historical data up to current date

    Example:
        >>> fetcher = CoinGeckoFetcher()
        >>> # Get Bitcoin data
        >>> btc_data = fetcher.get_ohlcv('bitcoin', days=365)
        >>> print(btc_data.head())
        >>>
        >>> # Get multiple cryptocurrencies
        >>> cryptos = fetcher.get_multiple(['bitcoin', 'ethereum'], days=90)
    """

    BASE_URL = "https://api.coingecko.com/api/v3"

    # Popular cryptocurrency IDs (CoinGecko uses IDs, not tickers)
    POPULAR_COINS = {
        'bitcoin': 'BTC',
        'ethereum': 'ETH',
        'binancecoin': 'BNB',
        'ripple': 'XRP',
        'cardano': 'ADA',
        'solana': 'SOL',
        'polkadot': 'DOT',
        'dogecoin': 'DOGE',
        'avalanche-2': 'AVAX',
        'chainlink': 'LINK',
        'litecoin': 'LTC',
        'uniswap': 'UNI',
        'cosmos': 'ATOM',
        'algorand': 'ALGO',
        'stellar': 'XLM',
    }

    def __init__(
        self,
        rate_limit_delay: float = 1.2,
        currency: str = 'usd',
        verbose: bool = True,
    ):
        """
        Initialize CoinGecko fetcher.

        Args:
            rate_limit_delay: Delay between API calls in seconds (default 1.2s)
            currency: Base currency for prices (default 'usd')
            verbose: Print progress messages
        """
        self.rate_limit_delay = rate_limit_delay
        self.currency = currency
        self.verbose = verbose
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Make API request with error handling.

        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters

        Returns:
            JSON response as dictionary

        Raises:
            requests.HTTPError: If request fails
        """
        self._rate_limit()

        url = f"{self.BASE_URL}/{endpoint}"

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                warnings.warn("Rate limit exceeded. Waiting 60 seconds...")
                time.sleep(60)
                return self._make_request(endpoint, params)  # Retry
            else:
                raise e

        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")

    def search_coin(self, query: str) -> List[Dict]:
        """
        Search for cryptocurrency by name or symbol.

        Args:
            query: Search term (e.g., 'bitcoin', 'BTC')

        Returns:
            List of matching coins with id, symbol, name

        Example:
            >>> fetcher.search_coin('bitcoin')
            [{'id': 'bitcoin', 'symbol': 'btc', 'name': 'Bitcoin'}, ...]
        """
        endpoint = "search"
        params = {'query': query}

        result = self._make_request(endpoint, params)
        coins = result.get('coins', [])

        # Format results
        formatted = []
        for coin in coins[:10]:  # Top 10 results
            formatted.append({
                'id': coin['id'],
                'symbol': coin['symbol'].upper(),
                'name': coin['name'],
                'market_cap_rank': coin.get('market_cap_rank'),
            })

        return formatted

    def get_ohlcv(
        self,
        coin_id: str,
        days: int = 365,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Get OHLCV data for a cryptocurrency.

        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin', 'ethereum')
                    Use search_coin() to find the correct ID
            days: Number of days of history (ignored if start_date provided)
                 Options: 1, 7, 14, 30, 90, 180, 365, 'max'
            start_date: Start date (optional, overrides days parameter)
            end_date: End date (optional, defaults to today)

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume

        Example:
            >>> # Last 365 days
            >>> btc = fetcher.get_ohlcv('bitcoin', days=365)
            >>>
            >>> # Specific date range
            >>> btc = fetcher.get_ohlcv(
            ...     'bitcoin',
            ...     start_date=date(2020, 1, 1),
            ...     end_date=date(2023, 12, 31)
            ... )
        """
        if self.verbose:
            print(f"Fetching OHLCV data for {coin_id}...")

        # Determine date range
        if start_date is not None:
            # Use range endpoint for specific dates
            return self._get_ohlcv_range(coin_id, start_date, end_date)
        else:
            # Use days endpoint
            return self._get_ohlcv_days(coin_id, days)

    def _get_ohlcv_days(self, coin_id: str, days: int) -> pd.DataFrame:
        """
        Get OHLCV using days parameter.

        CoinGecko returns:
        - 1-90 days: hourly data
        - 90+ days: daily data
        """
        endpoint = f"coins/{coin_id}/ohlc"
        params = {
            'vs_currency': self.currency,
            'days': days if days != 'max' else 'max',
        }

        data = self._make_request(endpoint, params)

        # Parse response: [[timestamp, open, high, low, close], ...]
        if not data:
            raise ValueError(f"No data returned for {coin_id}")

        df = pd.DataFrame(data, columns=['timestamp', 'Open', 'High', 'Low', 'Close'])

        # Convert timestamp to datetime
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop('timestamp', axis=1)

        # Get volume data separately (OHLC endpoint doesn't include volume)
        volume_data = self._get_volume_data(coin_id, days)

        # Merge volume
        if volume_data is not None:
            df = df.merge(volume_data, on='Date', how='left')
        else:
            df['Volume'] = np.nan

        # Reorder columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.sort_values('Date').reset_index(drop=True)

        if self.verbose:
            print(f"  Retrieved {len(df)} data points")
            print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")

        return df

    def _get_ohlcv_range(
        self,
        coin_id: str,
        start_date: date,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Get OHLCV for specific date range.

        Uses market_chart/range endpoint which provides price and volume.
        """
        if end_date is None:
            end_date = date.today()

        # Convert to Unix timestamps
        start_ts = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        end_ts = int(datetime.combine(end_date, datetime.max.time()).timestamp())

        endpoint = f"coins/{coin_id}/market_chart/range"
        params = {
            'vs_currency': self.currency,
            'from': start_ts,
            'to': end_ts,
        }

        data = self._make_request(endpoint, params)

        # Parse prices: [[timestamp, price], ...]
        prices = data.get('prices', [])
        volumes = data.get('total_volumes', [])

        if not prices:
            raise ValueError(f"No data returned for {coin_id}")

        # Create DataFrame from prices
        df_price = pd.DataFrame(prices, columns=['timestamp', 'Close'])
        df_price['Date'] = pd.to_datetime(df_price['timestamp'], unit='ms')

        # Add volume
        if volumes:
            df_volume = pd.DataFrame(volumes, columns=['timestamp', 'Volume'])
            df_volume['Date'] = pd.to_datetime(df_volume['timestamp'], unit='ms')
            df = df_price.merge(df_volume[['Date', 'Volume']], on='Date', how='left')
        else:
            df = df_price
            df['Volume'] = np.nan

        df = df.drop('timestamp', axis=1)

        # Aggregate to daily (market_chart/range returns hourly/daily depending on range)
        df['Date'] = df['Date'].dt.date
        df_daily = df.groupby('Date').agg({
            'Close': 'last',  # Last price of day
            'Volume': 'sum',   # Total volume
        }).reset_index()

        # For OHLC, we need to use the OHLC endpoint
        # But for range queries, we approximate:
        df_daily['Open'] = df_daily['Close'].shift(1)
        df_daily['High'] = df_daily['Close']  # Approximation
        df_daily['Low'] = df_daily['Close']   # Approximation

        # Fill first open with close
        df_daily.loc[0, 'Open'] = df_daily.loc[0, 'Close']

        df_daily = df_daily[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df_daily = df_daily.sort_values('Date').reset_index(drop=True)

        # Convert Date back to datetime
        df_daily['Date'] = pd.to_datetime(df_daily['Date'])

        if self.verbose:
            print(f"  Retrieved {len(df_daily)} data points")
            print(f"  Date range: {df_daily['Date'].min()} to {df_daily['Date'].max()}")
            warnings.warn(
                "Range query uses approximated OHLC (High/Low = Close). "
                "For accurate OHLC, use days parameter instead."
            )

        return df_daily

    def _get_volume_data(self, coin_id: str, days: int) -> Optional[pd.DataFrame]:
        """
        Get volume data separately.

        The OHLC endpoint doesn't include volume, so we fetch it from market_chart.
        """
        try:
            endpoint = f"coins/{coin_id}/market_chart"
            params = {
                'vs_currency': self.currency,
                'days': days if days != 'max' else 'max',
                'interval': 'daily',
            }

            data = self._make_request(endpoint, params)
            volumes = data.get('total_volumes', [])

            if not volumes:
                return None

            df = pd.DataFrame(volumes, columns=['timestamp', 'Volume'])
            df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.drop('timestamp', axis=1)

            # Round to date (remove time component for merging)
            df['Date'] = df['Date'].dt.floor('D')

            return df[['Date', 'Volume']]

        except Exception as e:
            warnings.warn(f"Could not fetch volume data: {e}")
            return None

    def get_multiple(
        self,
        coin_ids: List[str],
        days: int = 365,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Get OHLCV data for multiple cryptocurrencies.

        Args:
            coin_ids: List of CoinGecko coin IDs
            days: Number of days of history
            start_date: Start date (optional)
            end_date: End date (optional)

        Returns:
            Dictionary mapping coin_id to DataFrame

        Example:
            >>> cryptos = fetcher.get_multiple(
            ...     ['bitcoin', 'ethereum', 'solana'],
            ...     days=180
            ... )
            >>> btc_data = cryptos['bitcoin']
            >>> eth_data = cryptos['ethereum']
        """
        results = {}

        if self.verbose:
            print(f"Fetching data for {len(coin_ids)} cryptocurrencies...")

        for coin_id in coin_ids:
            try:
                df = self.get_ohlcv(coin_id, days, start_date, end_date)
                results[coin_id] = df
            except Exception as e:
                warnings.warn(f"Failed to fetch {coin_id}: {e}")
                results[coin_id] = None

        return results

    def get_current_price(self, coin_ids: List[str]) -> Dict[str, float]:
        """
        Get current price for cryptocurrencies.

        Args:
            coin_ids: List of coin IDs

        Returns:
            Dictionary mapping coin_id to current price

        Example:
            >>> prices = fetcher.get_current_price(['bitcoin', 'ethereum'])
            >>> print(f"BTC: ${prices['bitcoin']:,.2f}")
        """
        endpoint = "simple/price"
        params = {
            'ids': ','.join(coin_ids),
            'vs_currencies': self.currency,
        }

        data = self._make_request(endpoint, params)

        # Parse response
        prices = {}
        for coin_id in coin_ids:
            if coin_id in data:
                prices[coin_id] = data[coin_id][self.currency]
            else:
                prices[coin_id] = None

        return prices

    def get_market_data(
        self,
        coin_id: str,
        days: int = 365,
    ) -> Dict:
        """
        Get comprehensive market data including price stats, market cap, etc.

        Args:
            coin_id: CoinGecko coin ID
            days: Number of days for historical data

        Returns:
            Dictionary with:
            - 'ohlcv': OHLCV DataFrame
            - 'current_price': Current price
            - 'market_cap': Current market cap
            - 'volume_24h': 24h trading volume
            - 'price_change_24h': 24h price change %
            - 'ath': All-time high price
            - 'atl': All-time low price

        Example:
            >>> data = fetcher.get_market_data('bitcoin', days=365)
            >>> print(f"Current: ${data['current_price']:,.2f}")
            >>> print(f"ATH: ${data['ath']:,.2f}")
            >>> df = data['ohlcv']
        """
        # Get OHLCV data
        ohlcv = self.get_ohlcv(coin_id, days)

        # Get current market data
        endpoint = f"coins/{coin_id}"
        params = {
            'localization': 'false',
            'tickers': 'false',
            'community_data': 'false',
            'developer_data': 'false',
        }

        data = self._make_request(endpoint, params)

        market_data = data.get('market_data', {})

        return {
            'ohlcv': ohlcv,
            'current_price': market_data.get('current_price', {}).get(self.currency),
            'market_cap': market_data.get('market_cap', {}).get(self.currency),
            'volume_24h': market_data.get('total_volume', {}).get(self.currency),
            'price_change_24h': market_data.get('price_change_percentage_24h'),
            'ath': market_data.get('ath', {}).get(self.currency),
            'ath_date': market_data.get('ath_date', {}).get(self.currency),
            'atl': market_data.get('atl', {}).get(self.currency),
            'atl_date': market_data.get('atl_date', {}).get(self.currency),
            'circulating_supply': market_data.get('circulating_supply'),
            'total_supply': market_data.get('total_supply'),
        }

    def list_popular_coins(self) -> pd.DataFrame:
        """
        List popular cryptocurrencies with their IDs.

        Returns:
            DataFrame with coin_id, symbol, name

        Example:
            >>> coins = fetcher.list_popular_coins()
            >>> print(coins)
        """
        coins = []
        for coin_id, symbol in self.POPULAR_COINS.items():
            coins.append({
                'coin_id': coin_id,
                'symbol': symbol,
                'name': coin_id.replace('-', ' ').title(),
            })

        return pd.DataFrame(coins)


def download_bitcoin(
    days: int = 365,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Convenience function to download Bitcoin data.

    Args:
        days: Number of days (default 365 = 1 year)
        start_date: Optional start date
        end_date: Optional end date

    Returns:
        DataFrame with Bitcoin OHLCV data

    Example:
        >>> from calibration.data import download_bitcoin
        >>> btc = download_bitcoin(days='max')  # All available history
        >>> btc = download_bitcoin(start_date=date(2020, 1, 1))
    """
    fetcher = CoinGeckoFetcher()
    return fetcher.get_ohlcv('bitcoin', days, start_date, end_date)


def download_crypto_basket(
    coins: List[str] = None,
    days: int = 365,
) -> Dict[str, pd.DataFrame]:
    """
    Download data for a basket of cryptocurrencies.

    Args:
        coins: List of coin IDs (default: BTC, ETH, BNB, SOL)
        days: Number of days

    Returns:
        Dictionary mapping coin_id to DataFrame

    Example:
        >>> basket = download_crypto_basket(
        ...     coins=['bitcoin', 'ethereum', 'solana'],
        ...     days=180
        ... )
        >>> btc = basket['bitcoin']
    """
    if coins is None:
        coins = ['bitcoin', 'ethereum', 'binancecoin', 'solana']

    fetcher = CoinGeckoFetcher()
    return fetcher.get_multiple(coins, days=days)
