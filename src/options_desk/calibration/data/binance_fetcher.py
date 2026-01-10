"""
Binance Data Fetcher for Cryptocurrency Historical Data

Fetch OHLCV data directly from Binance exchange.

Advantages over CoinGecko:
- NO API key required for public data
- Real exchange data (not aggregated)
- Higher rate limits
- More granular intervals (1m, 5m, 1h, 1d, etc.)
- Very reliable and fast

Binance API Docs: https://binance-docs.github.io/apidocs/spot/en/

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


class BinanceFetcher:
    """
    Fetch cryptocurrency historical data from Binance exchange.

    Public API (no authentication required):
    - Historical OHLCV (klines/candlestick data)
    - 24h ticker statistics
    - Exchange info and trading pairs

    Rate Limits:
    - 1200 requests per minute (IP-based)
    - Much higher than CoinGecko!

    Example:
        >>> fetcher = BinanceFetcher()
        >>> # Get Bitcoin data
        >>> btc_data = fetcher.get_ohlcv('BTCUSDT', interval='1d', days=365)
        >>> print(btc_data.head())
        >>>
        >>> # Get multiple cryptocurrencies
        >>> cryptos = fetcher.get_multiple(['BTCUSDT', 'ETHUSDT'], days=90)
    """

    BASE_URL = "https://api.binance.com"
    BASE_URL_US = "https://api.binance.us"  # US-specific endpoint

    # Popular trading pairs (symbol: description)
    POPULAR_PAIRS = {
        'BTCUSDT': 'Bitcoin/USD Tether',
        'ETHUSDT': 'Ethereum/USD Tether',
        'BNBUSDT': 'Binance Coin/USD Tether',
        'SOLUSDT': 'Solana/USD Tether',
        'XRPUSDT': 'Ripple/USD Tether',
        'ADAUSDT': 'Cardano/USD Tether',
        'DOGEUSDT': 'Dogecoin/USD Tether',
        'MATICUSDT': 'Polygon/USD Tether',
        'DOTUSDT': 'Polkadot/USD Tether',
        'AVAXUSDT': 'Avalanche/USD Tether',
        'LINKUSDT': 'Chainlink/USD Tether',
        'ATOMUSDT': 'Cosmos/USD Tether',
        'LTCUSDT': 'Litecoin/USD Tether',
        'UNIUSDT': 'Uniswap/USD Tether',
    }

    # Available intervals
    INTERVALS = {
        '1m': '1 minute',
        '3m': '3 minutes',
        '5m': '5 minutes',
        '15m': '15 minutes',
        '30m': '30 minutes',
        '1h': '1 hour',
        '2h': '2 hours',
        '4h': '4 hours',
        '6h': '6 hours',
        '8h': '8 hours',
        '12h': '12 hours',
        '1d': '1 day',
        '3d': '3 days',
        '1w': '1 week',
        '1M': '1 month',
    }

    def __init__(
        self,
        rate_limit_delay: float = 0.1,
        verbose: bool = True,
    ):
        """
        Initialize Binance fetcher.

        Args:
            rate_limit_delay: Delay between API calls in seconds (default 0.1s)
            verbose: Print progress messages
        """
        self.rate_limit_delay = rate_limit_delay
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
            JSON response as dictionary or list

        Raises:
            requests.HTTPError: If request fails
        """
        self._rate_limit()

        url = f"{self.BASE_URL}{endpoint}"

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

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = '1d',
        days: Optional[int] = 365,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Get OHLCV (candlestick) data for a trading pair.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT')
                   Use uppercase! Find symbols with list_trading_pairs()
            interval: Candlestick interval (default '1d' for daily)
                     Options: '1m', '5m', '15m', '1h', '4h', '1d', '1w', etc.
                     See INTERVALS dict for all options
            days: Number of days of history (ignored if start_date provided)
            start_date: Start date (optional, overrides days parameter)
            end_date: End date (optional, defaults to now)
            limit: Max number of candles per request (max 1000)

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume

        Example:
            >>> # Last 365 days of daily data
            >>> btc = fetcher.get_ohlcv('BTCUSDT', interval='1d', days=365)
            >>>
            >>> # Specific date range
            >>> btc = fetcher.get_ohlcv(
            ...     'BTCUSDT',
            ...     interval='1d',
            ...     start_date=date(2020, 1, 1),
            ...     end_date=date(2023, 12, 31)
            ... )
            >>>
            >>> # 4-hour candles for last 30 days
            >>> btc = fetcher.get_ohlcv('BTCUSDT', interval='4h', days=30)
        """
        if self.verbose:
            print(f"Fetching {symbol} OHLCV data ({interval} interval)...")

        # Validate interval
        if interval not in self.INTERVALS:
            raise ValueError(
                f"Invalid interval: {interval}. "
                f"Options: {', '.join(self.INTERVALS.keys())}"
            )

        # Determine date range
        if end_date is None:
            end_date = datetime.now()
        else:
            end_date = datetime.combine(end_date, datetime.max.time())

        if start_date is not None:
            start_date = datetime.combine(start_date, datetime.min.time())
        else:
            start_date = end_date - timedelta(days=days)

        # Convert to millisecond timestamps (Binance requirement)
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)

        # Fetch data (may need multiple requests due to 1000 limit)
        all_candles = []
        current_start = start_ts

        while current_start < end_ts:
            endpoint = "/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_start,
                'endTime': end_ts,
                'limit': limit,
            }

            candles = self._make_request(endpoint, params)

            if not candles:
                break

            all_candles.extend(candles)

            # Update start time for next batch
            # Binance returns [open_time, open, high, low, close, volume, close_time, ...]
            last_close_time = candles[-1][6]  # close_time is 7th element
            current_start = last_close_time + 1

            if self.verbose and len(candles) == limit:
                print(f"  Fetched {len(all_candles)} candles so far...")

            # If we got less than limit, we're done
            if len(candles) < limit:
                break

        if not all_candles:
            raise ValueError(f"No data returned for {symbol}")

        # Parse response
        # Format: [open_time, open, high, low, close, volume, close_time, quote_volume, trades, ...]
        df = pd.DataFrame(all_candles, columns=[
            'open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])

        # Convert types
        df['Date'] = pd.to_datetime(df['open_time'], unit='ms')
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Close'] = df['Close'].astype(float)
        df['Volume'] = df['Volume'].astype(float)

        # Select and reorder columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.sort_values('Date').reset_index(drop=True)

        if self.verbose:
            print(f"  Retrieved {len(df)} candles")
            print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")

        return df

    def get_current_price(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current price for trading pairs.

        Args:
            symbols: List of trading pair symbols (e.g., ['BTCUSDT', 'ETHUSDT'])

        Returns:
            Dictionary mapping symbol to current price

        Example:
            >>> prices = fetcher.get_current_price(['BTCUSDT', 'ETHUSDT'])
            >>> print(f"BTC: ${prices['BTCUSDT']:,.2f}")
        """
        endpoint = "/api/v3/ticker/price"

        prices = {}
        for symbol in symbols:
            params = {'symbol': symbol}
            data = self._make_request(endpoint, params)
            prices[symbol] = float(data['price'])

        return prices

    def get_24h_stats(self, symbol: str) -> Dict:
        """
        Get 24-hour statistics for a trading pair.

        Args:
            symbol: Trading pair symbol

        Returns:
            Dictionary with 24h stats:
            - price_change: Absolute price change
            - price_change_percent: Percentage change
            - high: 24h high
            - low: 24h low
            - volume: 24h volume
            - quote_volume: 24h volume in quote asset

        Example:
            >>> stats = fetcher.get_24h_stats('BTCUSDT')
            >>> print(f"24h change: {stats['price_change_percent']:.2f}%")
        """
        endpoint = "/api/v3/ticker/24hr"
        params = {'symbol': symbol}

        data = self._make_request(endpoint, params)

        return {
            'symbol': data['symbol'],
            'price_change': float(data['priceChange']),
            'price_change_percent': float(data['priceChangePercent']),
            'weighted_avg_price': float(data['weightedAvgPrice']),
            'last_price': float(data['lastPrice']),
            'high': float(data['highPrice']),
            'low': float(data['lowPrice']),
            'volume': float(data['volume']),
            'quote_volume': float(data['quoteVolume']),
            'open_time': pd.to_datetime(data['openTime'], unit='ms'),
            'close_time': pd.to_datetime(data['closeTime'], unit='ms'),
            'trades': int(data['count']),
        }

    def get_multiple(
        self,
        symbols: List[str],
        interval: str = '1d',
        days: int = 365,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Get OHLCV data for multiple trading pairs.

        Args:
            symbols: List of trading pair symbols
            interval: Candlestick interval
            days: Number of days of history
            start_date: Start date (optional)
            end_date: End date (optional)

        Returns:
            Dictionary mapping symbol to DataFrame

        Example:
            >>> cryptos = fetcher.get_multiple(
            ...     ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
            ...     interval='1d',
            ...     days=180
            ... )
            >>> btc_data = cryptos['BTCUSDT']
            >>> eth_data = cryptos['ETHUSDT']
        """
        results = {}

        if self.verbose:
            print(f"Fetching data for {len(symbols)} trading pairs...")

        for symbol in symbols:
            try:
                df = self.get_ohlcv(
                    symbol,
                    interval=interval,
                    days=days,
                    start_date=start_date,
                    end_date=end_date,
                )
                results[symbol] = df
            except Exception as e:
                warnings.warn(f"Failed to fetch {symbol}: {e}")
                results[symbol] = None

        return results

    def list_trading_pairs(self, quote_asset: str = 'USDT') -> pd.DataFrame:
        """
        List all available trading pairs for a quote asset.

        Args:
            quote_asset: Quote asset (default 'USDT')
                        Other options: 'BUSD', 'BTC', 'ETH', 'BNB'

        Returns:
            DataFrame with trading pair information

        Example:
            >>> pairs = fetcher.list_trading_pairs('USDT')
            >>> print(pairs.head(20))
        """
        endpoint = "/api/v3/exchangeInfo"

        data = self._make_request(endpoint)

        # Filter trading pairs
        pairs = []
        for symbol_info in data['symbols']:
            symbol = symbol_info['symbol']
            if symbol.endswith(quote_asset) and symbol_info['status'] == 'TRADING':
                pairs.append({
                    'symbol': symbol,
                    'base_asset': symbol_info['baseAsset'],
                    'quote_asset': symbol_info['quoteAsset'],
                    'status': symbol_info['status'],
                })

        return pd.DataFrame(pairs)

    def search_symbol(self, query: str) -> List[str]:
        """
        Search for trading pairs by base asset name.

        Args:
            query: Search term (e.g., 'BTC', 'ETH', 'SOL')

        Returns:
            List of matching symbols

        Example:
            >>> fetcher.search_symbol('BTC')
            ['BTCUSDT', 'BTCBUSD', 'BTCEUR', ...]
        """
        query_upper = query.upper()

        pairs = self.list_trading_pairs('USDT')
        matches = pairs[pairs['symbol'].str.contains(query_upper)]['symbol'].tolist()

        # Also check popular pairs
        if query_upper in ['BTC', 'BITCOIN']:
            matches.insert(0, 'BTCUSDT')
        elif query_upper in ['ETH', 'ETHEREUM']:
            matches.insert(0, 'ETHUSDT')

        return list(dict.fromkeys(matches))  # Remove duplicates, preserve order

    def list_popular_pairs(self) -> pd.DataFrame:
        """
        List popular cryptocurrency trading pairs.

        Returns:
            DataFrame with symbol and description

        Example:
            >>> pairs = fetcher.list_popular_pairs()
            >>> print(pairs)
        """
        pairs = []
        for symbol, description in self.POPULAR_PAIRS.items():
            pairs.append({
                'symbol': symbol,
                'description': description,
            })

        return pd.DataFrame(pairs)


def download_bitcoin(
    interval: str = '1d',
    days: int = 365,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Convenience function to download Bitcoin data from Binance.

    Args:
        interval: Data interval (default '1d' for daily)
        days: Number of days (default 365)
        start_date: Optional start date
        end_date: Optional end date

    Returns:
        DataFrame with Bitcoin OHLCV data

    Example:
        >>> from calibration.data import download_bitcoin
        >>> btc = download_bitcoin(interval='1d', days=730)  # 2 years
        >>> btc = download_bitcoin(start_date=date(2020, 1, 1))
    """
    fetcher = BinanceFetcher()
    return fetcher.get_ohlcv(
        'BTCUSDT',
        interval=interval,
        days=days,
        start_date=start_date,
        end_date=end_date,
    )


def download_crypto_basket(
    symbols: List[str] = None,
    interval: str = '1d',
    days: int = 365,
) -> Dict[str, pd.DataFrame]:
    """
    Download data for a basket of cryptocurrencies from Binance.

    Args:
        symbols: List of symbols (default: BTC, ETH, BNB, SOL)
        interval: Data interval
        days: Number of days

    Returns:
        Dictionary mapping symbol to DataFrame

    Example:
        >>> basket = download_crypto_basket(
        ...     symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
        ...     days=180
        ... )
        >>> btc = basket['BTCUSDT']
    """
    if symbols is None:
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']

    fetcher = BinanceFetcher()
    return fetcher.get_multiple(symbols, interval=interval, days=days)
