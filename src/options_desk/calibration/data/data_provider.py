"""
Data provider interfaces and data structures for calibration.

author: Yunian Pan
email: yp1170@nyu.edu
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import numpy as np
import pandas as pd


@dataclass
class OptionQuote:
    """Single option quote with market data."""
    strike: float
    expiry: date
    option_type: str  # 'call' or 'put'
    bid: float
    ask: float
    mid: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: Optional[float] = None
    
    @property
    def is_call(self) -> bool:
        return self.option_type.lower() == 'call'
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid
    
    @property
    def relative_spread(self) -> float:
        if self.mid > 0:
            return self.spread / self.mid
        return float('inf')


@dataclass
class OptionChain:
    """Option chain for a single underlying at a specific date."""
    underlying: str
    spot_price: float
    reference_date: date
    risk_free_rate: float
    dividend_yield: float
    options: List[OptionQuote] = field(default_factory=list)
    
    def to_dataframe(self) -> pd.DataFrame:
        if not self.options:
            return pd.DataFrame()
        
        data = [{
            'strike': o.strike,
            'expiry': o.expiry,
            'option_type': o.option_type,
            'bid': o.bid,
            'ask': o.ask,
            'mid': o.mid,
            'last': o.last,
            'volume': o.volume,
            'open_interest': o.open_interest,
            'implied_volatility': o.implied_volatility,
            'moneyness': o.strike / self.spot_price,
            'time_to_expiry': (o.expiry - self.reference_date).days / 365.0,
        } for o in self.options]
        
        return pd.DataFrame(data)
    
    def filter(
        self,
        min_volume: int = 0,
        min_open_interest: int = 0,
        max_spread_pct: float = 1.0,
        moneyness_range: tuple = (0.8, 1.2),
        option_type: Optional[str] = None,
    ) -> 'OptionChain':
        filtered = []
        for opt in self.options:
            moneyness = opt.strike / self.spot_price
            if opt.volume < min_volume:
                continue
            if opt.open_interest < min_open_interest:
                continue
            if opt.relative_spread > max_spread_pct:
                continue
            if not (moneyness_range[0] <= moneyness <= moneyness_range[1]):
                continue
            if option_type and opt.option_type.lower() != option_type.lower():
                continue
            filtered.append(opt)
        
        return OptionChain(
            underlying=self.underlying,
            spot_price=self.spot_price,
            reference_date=self.reference_date,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            options=filtered,
        )
    
    def get_expiries(self) -> List[date]:
        return sorted(set(o.expiry for o in self.options))
    
    def get_strikes(self, expiry: Optional[date] = None) -> List[float]:
        if expiry:
            strikes = [o.strike for o in self.options if o.expiry == expiry]
        else:
            strikes = [o.strike for o in self.options]
        return sorted(set(strikes))
    
    def get_slice(self, expiry: date, option_type: str = 'call') -> pd.DataFrame:
        slice_opts = [
            o for o in self.options 
            if o.expiry == expiry and o.option_type.lower() == option_type.lower()
        ]
        if not slice_opts:
            return pd.DataFrame()
        
        df = pd.DataFrame([{
            'strike': o.strike,
            'moneyness': o.strike / self.spot_price,
            'log_moneyness': np.log(o.strike / self.spot_price),
            'mid_price': o.mid,
            'implied_volatility': o.implied_volatility,
            'bid': o.bid,
            'ask': o.ask,
            'volume': o.volume,
            'open_interest': o.open_interest,
        } for o in slice_opts])
        
        return df.sort_values('strike').reset_index(drop=True)


@dataclass
class MarketData:
    """Container for all market data needed for calibration."""
    option_chain: OptionChain
    spot_history: Optional[pd.DataFrame] = None
    
    @property
    def spot(self) -> float:
        return self.option_chain.spot_price
    
    @property
    def underlying(self) -> str:
        return self.option_chain.underlying
    
    def get_historical_returns(self, log: bool = True) -> np.ndarray:
        if self.spot_history is None:
            raise ValueError("No spot history available")
        prices = self.spot_history['Close'].values
        if log:
            return np.diff(np.log(prices))
        return np.diff(prices) / prices[:-1]
    
    def get_realized_variance(self, window: int = 252) -> float:
        returns = self.get_historical_returns(log=True)
        if len(returns) < window:
            window = len(returns)
        recent_returns = returns[-window:]
        return np.var(recent_returns, ddof=1) * 252


class DataProvider(ABC):
    """Abstract base class for market data providers."""
    
    @abstractmethod
    def get_spot(self, ticker: str) -> float:
        pass
    
    @abstractmethod
    def get_history(
        self, 
        ticker: str, 
        start: date, 
        end: date,
    ) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def get_option_chain(
        self, 
        ticker: str,
        reference_date: Optional[date] = None,
    ) -> OptionChain:
        pass
    
    def get_market_data(
        self,
        ticker: str,
        history_days: int = 252,
    ) -> MarketData:
        """Convenience method to get all data for calibration."""
        from datetime import timedelta
        
        today = date.today()
        start = today - timedelta(days=history_days + 30)
        
        option_chain = self.get_option_chain(ticker)
        spot_history = self.get_history(ticker, start, today)
        
        return MarketData(
            option_chain=option_chain,
            spot_history=spot_history,
        )
