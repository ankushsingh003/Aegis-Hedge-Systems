import yfinance as yf
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class DataProvider(ABC):
    """Abstract base class for financial data providers."""
    
    @abstractmethod
    def get_spot_price(self, ticker: str) -> float:
        pass
    
    @abstractmethod
    def get_historical_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        pass

class YFinanceProvider(DataProvider):
    """Data provider using the Yahoo Finance API."""
    
    def get_spot_price(self, ticker: str) -> float:
        """Fetches the latest closing price for a ticker."""
        data = yf.Ticker(ticker)
        # Using fast_info or history(period="1d")
        history = data.history(period="1d")
        if history.empty:
            raise ValueError(f"Could not fetch data for ticker: {ticker}")
        return float(history['Close'].iloc[-1])
    
    def get_historical_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Fetches historical price data."""
        data = yf.Ticker(ticker)
        df = data.history(period=period)
        if df.empty:
            raise ValueError(f"No historical data for ticker: {ticker}")
        return df

    def estimate_volatility(self, ticker: str, days: int = 252) -> float:
        """Estimates annualized historical volatility."""
        df = self.get_historical_data(ticker, period=f"{days}d")
        log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        # Annualized volatility = daily_std * sqrt(252)
        return float(log_returns.std() * np.sqrt(252))
