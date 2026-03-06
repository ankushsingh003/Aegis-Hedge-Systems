import yfinance as yf
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class DataProvider(ABC):
    @abstractmethod
    def get_spot_price(self, ticker: str) -> float:
        pass
    
    @abstractmethod
    def get_historical_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        pass

class YFinanceProvider(DataProvider):
    def get_spot_price(self, ticker: str) -> float:
        data = yf.Ticker(ticker)
        history = data.history(period="1d")
        if history.empty:
            raise ValueError(f"Could not fetch data for ticker: {ticker}")
        return float(history['Close'].iloc[-1])
    
    def get_historical_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        data = yf.Ticker(ticker)
        df = data.history(period=period)
        if df.empty:
            raise ValueError(f"No historical data for ticker: {ticker}")
        return df

    def estimate_volatility(self, ticker: str, days: int = 252) -> float:
        df = self.get_historical_data(ticker, period=f"{days}d")
        log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        return float(log_returns.std() * np.sqrt(252))
