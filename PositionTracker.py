import pandas as pd
from datetime import datetime

class PositionTracker:
    def __init__(self, initial_positions=None):
        """
        Initialize portfolio with initial positions
        
        Parameters:
        - initial_positions: List of dicts [{'date': str, 'ticker': str, 'quantity': int}]
        """
        self.current_holdings = {}
        self.history = pd.DataFrame()
        
        if initial_positions:
            for position in initial_positions:
                self._process_transaction(
                    date=position['date'],
                    ticker=position['ticker'],
                    delta=position['quantity']
                )

    def _process_transaction(self, date, ticker, delta):
        """Update current holdings and maintain transaction history"""
        # Update current positions
        self.current_holdings[ticker] = self.current_holdings.get(ticker, 0) + delta
        
        # Remove positions with zero shares
        if self.current_holdings[ticker] == 0:
            del self.current_holdings[ticker]
        
        # Create new history entry
        new_entry = pd.DataFrame(
            [self.current_holdings],
            index=[pd.to_datetime(date)]
        )
        
        # Update main dataframe
        self.history = pd.concat([self.history, new_entry]).sort_index()
        
        # Forward fill missing values between dates
        self.history = self.history.reindex(
            pd.date_range(start=self.history.index.min(),
                          end=self.history.index.max(),
                          freq='D')
        ).ffill().fillna(0)

    def add_transaction(self, date, ticker, shares):
        """
        Add a new position change transaction
        
        Parameters:
        - date: str (YYYY-MM-DD)
        - ticker: str (stock symbol)
        - shares: int (+/- for buy/sell)
        """
        self._process_transaction(date, ticker, shares)

    def get_portfolio(self, resample_freq=None):
        """
        Get formatted portfolio dataframe
        
        Parameters:
        - resample_freq: None or pandas frequency string (e.g., 'B', 'W-FRI')
        """
        df = self.history.copy()
        if resample_freq:
            return df.resample(resample_freq).last().ffill().fillna(0)
        return df

# Example Usage
if __name__ == "__main__":
    # Initialize with starting positions
    portfolio = PositionTracker([
        {'date': '2023-01-01', 'ticker': 'AAPL', 'quantity': 100},
        {'date': '2023-01-01', 'ticker': 'MSFT', 'quantity': 50}
    ])

    # Add subsequent transactions
    portfolio.add_transaction('2023-01-05', 'AAPL', 20)
    portfolio.add_transaction('2023-01-05', 'MSFT', -10)
    portfolio.add_transaction('2023-01-10', 'GOOG', 75)

    # Get full transaction history
    print("Detailed Transaction History:")
    print(portfolio.get_portfolio())

    # Get weekly Friday portfolio snapshot
    print("\nWeekly Portfolio Snapshots:")
    print(portfolio.get_portfolio(resample_freq='W-FRI'))
