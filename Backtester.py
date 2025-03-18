import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Backtester:
    def __init__(self, portfolio_data, price_data, initial_cash=10000):
        """
        Initialize backtester with dynamic stock prices
        :param portfolio_data: DataFrame from PositionTracker (holding quantities)
        :param price_data: DataFrame with historical stock prices
        :param initial_cash: Initial capital
        """
        self.df = portfolio_data
        self.price_data = price_data
        self.initial_cash = initial_cash
        self.df["Portfolio Value"] = self._calculate_portfolio_value()

    def _calculate_portfolio_value(self):
        """Calculate portfolio value using actual stock prices"""
        portfolio_value = (self.df * self.price_data).sum(axis=1)
        return self.initial_cash + portfolio_value

    def calculate_metrics(self):
        """Calculate performance metrics"""
        final_value = self.df["Portfolio Value"].iloc[-1]
        cumulative_return = (final_value - self.initial_cash) / self.initial_cash

        # Maximum drawdown
        running_max = self.df["Portfolio Value"].cummax()
        drawdown = (running_max - self.df["Portfolio Value"]) / running_max
        max_drawdown = drawdown.max()

        # Sharpe ratio
        daily_returns = self.df["Portfolio Value"].pct_change().dropna()
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)  # 252 為交易日數

        return {
            "Cumulative Return": cumulative_return, 
            "Max Drawdown": max_drawdown, 
            "Sharpe Ratio": sharpe_ratio
        }

    def plot_results(self):
        """Visualize backtest results"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.df.index, self.df["Portfolio Value"], label="Portfolio Value", color='blue')
        plt.title("Backtest Portfolio Performance")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid()
        plt.show()