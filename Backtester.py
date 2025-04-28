import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Backtester:
    def __init__(self, portfolio_data, price_data, initial_cash=10000):
        self.signals = portfolio_data.copy()
        self.price_data = price_data.copy()
        self.initial_cash = initial_cash
        self.portfolio_value = self._calculate_portfolio_value()

    def _calculate_portfolio_value(self):
        adjusted_signals = self.signals.shift(1).fillna(0)  # Shift signals to simulate next day execution
        portfolio_value = (adjusted_signals * self.price_data).sum(axis=1)
        return self.initial_cash + portfolio_value

    def calculate_metrics(self):
        final_value = self.portfolio_value.iloc[-1]
        cumulative_return = (final_value - self.initial_cash) / self.initial_cash
        running_max = self.portfolio_value.cummax()
        drawdown = (running_max - self.portfolio_value) / running_max
        max_drawdown = drawdown.max()
        daily_returns = self.portfolio_value.pct_change().dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        return {
            "Cumulative Return": cumulative_return,
            "Max Drawdown": max_drawdown,
            "Sharpe Ratio": sharpe_ratio
        }

    def plot_results(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.portfolio_value.index, self.portfolio_value, label="Portfolio Value", color='blue')
        plt.title("Backtest Portfolio Performance")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid()
        plt.show()



