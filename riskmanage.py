import logging
import numpy as np
import pandas as pd
import MeanVarianceOptimizer
from scipy.optimize import minimize

class RiskManager:
    def __init__(self, portfolio_value, risk_tolerance=0.02):
        if portfolio_value <= 0:
            raise ValueError("Portfolio value must be greater than 0.")
        if not (0 < risk_tolerance <= 1):
            raise ValueError("Risk tolerance must be between 0 and 1.")
        self.portfolio_value = portfolio_value
        self.risk_tolerance = risk_tolerance

    def calculate_position_size(self, stop_loss_price, entry_price):
        if stop_loss_price <= 0 or entry_price <= 0:
            raise ValueError("Prices must be greater than 0.")
        if stop_loss_price >= entry_price:
            raise ValueError("Stop-loss price must be less than the entry price.")
        
        risk_per_share = abs(entry_price - stop_loss_price)
        max_risk = self.portfolio_value * self.risk_tolerance
        position_size = max_risk / risk_per_share
        logging.info(f"Calculated position size: {position_size} shares (Max risk: {max_risk}, Risk per share: {risk_per_share})")
        return int(position_size)
    
    def set_risk_tolerance(self, new_risk_tolerance):
        if not (0 < new_risk_tolerance <= 1):
            raise ValueError("Risk tolerance must be between 0 and 1.")
        self.risk_tolerance = new_risk_tolerance
        logging.info(f"Risk tolerance updated to {new_risk_tolerance}.")
    
    def apply_stop_loss(self, current_price, stop_loss_price):
        """
        Checks if the stop-loss condition is met.

        :param current_price: The current price of the asset.
        :param stop_loss_price: The stop-loss price.
        :return: True if stop-loss condition is met, False otherwise.
        """
        return current_price <= stop_loss_price

    def apply_trailing_stop_loss(self, current_price, highest_price, trailing_percentage):
        """
        Applies a trailing stop-loss mechanism.
        :param current_price: The current price of the asset.
        :param highest_price: The highest price reached since the position was opened.
        :param trailing_percentage: The trailing stop percentage (e.g., 0.05 for 5%).
        :return: True if the trailing stop-loss condition is met, False otherwise.
        """
        trailing_stop_price = highest_price * (1 - trailing_percentage)
        if current_price <= trailing_stop_price:
            logging.info(f"Trailing stop-loss triggered. Current price: {current_price}, Trailing stop price: {trailing_stop_price}")
            return True
        return False

    def optimize_portfolio(self, stock_returns, target_return=None, shorting=False):
        optimizer = MeanVarianceOptimizer(stock_returns)
        if target_return:
            weights = optimizer.get_efficient_portfolio(target_return)
        else:
            weights = optimizer.get_min_variance_portfolio()

        expected_return = np.dot(weights, stock_returns.mean() * 52)  # Annualized return
        volatility = np.sqrt(np.dot(weights.T, np.dot(stock_returns.cov() * 52, weights)))  # Annualized volatility

        return {
        "weights": weights,
        "expected_return": expected_return,
        "volatility": volatility
        }

    @staticmethod
    def calculate_var_cvar(returns, confidence_level=0.05):
        """
        Calculates Value at Risk (VaR) and Conditional Value at Risk (CVaR) for a given set of returns.
        """
        if not isinstance(returns, pd.Series):
            raise TypeError("Returns must be a Pandas Series.")
        if not (0 < confidence_level < 1):
            raise ValueError("Confidence level must be between 0 and 1.")
        if returns.empty:
            raise ValueError("Returns series is empty.")

        sorted_returns = returns.sort_values()
        index = int(confidence_level * len(sorted_returns))
        if index == 0:
            raise ValueError("Confidence level too low for the given data size.")

        var = sorted_returns.iloc[index]
        cvar = sorted_returns.iloc[:index].mean()

        logging.info(f"Calculated VaR: {var}, CVaR: {cvar} at confidence level: {confidence_level}")
        return var, cvar
    
    
    def calculate_max_drawdown(self, portfolio_values):
        """
        Calculates the maximum drawdown of a portfolio.
        :param portfolio_values: Pandas Series of portfolio values over time.
        :return: Maximum drawdown as a percentage.
        """
        rolling_max = portfolio_values.cummax()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        return max_drawdown
    