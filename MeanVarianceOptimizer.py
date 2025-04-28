from scipy.optimize import minimize
import numpy as np
import pandas as pd
from typing import Tuple, List

class MeanVarianceOptimizer:
    def __init__(self, stock_returns):
        """
        Args:
            stock_returns: DataFrame of weekly returns (30 columns = stocks, rows = weeks)
        """
        # Strict 30-stock validation
        if len(stock_returns.columns) != 30:
            raise ValueError(f"Must have exactly 30 stocks. Got {len(stock_returns.columns)}")
            
        self.returns = stock_returns
        self.n_assets = 30
        
        # Compute key parameters
        self.mean_returns = self.returns.mean() * 52  # Annualized
        self.cov_matrix = self.returns.cov() * 52     # Annualized

    def optimize(self, target_return=None, shorting=False):
        """
        Optimize portfolio weights for:
        - Minimum variance (default)
        - Target return (if specified)
        
        Args:
            target_return: Annualized target return (e.g., 0.12 for 12%)
            shorting: Allow negative weights if True
            
        Returns:
            weights: Optimal portfolio weights (30 elements)
            performance: Tuple of (return, volatility)
        """
        # Constraints and bounds
        bounds = [(-1, 1) if shorting else (0, 1) for _ in range(30)]
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.dot(x, self.mean_returns) - target_return
            })

        # Objective function: minimize volatility
        def objective(weights):
            return weights.T @ self.cov_matrix @ weights

        # Optimization
        result = minimize(
            objective,
            x0=np.ones(30)/30,  # Equal initial weights
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
            
        weights = result.x
        ret = np.dot(weights, self.mean_returns)
        vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
        
        return weights, (ret, vol)

    def get_min_variance_portfolio(self):
        """Get minimum volatility portfolio"""
        return self.optimize()[0]

    def get_efficient_portfolio(self, target_return):
        """Get portfolio with specific target return"""
        return self.optimize(target_return=target_return)[0]