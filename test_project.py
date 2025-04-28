import unittest
import pandas as pd
import numpy as np
from riskmanage import RiskManager
from MeanVarianceOptimizer import MeanVarianceOptimizer
from PositionTracker import PositionTracker
from TradingStrategy import (
    SMACrossoverStrategy, RSIStrategy, BollingerBandsStrategy, 
    ATRBreakoutStrategy, OBVStrategy, ADIStrategy, calculate_sharpe_ratio
)

class TestProject(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        self.data = pd.DataFrame({
            "Open": np.random.uniform(100, 200, len(dates)),
            "High": np.random.uniform(200, 300, len(dates)),
            "Low": np.random.uniform(50, 100, len(dates)),
            "Close": np.random.uniform(100, 200, len(dates)),
            "Volume": np.random.randint(1000, 5000, len(dates))
        }, index=dates)

        self.returns = pd.Series(np.random.uniform(-0.05, 0.05, len(dates)), index=dates)

    # Test RiskManager
    def test_risk_manager(self):
        rm = RiskManager(portfolio_value=100000, risk_tolerance=0.02)
        position_size = rm.calculate_position_size(stop_loss_price=95, entry_price=100)
        self.assertEqual(position_size, 400)  # Expected position size

        self.assertTrue(rm.apply_stop_loss(current_price=90, stop_loss_price=95))
        self.assertFalse(rm.apply_stop_loss(current_price=100, stop_loss_price=95))

        var, cvar = rm.calculate_var_cvar(self.returns, confidence_level=0.05)
        self.assertIsInstance(var, float)
        self.assertIsInstance(cvar, float)

    # Test MeanVarianceOptimizer
    def test_mean_variance_optimizer(self):
        stock_returns = pd.DataFrame(np.random.uniform(-0.05, 0.05, (52, 30)))
        optimizer = MeanVarianceOptimizer(stock_returns)
        weights, performance = optimizer.optimize()
        self.assertEqual(len(weights), 30)
        self.assertIsInstance(performance, tuple)

    # Test PositionTracker
    def test_position_tracker(self):
        tracker = PositionTracker([
            {"date": "2023-01-01", "ticker": "AAPL", "quantity": 100},
            {"date": "2023-01-01", "ticker": "MSFT", "quantity": 50}
        ])
        tracker.add_transaction("2023-01-05", "AAPL", 20)
        tracker.add_transaction("2023-01-05", "MSFT", -10)
        portfolio = tracker.get_portfolio()
        self.assertIn("AAPL", portfolio.columns)
        self.assertIn("MSFT", portfolio.columns)

    # Test Trading Strategies
    def test_trading_strategies(self):
        # SMA Crossover Strategy
        sma_strategy = SMACrossoverStrategy(self.data, short_window=3, long_window=5)
        sma_signals = sma_strategy.generate_signals()
        self.assertIn("signal", sma_signals.columns)

        # RSI Strategy
        rsi_strategy = RSIStrategy(self.data, rsi_window=14, overbought=70, oversold=30)
        rsi_signals = rsi_strategy.generate_signals()
        self.assertIn("signal", rsi_signals.columns)

        # Bollinger Bands Strategy
        bb_strategy = BollingerBandsStrategy(self.data, window=20, num_std_dev=2)
        bb_signals = bb_strategy.generate_signals()
        self.assertIn("signal", bb_signals.columns)

        # ATR Breakout Strategy
        atr_strategy = ATRBreakoutStrategy(self.data, atr_window=14, atr_multiplier=2)
        atr_signals = atr_strategy.generate_signals()
        self.assertIn("signal", atr_signals.columns)

        # OBV Strategy
        obv_strategy = OBVStrategy(self.data)
        obv_signals = obv_strategy.generate_signals()
        self.assertIn("signal", obv_signals.columns)

        # ADI Strategy
        adi_strategy = ADIStrategy(self.data)
        adi_signals = adi_strategy.generate_signals()
        self.assertIn("signal", adi_signals.columns)

    # Test Sharpe Ratio Calculation
    def test_sharpe_ratio(self):
        sharpe_ratio = calculate_sharpe_ratio(self.returns, risk_free_rate=0.01)
        self.assertIsInstance(sharpe_ratio, float)

if __name__ == "__main__":
    unittest.main()