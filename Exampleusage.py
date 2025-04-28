import pandas as pd
from TradingStrategy import RSIStrategy
from riskmanage import RiskManager
from PositionTracker import PositionTracker
from Backtester import Backtester

# 1. Load market data
data = pd.read_csv('nvda_data.csv', index_col='Date', parse_dates=True)

# 2. Generate signals
strategy = RSIStrategy(data)
signals = strategy.generate_signals()

# 3. Initialize risk manager and position tracker
risk_manager = RiskManager(portfolio_value=100_000)
tracker = PositionTracker()

# 4. Loop through signals to simulate trades
for date, signal in signals.items():
    close_price = data.loc[date, 'Close']

    if signal == 1.0:  # Buy signal
        stop_loss_price = close_price * 0.98  # Example stop-loss
        size = risk_manager.calculate_position_size(stop_loss_price, close_price)
        tracker.add_transaction(date.strftime('%Y-%m-%d'), 'NVDA', size)

    elif signal == -1.0:  # Sell signal
        tracker.add_transaction(date.strftime('%Y-%m-%d'), 'NVDA', -tracker.current_holdings.get('NVDA', 0))

# 5. Portfolio backtesting
portfolio_data = tracker.get_portfolio()
price_data = data[['Close']].rename(columns={'Close': 'NVDA'})  # Align with tracker
backtester = Backtester(portfolio_data, price_data)
metrics = backtester.calculate_metrics()

# 6. Risk metrics (VaR/CVaR)
returns = portfolio_data['NVDA'].pct_change().dropna()
var, cvar = risk_manager.calculate_var_cvar(returns)

print(metrics)
print(f"VaR: {var:.4f}, CVaR: {cvar:.4f}")
