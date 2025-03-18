import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class TradingStrategy:
    def __init__(self, data):
        self.data = data
        self.indicators = {}

    def add_indicator(self, indicator_name, indicator_function, **kwargs):
        self.indicators[indicator_name] = indicator_function(self.data, **kwargs)

    def generate_signals(self):
        raise NotImplementedError

    def optimize_parameters(self, parameter_grid):
        raise NotImplementedError


# Example Technical Indicator: Moving Average
def moving_average(data, window):
    return data['Close'].rolling(window=window).mean()


# Example Strategy: Simple Moving Average Crossover
class SMACrossoverStrategy(TradingStrategy):
    def __init__(self, data, short_window, long_window):
        super().__init__(data)
        self.short_window = short_window
        self.long_window = long_window
        self.add_indicator('short_ma', moving_average, window=short_window)
        self.add_indicator('long_ma', moving_average, window=long_window)

    def generate_signals(self):
        signals = pd.DataFrame(index=self.data.index)
        signals['signal'] = 0.0
        signals['signal'][self.indicators['short_ma'] > self.indicators['long_ma']] = 1.0
        signals['signal'][self.indicators['short_ma'] < self.indicators['long_ma']] = -1.0
        return signals


# Example Machine Learning Strategy: Linear Regression for Price Prediction
class LinearRegressionStrategy(TradingStrategy):
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        # Prepare data for model training
        X = self.data[['Open', 'High', 'Low', 'Volume']]
        y = self.data['Close']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Generate predictions
        predictions = model.predict(X)

        # Generate signals based on predictions
        signals = pd.DataFrame(index=self.data.index)
        signals['signal'] = 0.0
        signals['signal'][predictions > self.data['Close']] = 1.0
        signals['signal'][predictions < self.data['Close']] = -1.0

        return signals


# Example Optimization: Grid Search for SMACrossover Strategy
def optimize_sma_crossover(data, short_window_range, long_window_range):
    best_performance = float('-inf')
    best_parameters = None

    for short_window in short_window_range:
        for long_window in long_window_range:
            strategy = SMACrossoverStrategy(data, short_window, long_window)
            signals = strategy.generate_signals()
            # Calculate performance metric (e.g., Sharpe Ratio, Profit Factor)
            # ...
            # if performance > best_performance:
            #     best_performance = performance
            #     best_parameters = (short_window, long_window)

    return best_parameters


# ... (More strategies, indicators, and optimization techniques can be added)