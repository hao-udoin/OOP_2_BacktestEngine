import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import logging
logging.basicConfig(level=logging.INFO)

class TradingStrategy:
    def __init__(self, data):
        self.data = data
        self.indicators = {}
        self.cached_indicators = {}

    def add_indicator(self, indicator_name, indicator_function, **kwargs):
        if indicator_name not in self.cached_indicators:
            self.cached_indicators[indicator_name] = indicator_function(self.data, **kwargs)
        self.indicators[indicator_name] = self.cached_indicators[indicator_name]

    def generate_signal(self, condition_buy, condition_sell):
        signals = pd.Series(0.0, index=self.data.index)
        signals[condition_buy] = 1.0
        signals[condition_sell] = -1.0
        return signals

    def optimize_parameters(self, parameter_grid):
        raise NotImplementedError

def validate_columns(data, required_columns):
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

# Indicator functions
def relative_strength_index(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def simple_moving_average(data, window=20):
    return data['Close'].rolling(window=window).mean()

def stochastic_oscillator(data, window=14):
    lowest_low = data['Low'].rolling(window=window).min()
    highest_high = data['High'].rolling(window=window).max()
    return 100 * (data['Close'] - lowest_low) / (highest_high - lowest_low)

def moving_average_convergence_divergence(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window).mean()
    long_ema = data['Close'].ewm(span=long_window).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window).mean()
    return pd.DataFrame({'MACD': macd, 'Signal': signal})

def bollinger_bands(data, window=20, num_std_dev=2):
    ma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    return pd.DataFrame({
        'Upper Band': ma + num_std_dev * std,
        'Lower Band': ma - num_std_dev * std,
        'Moving Average': ma
    })

def average_true_range(data, window=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def on_balance_volume(data):
    return (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()

def accumulation_distribution_index(data):
    mfm = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
    mfv = mfm * data['Volume']
    return mfv.cumsum()

# Strategies
class SMACrossoverStrategy(TradingStrategy):
    def __init__(self, data, short_window=5, long_window=20):
        super().__init__(data)
        self.add_indicator('short_ma', simple_moving_average, window=short_window)
        self.add_indicator('long_ma', simple_moving_average, window=long_window)

    def generate_signals(self):
        short_ma = self.indicators['short_ma']
        long_ma = self.indicators['long_ma']
        signals = self.generate_signal(short_ma > long_ma, short_ma < long_ma)
        signals = pd.DataFrame({'signal': signals})
        logging.info(f"SMA signals: {signals['signal'].value_counts().to_dict()}")
        return signals

class LinearRegressionStrategy(TradingStrategy):
    def __init__(self, data):
        super().__init__(data)
        validate_columns(data, ['Open', 'High', 'Low', 'Volume', 'Close'])
        self.data = self.data.ffill()

    def generate_signals(self):
        X = self.data[['Open', 'High', 'Low', 'Volume']]
        y = self.data['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        signals = pd.Series(0.0, index=X_test.index)
        signals[predictions > y_test] = 1.0
        signals[predictions < y_test] = -1.0
        signals = pd.DataFrame({'signal': signals})
        logging.info(f"Linear Regression signals: {signals['signal'].value_counts().to_dict()}")
        return signals

class RSIStrategy(TradingStrategy):
    def __init__(self, data, rsi_window=14, overbought=60, oversold=40):
        super().__init__(data)
        self.add_indicator('RSI', relative_strength_index, window=rsi_window)
        self.overbought = overbought
        self.oversold = oversold

    def generate_signals(self):
        rsi = self.indicators['RSI']
        signals = self.generate_signal(rsi < self.oversold, rsi > self.overbought)
        signals = pd.DataFrame({'signal': signals})
        logging.info(f"RSI signals: {signals['signal'].value_counts().to_dict()}")
        return signals

class StochasticOscillatorStrategy(TradingStrategy):
    def __init__(self, data, window=14, overbought=70, oversold=30):
        super().__init__(data)
        validate_columns(data, ['High', 'Low', 'Close'])
        self.add_indicator('Stochastic', stochastic_oscillator, window=window)
        self.overbought = overbought
        self.oversold = oversold

    def generate_signals(self):
        stoch = self.indicators['Stochastic']
        signals = self.generate_signal(stoch < self.oversold, stoch > self.overbought)
        signals = pd.DataFrame({'signal': signals})
        logging.info(f"Stochastic signals: {signals['signal'].value_counts().to_dict()}")
        return signals

class MACDStrategy(TradingStrategy):
    def __init__(self, data, short_window=12, long_window=26, signal_window=9):
        super().__init__(data)
        validate_columns(data, ['Close'])
        self.add_indicator('MACD', moving_average_convergence_divergence,
                           short_window=short_window, long_window=long_window, signal_window=signal_window)

    def generate_signals(self):
        macd = self.indicators['MACD']['MACD']
        signal = self.indicators['MACD']['Signal']
        signals = self.generate_signal(macd > signal, macd < signal)
        signals = pd.DataFrame({'signal': signals})
        logging.info(f"MACD signals: {signals['signal'].value_counts().to_dict()}")
        return signals

class BollingerBandsStrategy(TradingStrategy):
    def __init__(self, data, window=20, num_std_dev=2):
        super().__init__(data)
        validate_columns(data, ['Close'])
        self.add_indicator('BollingerBands', bollinger_bands, window=window, num_std_dev=num_std_dev)

    def generate_signals(self):
        bb = self.indicators['BollingerBands']
        close = self.data['Close']
        signals = self.generate_signal(close < bb['Lower Band'], close > bb['Upper Band'])
        signals = pd.DataFrame({'signal': signals})
        logging.info(f"Bollinger Bands signals: {signals['signal'].value_counts().to_dict()}")
        return signals

class ATRBreakoutStrategy(TradingStrategy):
    def __init__(self, data, atr_window=14, atr_multiplier=1.5):
        super().__init__(data)
        validate_columns(data, ['High', 'Low', 'Close'])
        self.add_indicator('ATR', average_true_range, window=atr_window)
        self.atr_multiplier = atr_multiplier

    def generate_signals(self):
        atr = self.indicators['ATR']
        close = self.data['Close']
        signals = self.generate_signal(
            close > close.shift(1) + self.atr_multiplier * atr,
            close < close.shift(1) - self.atr_multiplier * atr
        )
        signals = pd.DataFrame({'signal': signals})
        logging.info(f"ATR Breakout signals: {signals['signal'].value_counts().to_dict()}")
        return signals

class OBVStrategy(TradingStrategy):
    def __init__(self, data):
        super().__init__(data)
        self.add_indicator('OBV', on_balance_volume)

    def generate_signals(self):
        obv = self.indicators['OBV']
        signals = self.generate_signal(obv > obv.shift(1), obv < obv.shift(1))
        signals = pd.DataFrame({'signal': signals})
        logging.info(f"OBV signals: {signals['signal'].value_counts().to_dict()}")
        return signals

class ADIStrategy(TradingStrategy):
    def __init__(self, data):
        super().__init__(data)
        self.add_indicator('ADI', accumulation_distribution_index)

    def generate_signals(self):
        adi = self.indicators['ADI']
        signals = self.generate_signal(adi > adi.shift(1), adi < adi.shift(1))
        signals = pd.DataFrame({'signal': signals})
        logging.info(f"ADI signals: {signals['signal'].value_counts().to_dict()}")
        return signals
