import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
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
        """
        Helper method to generate signals based on buy and sell conditions.
        :param condition_buy: Boolean condition for buy signals.
        :param condition_sell: Boolean condition for sell signals.
        :return: Pandas Series of signals.
        """
        signals = pd.Series(0.0, index=self.data.index)
        signals[condition_buy] = 1.0
        signals[condition_sell] = -1.0
        return signals

    def optimize_parameters(self, parameter_grid):
        raise NotImplementedError

def validate_columns(data, required_columns):
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

# Momentum Indicators
def relative_strength_index(data, window=14):
    """
    Calculates the Relative Strength Index (RSI).
    :param data: DataFrame with a 'Close' column.
    :param window: Lookback period for RSI calculation.
    :return: Pandas Series of RSI values.
    """
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def stochastic_oscillator(data, window=14):
    """
    Calculates the Stochastic Oscillator.
    :param data: DataFrame with 'High', 'Low', and 'Close' columns.
    :param window: Lookback period for calculation.
    :return: Pandas Series of Stochastic Oscillator values.
    """
    lowest_low = data['Low'].rolling(window=window).min()
    highest_high = data['High'].rolling(window=window).max()
    stochastic = 100 * (data['Close'] - lowest_low) / (highest_high - lowest_low)
    return stochastic

# Trend Indicators
def moving_average_convergence_divergence(data, short_window=12, long_window=26, signal_window=9):
    """
    Calculates the Moving Average Convergence Divergence (MACD).
    :param data: DataFrame with a 'Close' column.
    :param short_window: Short-term EMA period.
    :param long_window: Long-term EMA period.
    :param signal_window: Signal line EMA period.
    :return: DataFrame with MACD line and Signal line.
    """
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return pd.DataFrame({'MACD': macd, 'Signal': signal})

def commodity_channel_index(data, window=20):
    """
    Calculates the Commodity Channel Index (CCI).
    :param data: DataFrame with 'High', 'Low', and 'Close' columns.
    :param window: Lookback period for calculation.
    :return: Pandas Series of CCI values.
    """
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    mean_tp = typical_price.rolling(window=window).mean()
    mean_deviation = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (typical_price - mean_tp) / (0.015 * mean_deviation)
    return cci

# Volatility Measures
def bollinger_bands(data, window=20, num_std_dev=2):
    """
    Calculates Bollinger Bands.
    :param data: DataFrame with a 'Close' column.
    :param window: Lookback period for moving average.
    :param num_std_dev: Number of standard deviations for the bands.
    :return: DataFrame with Upper Band, Lower Band, and Moving Average.
    """
    moving_avg = data['Close'].rolling(window=window).mean()
    std_dev = data['Close'].rolling(window=window).std()
    upper_band = moving_avg + (num_std_dev * std_dev)
    lower_band = moving_avg - (num_std_dev * std_dev)
    return pd.DataFrame({'Upper Band': upper_band, 'Lower Band': lower_band, 'Moving Average': moving_avg})

def average_true_range(data, window=14):
    """
    Calculates the Average True Range (ATR).
    :param data: DataFrame with 'High', 'Low', and 'Close' columns.
    :param window: Lookback period for ATR calculation.
    :return: Pandas Series of ATR values.
    """
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

# Volume-Based Indicators
def on_balance_volume(data):
    """
    Calculates the On-Balance Volume (OBV).
    :param data: DataFrame with 'Close' and 'Volume' columns.
    :return: Pandas Series of OBV values.
    """
    obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    return obv

def accumulation_distribution_index(data):
    """
    Calculates the Accumulation/Distribution Index (ADI).
    :param data: DataFrame with 'High', 'Low', 'Close', and 'Volume' columns.
    :return: Pandas Series of ADI values.
    """
    money_flow_multiplier = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
    money_flow_volume = money_flow_multiplier * data['Volume']
    adi = money_flow_volume.cumsum()
    return adi


#Strategy: Simple Moving Average Crossover
class SMACrossoverStrategy(TradingStrategy):
    def __init__(self, data, short_window, long_window):
        super().__init__(data)
        self.short_window = short_window
        self.long_window = long_window
        self.add_indicator('short_ma', moving_average_convergence_divergence, short_window=short_window, long_window=long_window)
        self.add_indicator('long_ma', moving_average_convergence_divergence, window=long_window)

    def generate_signals(self):

        if 'short_ma' not in self.indicators or 'long_ma' not in self.indicators:
            raise ValueError("Required indicators 'short_ma' and 'long_ma' are missing.")
    
        short_ma = self.indicators['short_ma']
        long_ma = self.indicators['long_ma']
    
        # Handle missing data
        if short_ma.isnull().any() or long_ma.isnull().any():
            logging.warning("Missing data in moving averages. Signals may be incomplete.")
    
        condition_buy = short_ma > long_ma
        condition_sell = short_ma < long_ma
    
        signals = self.generate_signal(condition_buy, condition_sell)
    
        logging.info(f"Generated {signals['signal'].value_counts().get(1.0, 0)} buy signals and "
                 f"{signals['signal'].value_counts().get(-1.0, 0)} sell signals.")
    
        return signals


# Machine Learning Strategy: Linear Regression for Price Prediction
class LinearRegressionStrategy(TradingStrategy):
    def __init__(self, data):
        super().__init__(data)
        required_columns = ['Open', 'High', 'Low', 'Volume', 'Close']
        validate_columns(data, required_columns)

        # Handle missing data
        if self.data.isnull().any().any():
            logging.warning("Missing data detected. Filling missing values with forward fill.")
            self.data.fillna(method='ffill', inplace=True)

    def generate_signals(self):
        # Prepare data for model training
        X = self.data[['Open', 'High', 'Low', 'Volume']]
        y = self.data['Close']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Generate predictions for the test set
        predictions = model.predict(X_test)

        # Evaluate model performance
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        logging.info(f"Model Performance: MSE = {mse:.2f}, R-squared = {r2:.2f}")

        # Generate signals based on predictions
        signals = pd.DataFrame(index=X_test.index)
        signals['signal'] = 0.0
        signals['signal'][predictions > y_test] = 1.0
        signals['signal'][predictions < y_test] = -1.0

        logging.info(f"Generated {signals['signal'].value_counts().get(1.0, 0)} buy signals and "
                     f"{signals['signal'].value_counts().get(-1.0, 0)} sell signals.")

        return signals
    
def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    excess_returns = returns - risk_free_rate
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)

# Grid Search for SMACrossover Strategy
def optimize_sma_crossover(data, short_window_range, long_window_range):
    best_performance = float('-inf')
    best_parameters = None

    for short_window in short_window_range:
        for long_window in long_window_range:
            if short_window >= long_window:
                continue

            logging.info(f"Testing Short Window: {short_window}, Long Window: {long_window}")
            strategy = SMACrossoverStrategy(data, short_window, long_window)
            signals = strategy.generate_signals()

            data['strategy_returns'] = signals['signal'].shift(1) * data['Close'].pct_change()
            sharpe_ratio = calculate_sharpe_ratio(data['strategy_returns'].dropna())

            if sharpe_ratio > best_performance:
                best_performance = sharpe_ratio
                best_parameters = (short_window, long_window)

    logging.info(f"Best Parameters: {best_parameters}, Sharpe Ratio: {best_performance}")
    return best_parameters

class RSIStrategy(TradingStrategy):
    def __init__(self, data, rsi_window=14, overbought=70, oversold=30):
        if not (0 < rsi_window <= 100):
            raise ValueError("RSI window must be between 1 and 100.")
        if not (0 <= oversold < overbought <= 100):
            raise ValueError("Oversold must be less than Overbought, and both must be between 0 and 100.")
        super().__init__(data)
        super().__init__(data)
        self.rsi_window = rsi_window
        self.overbought = overbought
        self.oversold = oversold
        self.add_indicator('RSI', relative_strength_index, window=rsi_window)

    def generate_signals(self):
        condition_buy = self.indicators['RSI'] < self.oversold
        condition_sell = self.indicators['RSI'] > self.overbought
        return self.generate_signal(condition_buy, condition_sell)
    
class StochasticOscillatorStrategy(TradingStrategy):
    def __init__(self, data, window=14, overbought=80, oversold=20):
        """
        Initializes the Stochastic Oscillator Strategy.
        :param data: DataFrame with market data.
        :param window: Lookback period for the Stochastic Oscillator.
        :param overbought: Upper threshold for sell signals.
        :param oversold: Lower threshold for buy signals.
        """
        super().__init__(data)
        required_columns = ['High', 'Low', 'Close']
        validate_columns(data, required_columns)
        self.window = window
        self.overbought = overbought
        self.oversold = oversold
        self.add_indicator('Stochastic', stochastic_oscillator, window=window)

    def generate_signals(self):
        """
        Generates buy and sell signals based on the Stochastic Oscillator.
        :return: Pandas DataFrame with signals.
        """
        stochastic = self.indicators['Stochastic']

        # Handle missing data
        if stochastic.isnull().any():
            logging.warning("Missing data in Stochastic Oscillator. Signals may be incomplete.")

        condition_buy = stochastic < self.oversold
        condition_sell = stochastic > self.overbought
        signals = self.generate_signal(condition_buy, condition_sell)

        logging.info(f"Generated {signals['signal'].value_counts().get(1.0, 0)} buy signals and "
                     f"{signals['signal'].value_counts().get(-1.0, 0)} sell signals.")

        return signals
    
class MACDStrategy(TradingStrategy):
    def __init__(self, data, short_window=12, long_window=26, signal_window=9):
        """
        Initializes the MACD Strategy.
        :param data: DataFrame with market data.
        :param short_window: Short-term EMA period.
        :param long_window: Long-term EMA period.
        :param signal_window: Signal line EMA period.
        """
        super().__init__(data)
        required_columns = ['Close']
        validate_columns(data, required_columns)
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window
        self.add_indicator('MACD', moving_average_convergence_divergence, 
                           short_window=short_window, long_window=long_window, signal_window=signal_window)

    def generate_signals(self):
        """
        Generates buy and sell signals based on the MACD indicator.
        :return: Pandas DataFrame with signals.
        """
        macd = self.indicators['MACD']['MACD']
        signal = self.indicators['MACD']['Signal']

        # Handle missing data
        if macd.isnull().any() or signal.isnull().any():
            logging.warning("Missing data in MACD or Signal line. Signals may be incomplete.")

        condition_buy = macd > signal
        condition_sell = macd < signal
        signals = self.generate_signal(condition_buy, condition_sell)

        logging.info(f"Generated {signals['signal'].value_counts().get(1.0, 0)} buy signals and "
                     f"{signals['signal'].value_counts().get(-1.0, 0)} sell signals.")

        return signals
    
class BollingerBandsStrategy(TradingStrategy):
    def __init__(self, data, window=20, num_std_dev=2):
        """
        Initializes the Bollinger Bands Strategy.
        :param data: DataFrame with market data.
        :param window: Lookback period for Bollinger Bands.
        :param num_std_dev: Number of standard deviations for the bands.
        """
        super().__init__(data)
        required_columns = ['Close']
        validate_columns(data, required_columns)
        self.window = window
        self.num_std_dev = num_std_dev
        self.add_indicator('BollingerBands', bollinger_bands, window=window, num_std_dev=num_std_dev)

    def generate_signals(self):
        """
        Generates buy and sell signals based on Bollinger Bands.
        :return: Pandas DataFrame with signals.
        """
        lower_band = self.indicators['BollingerBands']['Lower Band']
        upper_band = self.indicators['BollingerBands']['Upper Band']
        close = self.data['Close']

        # Handle missing data
        if lower_band.isnull().any() or upper_band.isnull().any():
            logging.warning("Missing data in Bollinger Bands. Signals may be incomplete.")

        condition_buy = close < lower_band
        condition_sell = close > upper_band
        signals = self.generate_signal(condition_buy, condition_sell)

        logging.info(f"Generated {signals['signal'].value_counts().get(1.0, 0)} buy signals and "
                     f"{signals['signal'].value_counts().get(-1.0, 0)} sell signals.")

        return signals
    
class ATRBreakoutStrategy(TradingStrategy):
    def __init__(self, data, atr_window=14, atr_multiplier=2):
        """
        Initializes the ATR Breakout Strategy.
        :param data: DataFrame with market data.
        :param atr_window: Lookback period for ATR calculation.
        :param atr_multiplier: Multiplier for breakout levels.
        """
        super().__init__(data)
        required_columns = ['High', 'Low', 'Close']
        validate_columns(data, required_columns)
        self.atr_window = atr_window
        self.atr_multiplier = atr_multiplier
        self.add_indicator('ATR', average_true_range, window=atr_window)

    def generate_signals(self):
        """
        Generates buy and sell signals based on ATR breakout levels.
        :return: Pandas DataFrame with signals.
        """
        atr = self.indicators['ATR']
        close = self.data['Close']

        # Handle missing data
        if atr.isnull().any():
            logging.warning("Missing data in ATR. Signals may be incomplete.")

        condition_buy = close > close.shift(1) + self.atr_multiplier * atr
        condition_sell = close < close.shift(1) - self.atr_multiplier * atr
        signals = self.generate_signal(condition_buy, condition_sell)

        logging.info(f"Generated {signals['signal'].value_counts().get(1.0, 0)} buy signals and "
                     f"{signals['signal'].value_counts().get(-1.0, 0)} sell signals.")

        return signals
    
class OBVStrategy(TradingStrategy):
    def __init__(self, data):
        """
        Initializes the OBV Strategy.
        :param data: DataFrame with market data.
        """
        super().__init__(data)
        self.add_indicator('OBV', on_balance_volume)

    def generate_signals(self):
        """
        Generates buy and sell signals based on the On-Balance Volume (OBV) indicator.
        :return: Pandas DataFrame with signals.
        """
        obv = self.indicators['OBV']

        # Handle missing data
        if obv.isnull().any():
            logging.warning("Missing data in OBV. Signals may be incomplete.")

        condition_buy = obv > obv.shift(1)
        condition_sell = obv < obv.shift(1)
        signals = self.generate_signal(condition_buy, condition_sell)

        logging.info(f"Generated {signals['signal'].value_counts().get(1.0, 0)} buy signals and "
                     f"{signals['signal'].value_counts().get(-1.0, 0)} sell signals.")

        return signals
    
class ADIStrategy(TradingStrategy):
    def __init__(self, data):
        """
        Initializes the ADI Strategy.
        :param data: DataFrame with market data.
        """
        super().__init__(data)
        self.add_indicator('ADI', accumulation_distribution_index)

    def generate_signals(self):
        """
        Generates buy and sell signals based on the Accumulation/Distribution Index (ADI) indicator.
        :return: Pandas DataFrame with signals.
        """
        adi = self.indicators['ADI']

        # Handle missing data
        if adi.isnull().any():
            logging.warning("Missing data in ADI. Signals may be incomplete.")

        condition_buy = adi > adi.shift(1)
        condition_sell = adi < adi.shift(1)
        signals = self.generate_signal(condition_buy, condition_sell)

        logging.info(f"Generated {signals['signal'].value_counts().get(1.0, 0)} buy signals and "
                     f"{signals['signal'].value_counts().get(-1.0, 0)} sell signals.")

        return signals
    
