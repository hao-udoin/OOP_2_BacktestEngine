
Midway Report for Python-Based Algorithmic Trading Tool Using Historical Data 
=

Project Overview
-

This project aims to develop a Python-based algorithmic trading tool that utilizes historical market data to backtest and implement trading strategies. The tool will provide a framework for traders and researchers to analyze past market behavior, develop and refine trading algorithms, and evaluate their potential performance before deploying them in live markets.

Project Structure
-----------------

1.  **Data Acquisition and Management**:  
- Integrate with APIs from platforms like Yahoo Finance to fetch historical stock data.
- Implement data cleaning and preprocessing techniques to ensure data quality.

2.  **Strategy Development**: 
- Provide a flexible framework for users to code custom trading strategies in Python.
- Include pre-built modules for common technical indicators (e.g., RSI, Moving Averages).
- Allow for the integration of machine learning models for more advanced strategies.

3.  **Backtesting Engine**:
- Develop a robust backtesting system to simulate trading strategies on historical data.
- Implement performance metrics calculation (e.g., returns, Sharpe ratio, drawdowns).
- Visualize backtesting results using libraries like matplotlib.

4.  **Risk Management**:
- Incorporate risk management features such as position sizing and stop-loss mechanisms.
- Implement portfolio optimization techniques to balance risk and return.

5. **Reporting and Analysis**:
- Generate comprehensive reports on strategy performance and trade statistics.
- Provide data visualization tools for in-depth analysis of trading results.


Current Progress
----------------

### PostionTracker Class:

PostionTracker Class: This class is designed to keep track of stock portfolio holdings over time. It includes the following members:

-   **Attributes**:
    -   `current_holdings`: It stores the current positions in the portfolio. It will be updated as transactions are added.
    -   `history`: It stores store the historical transactions and positions.
    
-   **Methods**:
    -   `_process_transaction(self, date, ticker, delta)`: This is a private method used internally to handle transactions. The parameters represent the date of the transaction, the stock ticker, and the change in quantity. It updates the current_holdings and history attributes of the PositionTracker object to reflect the transaction.
    -   `add_transaction(self, date, ticker, shares)`: This method allows you to add new transactions to the portfolio. It takes the date, ticker, and shares as input and calls the _process_transaction method to update the portfolio.
    -   `get_portfolio(self, resample_freq=None)` : This method returns a pandas DataFrame representing the portfolio history.

### PositionTrackerApp Class:

This class is designed to create a graphical user interface (GUI) for managing and visualizing a stock portfolio. It uses the PositionTracker class to handle the underlying portfolio data.

-   **Attributes**:
    -   `master`: The main Tkinter window (root) of the application.
    -   `portfolio`: An instance of the PositionTracker class, used to store and manage the portfolio data.
    -   `main_frame`: The main frame (container) for the input UI elements.
    -   `results_frame`: The frame used to display the results (portfolio history, plot).
    -   `entry_fields`: A list to store dynamically added entry fields for displaying transactions.
    -   `date_entry, ticker_entry, delta_entry`: Entry widgets for user input of transaction details (date, ticker, shares delta).
    -   `dynamic_frame`: A frame to hold dynamically added transaction entries.
    
-   **Methods**:
    -   `__init__(self, master)`: Initializes the GUI application, creates the necessary frames and widgets, and initializes the PositionTracker object.
    -   `plot_portfolio_value(self)`: Calculates and displays a plot of the portfolio value over time using historical stock prices from Yahoo Finance. It uses matplotlib for plotting and integrates it into the Tkinter GUI.
    -   `create_input_ui(self)`: Creates the user interface elements for entering transactions, including labels, entry fields, and buttons.
    -   `add_transaction(self)`: Handles the submission of a new transaction by retrieving data from the input fields, calling the `add_transaction` method of the PositionTracker object, clearing the input fields, and dynamically adding new fields to display the entered transaction.
    -   `add_dynamic_entry_fields(self, date, ticker, delta)`: Creates and adds new entry fields to the `dynamic_frame` to display details of the entered transactions.
    -   `show_results(self)`: Hides the input UI and displays the portfolio history in a treeview widget within the `results_frame`. Includes a scrollbar and buttons for starting a new session and plotting the portfolio value.
    -   `reset_app(self)`: Resets the application to its initial state for a new session. It destroys the `results_frame`, resets the PositionTracker object, and recreates the input UI.

### TradingStrategy Class:

TradingStrategy Class: This class provides a framework for defining and implementing trading strategies. It allows you to add technical indicators, generate trading signals, and potentially optimize strategy parameters.

-   **Attributes**:
    -   `data`: Stores the market data (e.g., price, volume) used by the strategy.
    -   `indicators`: A dictionary that stores technical indicators calculated based on the market data.

-   **Methods**:
    -   `__init__(self, data)`: Initializes the TradingStrategy object with market data and an empty indicators dictionary.
    -   `add_indicator(self, indicator_name, indicator_function, **kwargs)`: Adds a technical indicator to the indicators dictionary. It takes the indicator name, the function used to calculate the indicator, and any additional keyword arguments for the indicator function.
    -   `generate_signals(self)`: This method is responsible for generating trading signals (buy, sell, or hold) based on the market data and indicators. It's an abstract method and needs to be implemented in concrete strategy classes.
    -   `optimize_parameters(self, parameter_grid)`: This method is intended for optimizing the strategy's parameters. It's also an abstract method and requires implementation in subclasses.

### Backtester Class:
The Backtester class is designed to simulate the performance of a trading strategy using historical data and calculate key performance metrics.

-   **Attributes**:
    -   `df`: A Pandas DataFrame containing the portfolio's transaction history and holding quantities (from PositionTracker).
    -   `price_data`: A Pandas DataFrame containing historical stock prices for the assets in the portfolio.
    -   `initial_cash`: The starting capital for the backtest. Defaults to 10,000.

-   **Methods**:
    -   `__init__(self, portfolio_data, price_data, initial_cash=10000)`: Initializes the Backtester with the portfolio data, price data, and initial cash. It also calculates the initial portfolio value.
    -   `_calculate_portfolio_value(self)`: A private method that calculates the daily portfolio value by multiplying the holdings by their respective prices and adding the initial cash.
    -   `calculate_metrics(self)`: Calculates performance metrics, including:
        -   **Cumulative Return**: The total percentage gain or loss.
        -   **Max Drawdown**: The largest percentage decline from a peak to a trough.
        -   **Sharpe Ratio**: A measure of risk-adjusted return.
    -   `plot_results(self)`: Generates a plot of the portfolio value over time to visualize the backtest results.

Future Work
-----------

1.  **Data Acquisition and Management**: Intergrate with APIs for data collection and implement data cleaning methods. 
2.  **Risk Management**: Incorporate risk management features such as position sizing and stop-loss mechanisms.
3.  **Reporting and Analysis**: Generate comprehensive reports on strategy performance and trade statistics.
4.  **Adding Trading Strategy Object**: Add other indicators or machine learning models for more advanced strategies.

Conclusion
----------

Overall, the project is on track to deliver a powerful and versatile algorithmic trading tool that utilize historical data to provide insights and performance evaluations, helping users make informed decisions before deploying strategies in live markets. Moving forward, the project will focus on integrating additional features such as advanced data acquisition and management, risk management tools, and comprehensive reporting and analysis capabilities. These enhancements will further improve the tool's usability and effectiveness, making it a valuable resource for traders and researchers looking to develop, test, and refine their trading strategies.