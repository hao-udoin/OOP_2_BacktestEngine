import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PositionTracker import PositionTracker
from Backtester import Backtester
from TradingStrategy import SMACrossoverStrategy, LinearRegressionStrategy

class PositionTrackerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Portfolio Tracker")

        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create manual transactions tab
        self.manual_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.manual_tab, text="Manual Transactions")
        self.init_manual_tab()

        # Create strategy backtest tab
        self.strategy_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.strategy_tab, text="Strategy Backtest")
        self.init_strategy_tab()

    # Initialize manual transactions tab
    def init_manual_tab(self):
        # Initialize position tracker
        self.portfolio = PositionTracker()
        
        # Create main container
        self.main_frame = ttk.Frame(self.manual_tab)
        self.main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Transaction input UI
        self.create_input_ui()
        
        # Results display UI (hidden initially)
        self.results_frame = ttk.Frame(self.main_frame)
        
        # List to store entry fields
        self.entry_fields = []

    # Initialize strategy backtest tab
    def init_strategy_tab(self):
        self.strategy_frame = ttk.Frame(self.strategy_tab)
        self.strategy_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.create_backtest_controls(self.strategy_frame)
        ttk.Button(self.strategy_frame, text="Run Backtest", command=self.run_backtest).pack(pady=10)
        ttk.Button(self.strategy_frame, text="Export Results", command=self.trigger_export).pack(pady=5)


    # Create input UI for transactions
    def create_input_ui(self):
        """Create transaction input interface"""
        input_frame = ttk.Frame(self.main_frame)
        input_frame.pack(fill=tk.X, pady=5)
        
        # Input labels
        ttk.Label(input_frame, text="Date (YYYY-MM-DD):").grid(row=0, column=0, padx=5)
        ttk.Label(input_frame, text="Ticker:").grid(row=0, column=1, padx=5)
        ttk.Label(input_frame, text="Shares Delta:").grid(row=0, column=2, padx=5)
        
        # Entry fields
        self.date_entry = ttk.Entry(input_frame, width=12)
        self.ticker_entry = ttk.Entry(input_frame, width=8)
        self.delta_entry = ttk.Entry(input_frame, width=8)
        
        self.date_entry.grid(row=1, column=0, padx=5)
        self.ticker_entry.grid(row=1, column=1, padx=5)
        self.delta_entry.grid(row=1, column=2, padx=5)
        
        # Buttons
        btn_frame = ttk.Frame(self.main_frame)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Add Transaction", 
                   command=self.add_transaction).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Done", 
                   command=self.show_results).pack(side=tk.LEFT, padx=5)
        
        # Dynamic entry fields frame
        self.dynamic_frame = ttk.Frame(self.main_frame)
        self.dynamic_frame.pack(fill=tk.X)

    def add_transaction(self):
        """Handle transaction submission"""
        try:
            date = self.date_entry.get()
            ticker = self.ticker_entry.get().upper()
            delta = int(self.delta_entry.get())
            
            if not date or not ticker:
                raise ValueError("Missing required fields")
            
            self.portfolio.add_transaction(date, ticker, delta)
            
            # Clear input fields
            self.date_entry.delete(0, tk.END)
            self.ticker_entry.delete(0, tk.END)
            self.delta_entry.delete(0, tk.END)
            
            # Add new dynamic entry fields
            self.add_dynamic_entry_fields(date, ticker, delta)
            
        except Exception as e:
            messagebox.showerror("Input Error", f"Invalid input: {str(e)}")

    def add_dynamic_entry_fields(self, date, ticker, delta):
        """Add new dynamic entry fields"""
        # Create a new frame for each transaction
        transaction_frame = ttk.Frame(self.dynamic_frame)
        transaction_frame.pack(fill=tk.X)
        
        # Add labels and values
        ttk.Label(transaction_frame, text=f"Date: {date}").pack(side=tk.LEFT)
        ttk.Label(transaction_frame, text=f"Ticker: {ticker}").pack(side=tk.LEFT, padx=10)
        ttk.Label(transaction_frame, text=f"Shares Delta: {delta}").pack(side=tk.LEFT, padx=10)
        
        # Store the frame for future reference
        self.entry_fields.append(transaction_frame)
    
    def show_results(self):
        """Show transaction history in a new frame"""
        # Hide input UI
        for widget in self.main_frame.winfo_children():
            widget.pack_forget()
            
        # Show results frame
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Get portfolio data
        df = self.portfolio.get_portfolio()
        
        # Reset index to include date as a column
        df_with_date = df.reset_index()
        
        # Create new column list with "Date" first
        new_columns = ['Date'] + df.columns.tolist()
        df_with_date.columns = new_columns  # This matches column count
        
        # Create treeview
        tree = ttk.Treeview(self.results_frame)
        tree["columns"] = new_columns
        tree["show"] = "headings"
        
        # Configure columns
        for col in new_columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor=tk.CENTER)
            
        # Insert data
        for _, row in df_with_date.iterrows():
            tree.insert("", tk.END, values=list(row))
        
        # Add scrollbar
        vsb = ttk.Scrollbar(self.results_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        
        # Layout
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add back button 
        ttk.Button(self.results_frame, text="New Session", 
                 command=self.reset_app).pack(pady=10)
        # Add plot button to results frame
        ttk.Button(self.results_frame, text="Plot Portfolio Value", 
                command=self.plot_portfolio_value).pack(pady=10)
        
    def plot_portfolio_value(self):
        """Calculate and plot portfolio value over time"""
        try:
            # Get portfolio history and current positions
            df = self.portfolio.get_portfolio()
            if df.empty:
                messagebox.showinfo("Info", "No transactions to plot")
                return

            # Get unique tickers and date range
            tickers = df.columns.tolist()
            start_date = df.index.min().strftime('%Y-%m-%d')
            end_date = df.index.max().strftime('%Y-%m-%d')

            # Fetch historical prices from Yahoo Finance
            price_data = yf.download(tickers, start=start_date, end=end_date)['Close']
            price_data = price_data.reindex(df.index).ffill()  # Align dates with positions

            # Calculate daily portfolio value
            portfolio_value = (df * price_data).sum(axis=1)

            # Create plot
            fig = plt.Figure(figsize=(10, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.plot(portfolio_value.index, portfolio_value)
            ax.set_title('Portfolio Value Over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value (USD)')
            ax.grid(True)

            # Embed plot in Tkinter window
            plot_window = tk.Toplevel(self.master)
            plot_window.title("Portfolio Performance")
            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to generate plot: {str(e)}")
        


    # Create backtest controls for strategy tab
    def create_backtest_controls(self, parent):
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=5)

        ttk.Label(control_frame, text="Backtest Tickers (comma separated):").grid(row=0, column=0, padx=5)
        ttk.Label(control_frame, text="Start Date (YYYY-MM-DD):").grid(row=0, column=1, padx=5)
        ttk.Label(control_frame, text="End Date (YYYY-MM-DD):").grid(row=0, column=2, padx=5)
        ttk.Label(control_frame, text="Strategy:").grid(row=0, column=3, padx=5)

        self.bt_ticker_entry = ttk.Entry(control_frame, width=20)
        self.bt_start_entry = ttk.Entry(control_frame, width=12)
        self.bt_end_entry = ttk.Entry(control_frame, width=12)
        self.strategy_var = tk.StringVar()
        self.strategy_combo = ttk.Combobox(control_frame, textvariable=self.strategy_var, state="readonly",
                                           values=["SMA Crossover", "Linear Regression"])
        self.strategy_combo.current(0)

        self.bt_ticker_entry.grid(row=1, column=0, padx=5)
        self.bt_start_entry.grid(row=1, column=1, padx=5)
        self.bt_end_entry.grid(row=1, column=2, padx=5)
        self.strategy_combo.grid(row=1, column=3, padx=5) 

    # Run backtest for selected strategy (SMA Crossover or Linear Regression)
    def run_backtest(self):
        ticker_input = self.bt_ticker_entry.get()
        start_date = self.bt_start_entry.get()
        end_date = self.bt_end_entry.get()
        strategy_choice = self.strategy_var.get()

        if not ticker_input or not start_date or not end_date:
            messagebox.showerror("Error", "Please enter tickers and date range.")
            return

        tickers = [t.strip().upper() for t in ticker_input.split(',')]

        try:
            price_data = yf.download(tickers, start=start_date, end=end_date)["Close"]
            price_data = price_data.ffill().dropna()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch stock prices: {str(e)}")
            return

        simulated_df = pd.DataFrame(index=price_data.index)

        for ticker in tickers:
            single_price = pd.DataFrame({"Close": price_data[ticker],
                                         "Open": price_data[ticker],
                                         "High": price_data[ticker],
                                         "Low": price_data[ticker],
                                         "Volume": np.random.randint(1000000, 5000000, size=len(price_data))})

            if strategy_choice == "SMA Crossover":
                strategy = SMACrossoverStrategy(single_price, short_window=5, long_window=20)
            elif strategy_choice == "Linear Regression":
                strategy = LinearRegressionStrategy(single_price)
            else:
                messagebox.showerror("Error", "Invalid strategy selected.")
                return

            signals = strategy.generate_signals()
            simulated_df[ticker] = signals['signal']

        # Adding trade cost and slippage to the backtest
        trade_cost_rate = 0.015 
        price_data_adj = price_data * (1 - trade_cost_rate)

        self.simulated_df = simulated_df
        self.price_data_adj = price_data_adj

        backtester = Backtester(simulated_df, price_data)
        metrics = backtester.calculate_metrics()

        messagebox.showinfo("Backtest Results",
                            f"Cumulative Return: {metrics['Cumulative Return']:.2%}\n"
                            f"Max Drawdown: {metrics['Max Drawdown']:.2%}\n"
                            f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")

        backtester.plot_results()

    # Export results to CSV
    def trigger_export(self):
        try:
            ticker_input = self.bt_ticker_entry.get()
            tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
            if not hasattr(self, 'simulated_df') or not hasattr(self, 'price_data_adj'):
                messagebox.showwarning("Warning", "Please run a backtest first.")
                return
            self.export_results(self.simulated_df, self.price_data_adj)
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def export_results(self, signals_df, prices_df):
        try:
            export_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                       filetypes=[("CSV files", "*.csv")],
                                                       title="Save Backtest Results")
            if export_path:
                combined = signals_df.copy()
                for col in prices_df.columns:
                    combined[f"Price_{col}"] = prices_df[col]
                combined.to_csv(export_path)
                messagebox.showinfo("Export", f"Results saved to {export_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Could not save file: {str(e)}")


    # Add a method to reset the application for a new session
    def reset_app(self):
        """Reset the application for new session"""
        self.results_frame.destroy()
        self.results_frame = ttk.Frame(self.main_frame)
        self.portfolio = PositionTracker()
        self.create_input_ui()
        self.entry_fields = []

if __name__ == "__main__":
    root = tk.Tk()
    app = PositionTrackerApp(root)
    root.mainloop()
