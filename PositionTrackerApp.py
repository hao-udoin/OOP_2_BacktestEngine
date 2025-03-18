import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import pandas as pd
import numpy as np
from PositionTracker import PositionTracker
from Backtester import Backtester

class PositionTrackerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Portfolio Tracker")
        
        # Initialize position tracker
        self.portfolio = PositionTracker()
        
        # Create main container
        self.main_frame = ttk.Frame(master)
        self.main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Transaction input UI
        self.create_input_ui()
        
        # Results display UI (hidden initially)
        self.results_frame = ttk.Frame(self.main_frame)
        
        # List to store entry fields
        self.entry_fields = []
        
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
        
        # Create results treeview
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Get portfolio data
        df = self.portfolio.get_portfolio()
        
        # Create treeview
        tree = ttk.Treeview(self.results_frame)
        tree["columns"] = list(df.columns)
        tree["show"] = "headings"
        
        # Configure columns
        tree.heading("#0", text="Date")
        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=80, anchor=tk.CENTER)
        
        # Insert data
        for index, row in df.iterrows():
            date_str = index.strftime('%Y-%m-%d')
            values = [int(row[col]) for col in df.columns]
            tree.insert("", tk.END, text=date_str, values=values)
        
        # Add scrollbar
        vsb = ttk.Scrollbar(self.results_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        
        # Layout
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add back button & Run Backtest button
        ttk.Button(self.results_frame, text="New Session",
                 command=self.reset_app).pack(pady=10)
        ttk.Button(self.results_frame, text="Run Backtest",
                 command=self.run_backtest).pack(pady=10)
        
    def run_backtest(self):
        df = self.portfolio.get_portfolio()
        tickers = list(df.columns)

        if df.empty:
            messagebox.showerror("Error", "No transactions to backtest.")
            return
        
        # download stock prices
        use_dummy_data = True  
        if use_dummy_data:
            price_data = self._get_dummy_price_data(df)
        # try:
        #     price_data = yf.download(tickers, start=df.index.min(), end=df.index.max())["Adj Close"]
        #     price_data = price_data.reindex(df.index).ffill().fillna(0)  # fill missing values
        # except Exception as e:
        #     messagebox.showerror("Error", f"Failed to fetch stock prices: {str(e)}")
        #     return
        
        # run backtest
        backtester = Backtester(df, price_data)
        metrics = backtester.calculate_metrics()
        
        # display results
        messagebox.showinfo("Backtest Results",
                            f"Cumulative Return: {metrics['Cumulative Return']:.2%}\n"
                            f"Max Drawdown: {metrics['Max Drawdown']:.2%}\n"
                            f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
        
        # plot results
        backtester.plot_results()


    def reset_app(self):
        """Reset the application for new session"""
        self.results_frame.destroy()
        self.results_frame = ttk.Frame(self.main_frame)
        self.portfolio = PositionTracker()
        self.create_input_ui()
        self.entry_fields = []

    def _get_dummy_price_data(self, df):
        """Generate dummy stock price data for testing"""
        np.random.seed(42)  # 設定隨機種子，確保結果一致
        tickers = list(df.columns)  # 取得所有持倉的股票代碼
        dates = df.index  # 獲取日期範圍

        # 生成隨機價格（初始價格 100，每日變動最多 2%）
        dummy_prices = pd.DataFrame(
            100 * (1 + np.random.randn(len(dates), len(tickers)) * 0.02).cumprod(axis=0),
            index=dates, columns=tickers
        )
        
        return dummy_prices

if __name__ == "__main__":
    root = tk.Tk()
    app = PositionTrackerApp(root)
    root.mainloop()
