# debug_dashboard.py
import streamlit as st
import traceback
import sys

st.set_page_config(layout="wide")

st.title("DEBUG MODE - Dashboard Error Check")

# Try to import and run your dashboard with error handling
try:
    print("=== ATTEMPTING TO IMPORT DASHBOARD ===")
    
    # Try to import your modules
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        import yfinance as yf
        
        print("✅ Basic imports successful")
        
    except ImportError as e:
        st.error(f"Import Error: {e}")
        print(f"❌ Import failed: {e}")
        st.stop()
    
    # Try to import your custom modules
    try:
        # We'll create simple mock classes for testing
        class StockDataFetcher:
            def __init__(self):
                self.tech_stocks = ['AAPL', 'GOOGL', 'MSFT']
                self.indian_stocks = ['RELIANCE.NS']
                self.crypto = ['BTC-USD']
            
            def get_stock_data(self, symbol, period):
                try:
                    stock = yf.Ticker(symbol)
                    df = stock.history(period=period)
                    if df.empty:
                        return None
                    
                    df['Daily_Return'] = df['Close'].pct_change()
                    df['SMA_20'] = df['Close'].rolling(window=20).mean()
                    
                    company_info = {
                        'name': symbol,
                        'sector': 'Test',
                        'market_cap': 1000000
                    }
                    return df, company_info
                except:
                    return None
        
        class StockPredictor:
            def train_model(self, df):
                return 0.8, 0.7
            def predict(self, df, days):
                return np.array([df['Close'].iloc[-1]] * days)
        
        class ProphetPredictor:
            def __init__(self):
                pass
        
        print("✅ Mock classes created")
        
    except Exception as e:
        st.error(f"Class creation error: {e}")
        print(f"❌ Class creation failed: {e}")
        traceback.print_exc()
    
    # Try to create and run dashboard
    st.subheader("Creating Dashboard Instance")
    try:
        # Create a minimal dashboard class
        class SimpleDashboard:
            def __init__(self):
                self.fetcher = StockDataFetcher()
                self.predictor = StockPredictor()
                self.prophet_predictor = ProphetPredictor()
            
            def run(self):
                st.success("✅ Dashboard is running!")
                st.write("Testing basic functionality...")
                
                # Test yfinance
                symbol = "AAPL"
                st.write(f"Testing {symbol}...")
                
                try:
                    data = yf.download(symbol, period="1d")
                    st.write(f"Data shape: {data.shape}")
                    if not data.empty:
                        st.write(f"Latest price: ${data['Close'].iloc[-1]:.2f}")
                except Exception as e:
                    st.error(f"yfinance error: {e}")
        
        dashboard = SimpleDashboard()
        dashboard.run()
        
    except Exception as e:
        st.error(f"Dashboard creation error: {e}")
        print(f"❌ Dashboard failed: {e}")
        traceback.print_exc()
        
except Exception as e:
    st.error(f"Fatal error: {e}")
    print(f"❌ FATAL ERROR: {e}")
    traceback.print_exc()

st.markdown("---")
st.subheader("System Info")
st.write(f"Python: {sys.version}")
st.write(f"Streamlit: {st.__version__}")