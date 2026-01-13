# data_fetcher.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests

class StockDataFetcher:
    def __init__(self):
        # Stock lists
        self.tech_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA']
        self.indian_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
        self.crypto = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD']
    
    def get_stock_data(self, symbol, period='1mo'):
        """
        Fetch stock data and calculate basic indicators
        Returns: (dataframe, company_info_dict) or None if failed
        """
        try:
            print(f"Fetching data for {symbol}...")  # Debug print
            
            # Fetch data
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            
            if df.empty or len(df) < 5:
                print(f"No data returned for {symbol}")
                return None
            
            # Calculate basic indicators
            df['Daily_Return'] = df['Close'].pct_change()
            df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252) * 100
            
            # Moving averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # Get company info
            try:
                info = stock.info
                company_info = {
                    'name': info.get('longName', symbol),
                    'sector': info.get('sector', 'N/A'),
                    'industry': info.get('industry', 'N/A'),
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'dividend_yield': info.get('dividendYield', 0),
                    'beta': info.get('beta', 1),
                    'website': info.get('website', 'N/A'),
                    'country': info.get('country', 'N/A')
                }
            except:
                # Fallback if info fails
                company_info = {
                    'name': symbol,
                    'sector': 'N/A',
                    'industry': 'N/A',
                    'market_cap': 0,
                    'pe_ratio': 0,
                    'dividend_yield': 0,
                    'beta': 1
                }
            
            print(f"Successfully fetched {len(df)} rows for {symbol}")
            return df, company_info
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """
        Calculate technical indicators for a dataframe
        """
        try:
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            return df
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return df
    
    def get_news(self, symbol, num_articles=5):
        """
        Get news for a stock (simplified version)
        """
        try:
            stock = yf.Ticker(symbol)
            news = stock.news[:num_articles]
            
            formatted_news = []
            for item in news:
                formatted_news.append({
                    'title': item.get('title', 'No title'),
                    'publisher': item.get('publisher', 'Unknown'),
                    'link': item.get('link', '#'),
                    'summary': item.get('summary', 'No summary available')
                })
            
            return formatted_news
        except:
            # Return mock news if API fails
            return [
                {
                    'title': f'{symbol} shows strong quarterly results',
                    'publisher': 'Financial Times',
                    'summary': f'{symbol} exceeded analyst expectations for Q4 earnings.'
                },
                {
                    'title': f'Analysts raise target price for {symbol}',
                    'publisher': 'Bloomberg',
                    'summary': f'Multiple analysts have increased their price targets for {symbol}.'
                }
            ]
    
    def get_market_summary(self):
        """
        Get summary of major market indices
        """
        indices = {
            '^GSPC': 'S&P 500',
            '^IXIC': 'NASDAQ',
            '^DJI': 'Dow Jones',
            '^NSEI': 'Nifty 50',
            '^BSESN': 'Sensex'
        }
        
        summary = {}
        for symbol, name in indices.items():
            try:
                data = yf.Ticker(symbol)
                hist = data.history(period='1d')
                
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    previous = hist['Open'].iloc[-1] if 'Open' in hist.columns else hist['Close'].iloc[0]
                    
                    change = current - previous
                    change_percent = (change / previous) * 100
                    
                    summary[name] = {
                        'price': round(current, 2),
                        'change': round(change, 2),
                        'change_percent': round(change_percent, 2)
                    }
            except:
                continue
        
        # If no real data, return mock data
        if not summary:
            summary = {
                'S&P 500': {'price': 4780.56, 'change': 15.23, 'change_percent': 0.32},
                'NASDAQ': {'price': 14942.76, 'change': 85.12, 'change_percent': 0.57},
                'Dow Jones': {'price': 37466.11, 'change': 201.94, 'change_percent': 0.54},
                'Nifty 50': {'price': 21778.70, 'change': 247.35, 'change_percent': 1.15},
                'Sensex': {'price': 72410.38, 'change': 847.27, 'change_percent': 1.18}
            }
        
        return summary
    
    def get_stock_info(self, symbol):
        """
        Get detailed stock information
        """
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            return {
                'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'day_high': info.get('dayHigh', 0),
                'day_low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'avg_volume': info.get('averageVolume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 1),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
                'employees': info.get('fullTimeEmployees', 0),
                'website': info.get('website', 'N/A'),
                'country': info.get('country', 'N/A')
            }
        except:
            return None

# Test the class (will run when file is executed directly)
if __name__ == "__main__":
    print("Testing StockDataFetcher...")
    fetcher = StockDataFetcher()
    print(f"Tech stocks: {fetcher.tech_stocks}")
    
    # Test fetching data
    test_symbol = "AAPL"
    data = fetcher.get_stock_data(test_symbol, "1mo")
    
    if data:
        df, info = data
        print(f"\nSuccessfully fetched {test_symbol}:")
        print(f"Data shape: {df.shape}")
        print(f"Latest close: ${df['Close'].iloc[-1]:.2f}")
        print(f"Company: {info['name']}")
        print(f"Sector: {info['sector']}")
    else:
        print(f"Failed to fetch data for {test_symbol}")