import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

from data_fetcher import StockDataFetcher
from predictor import StockPredictor, ProphetPredictor

# Page config
st.set_page_config(
    page_title="Real-Time Stock Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1E3A8A;
    }
    .positive {
        color: #10B981;
        font-weight: bold;
    }
    .negative {
        color: #EF4444;
        font-weight: bold;
    }
    .stock-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class StockDashboard:
    def __init__(self):
        self.fetcher = StockDataFetcher()
        self.predictor = StockPredictor()
        self.prophet_predictor = ProphetPredictor()
        
    def run(self):
        st.title("📈 Real-Time Stock Market Dashboard with AI Predictions")
        
        # Sidebar
        with st.sidebar:
            st.header("🔧 Dashboard Controls")
            
            # Stock selection
            stock_options = {
                "Technology Stocks": self.fetcher.tech_stocks,
                "Indian Stocks": self.fetcher.indian_stocks,
                "Cryptocurrency": self.fetcher.crypto
            }
            
            category = st.selectbox("Select Category", list(stock_options.keys()))
            selected_stock = st.selectbox("Select Stock", stock_options[category])
            
            # Time period
            period = st.selectbox("Time Period", 
                                 ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"])
            
            # Prediction days
            predict_days = st.slider("Prediction Days", 1, 30, 7)
            
            # Technical indicators
            show_indicators = st.checkbox("Show Technical Indicators", True)
            
            st.markdown("---")
            st.info("""
            **Features:**
            - Real-time stock data
            - AI price predictions
            - Technical analysis
            - Portfolio simulation
            - Risk analysis
            """)
        
        # Main content
        if selected_stock:
            self.display_stock_analysis(selected_stock, period, predict_days, show_indicators)
        
        # Market overview
        st.markdown("---")
        self.display_market_overview()
        
        # Portfolio simulator
        st.markdown("---")
        self.display_portfolio_simulator()
    
    def display_stock_analysis(self, symbol, period, predict_days, show_indicators):
        """Display comprehensive stock analysis"""
        st.header(f"📊 {symbol} Analysis")
        
        # Fetch data
        data = self.fetcher.get_stock_data(symbol, period)
        
        if data:
            df, company_info = data
            
            # Create tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Overview", "🤖 Predictions", "📊 Technical", "📰 News", "🏢 Company"
            ])
            
            with tab1:
                self.display_overview_tab(df, symbol, company_info)
            
            with tab2:
                self.display_predictions_tab(df, symbol, predict_days)
            
            with tab3:
                if show_indicators:
                    self.display_technical_tab(df)
            
            with tab4:
                self.display_news_tab(symbol)
            
            with tab5:
                self.display_company_tab(company_info)
        else:
            st.error(f"Could not fetch data for {symbol}")
    
    def display_overview_tab(self, df, symbol, company_info):
        """Display overview tab"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = df['Close'].iloc[-1]
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            daily_change = df['Close'].iloc[-1] - df['Open'].iloc[-1]
            change_percent = (daily_change / df['Open'].iloc[-1]) * 100
            st.metric("Daily Change", 
                     f"${daily_change:.2f}", 
                     f"{change_percent:.2f}%")
        
        with col3:
            volume = df['Volume'].iloc[-1]
            st.metric("Volume", f"{volume:,}")
        
        with col4:
            volatility = df['Volatility'].iloc[-1]
            st.metric("Volatility", f"{volatility:.2f}%")
        
        # Price chart
        st.subheader("Price Chart")
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ))
        
        # Add moving averages
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA_20'],
            name='SMA 20',
            line=dict(color='orange', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA_50'],
            name='SMA 50',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title=f"{symbol} Price Chart",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Returns Analysis")
            returns_df = pd.DataFrame({
                'Period': ['1D', '1W', '1M', '3M', '6M', '1Y'],
                'Return': [
                    df['Daily_Return'].iloc[-1],
                    df['Close'].pct_change(5).iloc[-1] * 100,
                    df['Close'].pct_change(21).iloc[-1] * 100,
                    df['Close'].pct_change(63).iloc[-1] * 100,
                    df['Close'].pct_change(126).iloc[-1] * 100,
                    df['Close'].pct_change(252).iloc[-1] * 100
                ]
            })
            st.dataframe(returns_df.style.format({'Return': '{:.2f}%'}))
        
        with col2:
            st.subheader("Statistics")
            stats = {
                'Mean Price': df['Close'].mean(),
                'Std Deviation': df['Close'].std(),
                '52W High': df['Close'].max(),
                '52W Low': df['Close'].min(),
                'Sharpe Ratio': (df['Daily_Return'].mean() / df['Daily_Return'].std()) * np.sqrt(252)
            }
            
            for key, value in stats.items():
                st.metric(key, f"{value:.2f}")
    
    def display_predictions_tab(self, df, symbol, predict_days):
        """Display predictions tab"""
        st.subheader("🤖 AI Price Predictions")
        
        # Train model
        with st.spinner("Training prediction model..."):
            train_score, test_score = self.predictor.train_model(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model Training Score", f"{train_score:.2%}")
        
        with col2:
            st.metric("Model Test Score", f"{test_score:.2%}")
        
        # Make predictions
        predictions = self.predictor.predict(df, predict_days)
        
        # Create future dates
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                    periods=predict_days, freq='B')
        
        # Create prediction dataframe
        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': predictions
        })
        
        # Plot predictions
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=df.index[-50:],  # Last 50 days
            y=df['Close'].iloc[-50:],
            name='Historical',
            line=dict(color='blue', width=2)
        ))
        
        # Predictions
        fig.add_trace(go.Scatter(
            x=pred_df['Date'],
            y=pred_df['Predicted_Price'],
            name='Predictions',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        # Confidence interval (simplified)
        std_dev = np.std(predictions)
        fig.add_trace(go.Scatter(
            x=pd.concat([pd.Series(df.index[-1]), pred_df['Date']]),
            y=pd.concat([pd.Series(df['Close'].iloc[-1]), 
                        pred_df['Predicted_Price'] + std_dev]),
            fill=None,
            mode='lines',
            line=dict(color='green', width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=pd.concat([pd.Series(df.index[-1]), pred_df['Date']]),
            y=pd.concat([pd.Series(df['Close'].iloc[-1]), 
                        pred_df['Predicted_Price'] - std_dev]),
            fill='tonexty',
            mode='lines',
            line=dict(color='green', width=0),
            fillcolor='rgba(0,255,0,0.2)',
            name='Confidence Interval'
        ))
        
        fig.update_layout(
            title=f"{symbol} Price Predictions (Next {predict_days} Days)",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction table
        st.subheader("Detailed Predictions")
        pred_df['Change'] = pred_df['Predicted_Price'].pct_change() * 100
        pred_df['Cumulative_Return'] = (1 + pred_df['Change']/100).cumprod() - 1
        
        st.dataframe(pred_df.style.format({
            'Predicted_Price': '${:.2f}',
            'Change': '{:.2f}%',
            'Cumulative_Return': '{:.2%}'
        }))
        
        # Trading signals
        st.subheader("📊 Trading Signals")
        
        current_price = df['Close'].iloc[-1]
        predicted_price = predictions[-1]
        
        if predicted_price > current_price * 1.05:
            st.success("**Strong BUY Signal** - Expected growth > 5%")
            recommendation = "Consider buying for short-term gains"
        elif predicted_price > current_price * 1.02:
            st.info("**Moderate BUY Signal** - Expected growth 2-5%")
            recommendation = "Could be a good entry point"
        elif predicted_price < current_price * 0.95:
            st.error("**SELL Signal** - Expected decline > 5%")
            recommendation = "Consider reducing position"
        else:
            st.warning("**HOLD Signal** - Minimal expected movement")
            recommendation = "Maintain current position"
        
        st.write(recommendation)
    
    def display_technical_tab(self, df):
        """Display technical analysis tab"""
        st.subheader("Technical Indicators")
        
        # Calculate technical indicators
        df_tech = self.fetcher.calculate_technical_indicators(df.copy())
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Price with Bollinger Bands', 'RSI', 'MACD', 'Volume'),
            vertical_spacing=0.1,
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # Price with Bollinger Bands
        fig.add_trace(go.Scatter(
            x=df_tech.index, y=df_tech['Close'],
            name='Close', line=dict(color='blue')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df_tech.index, y=df_tech['BB_Upper'],
            name='Upper BB', line=dict(color='red', dash='dash')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df_tech.index, y=df_tech['BB_Lower'],
            name='Lower BB', line=dict(color='green', dash='dash'),
            fill='tonexty', fillcolor='rgba(0,100,80,0.2)'
        ), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(
            x=df_tech.index, y=df_tech['RSI'],
            name='RSI', line=dict(color='purple')
        ), row=2, col=1)
        
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(
            x=df_tech.index, y=df_tech['MACD'],
            name='MACD', line=dict(color='blue')
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=df_tech.index, y=df_tech['MACD_Signal'],
            name='Signal', line=dict(color='orange')
        ), row=3, col=1)
        
        # Volume
        colors = ['red' if row['Close'] < row['Open'] else 'green' 
                 for _, row in df_tech.iterrows()]
        
        fig.add_trace(go.Bar(
            x=df_tech.index, y=df_tech['Volume'],
            name='Volume', marker_color=colors
        ), row=4, col=1)
        
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical signals
        st.subheader("Technical Signals")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rsi = df_tech['RSI'].iloc[-1]
            if rsi > 70:
                st.error("RSI: Overbought (>70)")
            elif rsi < 30:
                st.success("RSI: Oversold (<30)")
            else:
                st.info(f"RSI: {rsi:.1f} (Normal)")
        
        with col2:
            if df_tech['MACD'].iloc[-1] > df_tech['MACD_Signal'].iloc[-1]:
                st.success("MACD: Bullish Crossover")
            else:
                st.error("MACD: Bearish Crossover")
        
        with col3:
            price = df_tech['Close'].iloc[-1]
            bb_upper = df_tech['BB_Upper'].iloc[-1]
            bb_lower = df_tech['BB_Lower'].iloc[-1]
            
            if price > bb_upper:
                st.error("Price above Upper BB: Overbought")
            elif price < bb_lower:
                st.success("Price below Lower BB: Oversold")
            else:
                st.info("Price within BB: Normal")
    
    def display_news_tab(self, symbol):
        """Display news tab"""
        news = self.fetcher.get_news(symbol)
        
        if news:
            for item in news:
                with st.container():
                    st.markdown(f"### {item.get('title', 'No Title')}")
                    st.write(f"**Publisher:** {item.get('publisher', 'Unknown')}")
                    st.write(f"**Time:** {item.get('providerPublishTime', '')}")
                    
                    if 'thumbnail' in item and item['thumbnail']:
                        st.image(item['thumbnail']['resolutions'][0]['url'], width=200)
                    
                    st.write(item.get('summary', 'No summary available'))
                    st.markdown("---")
        else:
            st.info("No recent news available")
    
    def display_company_tab(self, company_info):
        """Display company information tab"""
        st.subheader("Company Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Company Name", company_info['name'])
            st.metric("Sector", company_info['sector'])
            st.metric("Industry", company_info['industry'])
        
        with col2:
            market_cap = company_info['market_cap']
            if market_cap > 1e12:
                market_cap_str = f"${market_cap/1e12:.2f}T"
            elif market_cap > 1e9:
                market_cap_str = f"${market_cap/1e9:.2f}B"
            else:
                market_cap_str = f"${market_cap:,.0f}"
            
            st.metric("Market Cap", market_cap_str)
            st.metric("P/E Ratio", f"{company_info['pe_ratio']:.2f}")
            st.metric("Dividend Yield", f"{company_info['dividend_yield']:.2%}")
            st.metric("Beta", f"{company_info['beta']:.2f}")
    
    def display_market_overview(self):
        """Display market overview"""
        st.header("🌐 Market Overview")
        
        summary = self.fetcher.get_market_summary()
        
        if summary:
            cols = st.columns(len(summary))
            
            for idx, (index_name, data) in enumerate(summary.items()):
                with cols[idx]:
                    change_class = "positive" if data['change'] >= 0 else "negative"
                    st.markdown(f"""
                    <div class='stock-card'>
                        <h4>{index_name}</h4>
                        <h3>${data['price']:.2f}</h3>
                        <p class='{change_class}'>
                            {data['change']:+.2f} ({data['change_percent']:+.2f}%)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Heatmap of stocks
        st.subheader("📊 Stock Performance Heatmap")
        
        all_stocks = self.fetcher.tech_stocks + self.fetcher.indian_stocks[:3]
        performance_data = []
        
        for stock in all_stocks:
            try:
                data = yf.Ticker(stock)
                hist = data.history(period='1mo')
                
                if not hist.empty:
                    monthly_return = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100
                    performance_data.append({
                        'Stock': stock,
                        'Return': monthly_return,
                        'Volume': hist['Volume'].mean()
                    })
            except:
                continue
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            
            fig = px.treemap(perf_df, path=['Stock'], values='Volume',
                            color='Return', color_continuous_scale='RdYlGn',
                            title='Stock Performance (Size: Volume, Color: Return)')
            
            st.plotly_chart(fig, use_container_width=True)
    
    def display_portfolio_simulator(self):
        """Display portfolio simulator"""
        st.header("💰 Portfolio Simulator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Build Your Portfolio")
            
            portfolio = {}
            stocks = st.multiselect("Select stocks for portfolio", 
                                   self.fetcher.tech_stocks + self.fetcher.indian_stocks)
            
            for stock in stocks:
                allocation = st.slider(f"Allocation for {stock} (%)", 0, 100, 
                                      value=100//len(stocks) if stocks else 0,
                                      key=f"alloc_{stock}")
                portfolio[stock] = allocation
            
            # Ensure allocations sum to 100%
            total_alloc = sum(portfolio.values())
            if total_alloc != 100 and portfolio:
                st.warning(f"Allocations sum to {total_alloc}%. Should be 100%.")
        
        with col2:
            if portfolio and total_alloc == 100:
                st.subheader("Portfolio Analysis")
                
                # Calculate portfolio performance
                portfolio_value = 10000  # Start with $10,000
                portfolio_history = {}
                
                for stock, alloc in portfolio.items():
                    data = self.fetcher.get_stock_data(stock, '1y')
                    if data:
                        df, _ = data
                        # Calculate returns
                        returns = df['Close'].pct_change().fillna(0)
                        portfolio_history[stock] = returns * (alloc/100)
                
                if portfolio_history:
                    # Combine returns
                    portfolio_returns = pd.DataFrame(portfolio_history).sum(axis=1)
                    
                    # Calculate metrics
                    cumulative_return = (1 + portfolio_returns).cumprod() - 1
                    sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
                    max_drawdown = (portfolio_returns.cumsum().expanding().max() - 
                                   portfolio_returns.cumsum()).max()
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Return", f"{cumulative_return.iloc[-1]:.2%}")
                    col2.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                    col3.metric("Max Drawdown", f"{max_drawdown:.2%}")
                    
                    # Plot portfolio growth
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=cumulative_return.index,
                        y=(1 + cumulative_return) * portfolio_value,
                        name='Portfolio Value',
                        line=dict(color='green', width=2)
                    ))
                    
                    fig.update_layout(
                        title="Portfolio Growth ($10,000 Initial)",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value ($)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

def main():
    dashboard = StockDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()