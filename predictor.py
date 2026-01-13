# predictor.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = MinMaxScaler()
    
    def train_model(self, df):
        """
        Train a simple linear regression model on stock data
        Returns: (train_score, test_score)
        """
        try:
            if len(df) < 10:
                return 0.0, 0.0
            
            # Use closing prices
            prices = df['Close'].values.reshape(-1, 1)
            
            # Scale the data
            scaled_prices = self.scaler.fit_transform(prices)
            
            # Create sequences (use last 5 days to predict next day)
            X, y = [], []
            sequence_length = 5
            
            for i in range(sequence_length, len(scaled_prices)):
                X.append(scaled_prices[i-sequence_length:i, 0])
                y.append(scaled_prices[i, 0])
            
            X, y = np.array(X), np.array(y)
            
            # Split into train/test (80/20)
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Calculate scores
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            return max(train_score, 0), max(test_score, 0)
            
        except Exception as e:
            print(f"Error training model: {e}")
            return 0.0, 0.0
    
    def predict(self, df, days=7):
        """
        Predict future prices
        Returns: numpy array of predicted prices
        """
        try:
            if len(df) < 10:
                # Return array of last price if insufficient data
                last_price = df['Close'].iloc[-1]
                return np.array([last_price] * days)
            
            # Use last sequence_length days for prediction
            prices = df['Close'].values.reshape(-1, 1)
            scaled_prices = self.scaler.transform(prices)
            
            sequence_length = 5
            last_sequence = scaled_prices[-sequence_length:, 0]
            
            predictions_scaled = []
            current_sequence = last_sequence.copy()
            
            for _ in range(days):
                # Reshape for model prediction
                X_pred = current_sequence.reshape(1, -1)
                pred = self.model.predict(X_pred)[0]
                predictions_scaled.append(pred)
                
                # Update sequence (remove first, add prediction)
                current_sequence = np.append(current_sequence[1:], pred)
            
            # Convert back to original scale
            predictions = self.scaler.inverse_transform(
                np.array(predictions_scaled).reshape(-1, 1)
            )
            
            return predictions.flatten()
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            # Fallback: return last price with small random variation
            last_price = df['Close'].iloc[-1]
            return np.array([last_price] * days)

class ProphetPredictor:
    """
    Simplified Prophet predictor (mock version since Prophet can be tricky to install)
    """
    def __init__(self):
        self.is_available = False
        
        # Try to import Prophet
        try:
            from prophet import Prophet
            self.Prophet = Prophet
            self.is_available = True
            print("Prophet is available for predictions")
        except ImportError:
            print("Prophet not installed. Using fallback predictions.")
    
    def predict(self, df, days=7):
        """
        Predict using Prophet or fallback to simple method
        """
        if not self.is_available or len(df) < 30:
            # Fallback to simple prediction
            last_price = df['Close'].iloc[-1]
            trend = (df['Close'].iloc[-1] - df['Close'].iloc[-20]) / 20 if len(df) >= 20 else 0
            
            predictions = []
            for i in range(1, days + 1):
                pred = last_price + (trend * i)
                # Add small random variation
                variation = np.random.normal(0, last_price * 0.01)
                predictions.append(pred + variation)
            
            return np.array(predictions)
        
        try:
            # Prepare data for Prophet
            prophet_df = pd.DataFrame({
                'ds': df.index,
                'y': df['Close'].values
            })
            
            # Create and fit model
            model = self.Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False
            )
            model.fit(prophet_df)
            
            # Make future dataframe
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            
            # Get predictions
            predictions = forecast['yhat'].iloc[-days:].values
            
            return predictions
            
        except Exception as e:
            print(f"Prophet prediction failed: {e}")
            # Fallback
            last_price = df['Close'].iloc[-1]
            return np.array([last_price] * days)

# Test the predictors
if __name__ == "__main__":
    print("Testing Stock Predictor...")
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    sample_df = pd.DataFrame({
        'Close': prices
    }, index=dates)
    
    # Test Linear Regression predictor
    lr_predictor = StockPredictor()
    train_score, test_score = lr_predictor.train_model(sample_df)
    print(f"Linear Regression - Train: {train_score:.2%}, Test: {test_score:.2%}")
    
    predictions = lr_predictor.predict(sample_df, days=5)
    print(f"Predictions (next 5 days): {predictions}")
    
    # Test Prophet predictor
    prophet_predictor = ProphetPredictor()
    prophet_preds = prophet_predictor.predict(sample_df, days=5)
    print(f"Prophet Predictions: {prophet_preds}")