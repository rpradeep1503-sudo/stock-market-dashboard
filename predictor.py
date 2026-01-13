# In predictor.py, modify ProphetPredictor to not require Prophet
class ProphetPredictor:
    def __init__(self):
        self.is_available = False  # Always use fallback on cloud
    
    def predict(self, df, days=7):
        """Simple trend-based prediction (Prophet fallback)"""
        last_price = float(df['Close'].iloc[-1])
        
        # Calculate simple trend
        if len(df) >= 20:
            trend = (float(df['Close'].iloc[-1]) - float(df['Close'].iloc[-20])) / 20
        else:
            trend = 0
        
        predictions = []
        for i in range(1, days + 1):
            pred = last_price + (trend * i)
            # Add small random variation
            variation = np.random.normal(0, last_price * 0.01)
            predictions.append(pred + variation)
        
        return np.array(predictions)