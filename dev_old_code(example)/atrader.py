import numpy as np
from hmmlearn import hmm
import pandas as pd
import loadData as load
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import talib


class ETHTrader:
    def __init__(self, n_states=5, exchange_id='binance', symbol='ETH/USDT', timeframe='1d', limit=500, random_seed=42):
        self.model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000, random_state=random_seed)
        self.states = ["Bullish", "Bearish", "High Volatility", "Low Volatility", "Correction"] 
        self.exchange_id = exchange_id  
        self.symbol = symbol
        self.timeframe = timeframe  
        self.limit = limit   

    def preprocess_data(self):
        # Fetch data
        df = load.fetch_ohlcv_data(self.exchange_id, self.symbol, self.timeframe, self.limit)

        # Calculate indicators
        df = load.calculate_log_returns(df)
        df = load.calculate_volatility(df)
        df = load.calculate_moving_averages(df)

        # Calculate RSI and MACD
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_signal'] = df['rsi'].rolling(window=9).mean()  # Example: 9-day SMA of RSI

        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)

        # Skip the first 50 rows to ensure all moving averages are available
        df = df.iloc[50:]

        # Select relevant features
        features = ['log_returns', 'volume', 'atr', 'bb_high', 'bb_low', 'sma_5', 'sma_20', 'sma_50', 'rsi', 'rsi_signal', 'macd', 'macd_signal', 'macd_hist']
        X = df[features]

        # Handle missing values
        X = X.dropna()

        # Normalize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled, df
    
    
    def train_model(self, features):
        print("Training")
        self.model.fit(features)
        print("Finished training")

    
    def predict_state(self, features):
        print("predicting")
        return self.model.predict(features)
    
    def generate_signals(self, states, df):
        signals = np.zeros_like(states)

        for i in range(5, len(states)):
            # Get the past states for reference
            past_1_day_state = states[i - 1]
            past_3_day_state = states[i - 3]
            past_5_day_state = states[i - 5]

            # Determine future state based on current and past states
            future_state = states[i]  # you can apply some logic here based on past states if needed

            # Implement your strategy

            # RSI and MACD Crossovers
            rsi_cross = df['rsi'].iloc[i] > df['rsi_signal'].iloc[i]
            macd_cross = df['macd'].iloc[i] > df['macd_signal'].iloc[i]

            # Moving Averages Crossovers
            sma_5_above_20 = df['sma_5'].iloc[i] > df['sma_20'].iloc[i]
            sma_20_above_50 = df['sma_20'].iloc[i] > df['sma_50'].iloc[i]

            # Bullish Market (Uptrend)
            if future_state == 0:
                if rsi_cross and macd_cross:
                    signals[i] = 2  # Buy
                if sma_5_above_20 and sma_20_above_50:
                    signals[i] = 5  # Buy a lot
                elif sma_5_above_20:
                    signals[i] = 1  # Buy a little

            # Bearish Market (Downtrend)
            elif future_state == 1:
                if not rsi_cross and macd_cross:
                    signals[i] = -0.1  # Sell
                if not sma_5_above_20 and sma_20_above_50:
                    signals[i] = -0.5  # Sell a lot
                elif not sma_5_above_20:
                    signals[i] = -0.1  # Sell a little

            # High Volatility (Choppy Market)
            elif future_state == 2:
                signals[i] = 0  # Hold (or implement a more aggressive strategy based on risk)

            # Low Volatility (Stable Market)
            elif future_state == 3:
                signals[i] = 0  # Hold (could prepare for a breakout in either direction)

            # Market Correction/Consolidation
            elif future_state == 4:
                if not rsi_cross and not macd_cross:
                    signals[i] = 0  # Sell (in expectation of further correction)
                else:
                    signals[i] = 0  # Hold

        return signals

    
    def generate_signals1(self, states, df):
        signals = np.zeros_like(states)
        for i in range(len(states)):
            if states[i] == 0:  # Bullish
                if df['rsi'].iloc[i] < 55 or df['macd'].iloc[i] > df['macd_signal'].iloc[i]:
                    signals[i] = -1  # Buy
            elif states[i] == 1:  # Bearish
                if df['rsi'].iloc[i] > 90 or df['macd'].iloc[i] < df['macd_signal'].iloc[i]:
                    signals[i] = 1  # Sell/Short
            elif states[i] == 2:  # High Volatility
                signals[i] = 0  # Hold
            elif states[i] == 3:  # Low Volatility
                signals[i] = 0  # Hold
            elif states[i] == 4:  # Correction
                if df['rsi'].iloc[i] > 90 or df['macd'].iloc[i] < df['macd_signal'].iloc[i]:
                    signals[i] = 0  # Sell/Short
        return signals
    
    def generate_signals2(self, states, df):
        signals = np.zeros_like(states)
        for i in range(len(states)):
            if states[i] == 0:  # Assuming this state is Bullish
                signals[i] = 2  # Buy
            elif states[i] == 1:  # Assuming this state is Bearish
                signals[i] = -2  # Sell/Short
            elif states[i] == 2:  # Assuming this state is High Volatility
                signals[i] = 0  # Hold
            elif states[i] == 3:  # Assuming this state is Low Volatility
                signals[i] = 1  # Hold
            elif states[i] == 4:  # Assuming this state is Correction
                signals[i] = -1  # Sell/Short (or Hold depending on strategy)
        return signals

    def backtest(self):
        # Preprocess data
        X_scaled, df = self.preprocess_data()

        # Split data into training and testing sets
        split_idx = int(len(X_scaled) * 0.7)  # 70% for training, 30% for testing
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        df_train, df_test = df.iloc[:split_idx], df.iloc[split_idx:]

        # Train the model on the training set
        self.train_model(X_train)

        # Predict states on the testing set
        predicted_states = self.predict_state(X_test)
        
        # Align df_test with the length of predicted_states
        df_test = df_test.iloc[:len(predicted_states)].copy()
        
        df_test['predicted_state'] = predicted_states

        # Generate signals
        signals = self.generate_signals(predicted_states, df_test)
        df_test['signal'] = signals

        # Calculate returns based on signals
        df_test['strategy_returns'] = df_test['log_returns'] * df_test['signal'].shift(1)

        # Calculate cumulative returns
        df_test['cumulative_strategy_returns'] = (df_test['strategy_returns'] + 1).cumprod()
        df_test['cumulative_market_returns'] = (df_test['log_returns'] + 1).cumprod()

        # Print or plot results
        print("Backtesting Results:")
        #print(df_test[['cumulative_strategy_returns', 'cumulative_market_returns']])
        
        # Plotting can be done using matplotlib
        df_test[['cumulative_strategy_returns', 'cumulative_market_returns']].plot(title='Backtesting Results')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.show()
        return df_test, predicted_states
    
    def trade(self):
        features, df = self.preprocess_data()
        state = self.predict_state(features)
        df['predicted_state'] = state
        #signal = self.generate_signals(state)
        # Execute trade based on signal
        pass

    def print_model_parameters(self):
        # Print transition matrix
        print("Transition Matrix:")
        print(self.model.transmat_)
        print("\nMeans and Covariances of emissions:")
        for i in range(self.model.n_components):
            print(f"Hidden State {i}:")
            print(f"Mean: {self.model.means_[i]}")
            print(f"Covariance: {self.model.covars_[i]}")
            print()


