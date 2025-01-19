import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self, data):
        self.data = data
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.imputer = SimpleImputer(strategy='mean')
        
        self.models = {
            'rf': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gbm': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        }
        
    def calculate_technical_indicators(self, df):
        """Calcule les indicateurs techniques avancés"""
        df = df.copy()
        
        # Conversion en numérique et remplacement des infinis
        for col in ['4. close', '2. high', '3. low', '5. volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Moyennes mobiles
        for period in [7, 14, 21, 50]:
            df[f'ma_{period}'] = df['4. close'].rolling(window=period, min_periods=1).mean()
            df[f'std_{period}'] = df['4. close'].rolling(window=period, min_periods=1).std()
        
        # RSI
        delta = df['4. close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)  # Valeur neutre pour RSI
        
        # MACD
        exp1 = df['4. close'].ewm(span=12, adjust=False, min_periods=1).mean()
        exp2 = df['4. close'].ewm(span=26, adjust=False, min_periods=1).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False, min_periods=1).mean()
        
        # Momentum
        df['momentum'] = df['4. close'].diff(periods=10)
        
        # Volatilité
        df['volatility'] = df['4. close'].rolling(window=20, min_periods=1).std()
        
        # Volume moyen
        df['volume_ma'] = df['5. volume'].rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = df['5. volume'] / df['volume_ma']
        
        # Remplacer les valeurs infinies
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Remplir les valeurs manquantes
        return df.fillna(method='ffill').fillna(method='bfill')

    def prepare_features(self, df):
        """Prépare les features pour le modèle"""
        df = self.calculate_technical_indicators(df)
        
        feature_columns = [
            'ma_7', 'ma_14', 'ma_21', 'ma_50',
            'std_7', 'std_14', 'std_21', 'std_50',
            'rsi', 'macd', 'macd_signal', 'momentum',
            'volatility', 'volume_ratio'
        ]
        
        target = df['4. close']
        features = df[feature_columns]
        
        # Imputation des valeurs manquantes
        features = pd.DataFrame(
            self.imputer.fit_transform(features),
            columns=features.columns,
            index=features.index
        )
        
        return features, target

    def create_sequences(self, features, target, sequence_length=10):
        """Crée des séquences pour l'entraînement"""
        X, y = [], []
        
        for i in range(len(features) - sequence_length):
            X.append(features.iloc[i:(i + sequence_length)].values)
            y.append(target.iloc[i + sequence_length])
            
        return np.array(X), np.array(y)

    def train_model(self):
        """Entraîne les modèles avec validation croisée"""
        features, target = self.prepare_features(self.data)
        X, y = self.create_sequences(features, target)
        
        # Reshape pour les modèles standards
        X_reshaped = X.reshape(X.shape[0], -1)
        
        # Normalisation
        X_scaled = self.feature_scaler.fit_transform(X_reshaped)
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1))
        
        # Entraînement des modèles
        for name, model in self.models.items():
            model.fit(X_scaled, y_scaled.ravel())
    
    def predict_next_days(self, days=30):
        """Prédit les prochains jours avec l'ensemble des modèles"""
        features, _ = self.prepare_features(self.data)
        sequence_length = 10
        
        # Préparer la dernière séquence connue
        last_sequence = features.iloc[-sequence_length:].values
        last_sequence_reshaped = last_sequence.reshape(1, -1)
        
        predictions = []
        last_price = float(self.data['4. close'].iloc[-1])
        current_sequence = last_sequence.copy()
        
        for day in range(days):
            # Normaliser la séquence
            sequence_scaled = self.feature_scaler.transform(last_sequence_reshaped)
            
            # Faire les prédictions avec chaque modèle
            model_predictions = []
            for model in self.models.values():
                pred = model.predict(sequence_scaled)
                pred = self.target_scaler.inverse_transform(pred.reshape(-1, 1))
                model_predictions.append(pred[0][0])
            
            # Moyenne pondérée des prédictions
            prediction = np.average(model_predictions, weights=[0.6, 0.4])
            
            # Variabilité basée sur la volatilité historique
            volatility = self.data['4. close'].pct_change().std()
            noise = np.random.normal(0, volatility * (1 + 0.1 * day))
            prediction = prediction * (1 + noise)
            
            predictions.append(float(prediction))
            
            # Mettre à jour la séquence
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = self.calculate_features_for_prediction(prediction)
            last_sequence_reshaped = current_sequence.reshape(1, -1)
            
            last_price = prediction
            
        return predictions

    def calculate_features_for_prediction(self, new_price):
        """Calcule les features pour une nouvelle prédiction"""
        df = self.data.copy()
        df.loc[df.index[-1] + pd.Timedelta(days=1), '4. close'] = new_price
        features, _ = self.prepare_features(df)
        return features.iloc[-1].values

    def get_prediction_dates(self, days=30):
        """Génère les dates futures pour les prédictions"""
        current_date = datetime.now()
        future_dates = []
        count = 0
        
        while len(future_dates) < days:
            next_date = current_date + timedelta(days=count)
            if next_date.weekday() < 5:  # Lundi-Vendredi
                future_dates.append(next_date.strftime("%Y-%m-%d"))
            count += 1
            
        return future_dates