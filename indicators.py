import pandas as pd
import numpy as np
from typing import Dict, Any

class TechnicalIndicators:
    def __init__(self, data: pd.DataFrame):
        """Initialise et prépare les données pour l'analyse"""
        self.data = data.copy()
        # Inverser l'ordre pour avoir les données les plus anciennes en premier
        self.data = self.data.sort_index()
        
        # Convertir les colonnes en numérique
        for col in ['1. open', '2. high', '3. low', '4. close', '5. volume']:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
        self.current_price = float(self.data['4. close'].iloc[-1])
        self.last_update = str(self.data.index[-1])

    def calculate_moving_averages(self) -> Dict[str, Any]:
        """Calcule les moyennes mobiles"""
        averages = {}
        prices = self.data['4. close']
        
        for period in [5, 10, 20, 50, 100, 200]:
            # Simple Moving Average
            sma = prices.rolling(window=period).mean()
            sma_value = sma.iloc[-1] if not pd.isna(sma.iloc[-1]) else self.current_price
            
            # Exponential Moving Average
            ema = prices.ewm(span=period, adjust=False).mean()
            ema_value = ema.iloc[-1] if not pd.isna(ema.iloc[-1]) else self.current_price
            
            averages[f'sma_{period}'] = float(sma_value)
            averages[f'ema_{period}'] = float(ema_value)

        return {
            'averages': averages,
            'current_price': self.current_price,
            'last_update': self.last_update
        }

    def calculate_rsi(self, period: int = 14) -> Dict[str, Any]:
        """Calcule le RSI"""
        prices = self.data['4. close']
        deltas = prices.diff()
        
        gain = deltas.copy()
        loss = deltas.copy()
        
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        
        return {
            'rsi': current_rsi,
            'current_price': self.current_price,
            'last_update': self.last_update
        }

    def calculate_macd(self) -> Dict[str, Any]:
        """Calcule le MACD"""
        prices = self.data['4. close']
        
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd_line': float(macd_line.iloc[-1]),
            'signal_line': float(signal_line.iloc[-1]),
            'histogram': float(histogram.iloc[-1]),
            'current_price': self.current_price,
            'last_update': self.last_update
        }

    def calculate_bollinger_bands(self) -> Dict[str, Any]:
        """Calcule les bandes de Bollinger"""
        prices = self.data['4. close']
        
        sma = prices.rolling(window=20).mean()
        std = prices.rolling(window=20).std()
        
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        
        return {
            'middle': float(sma.iloc[-1]),
            'upper': float(upper_band.iloc[-1]),
            'lower': float(lower_band.iloc[-1]),
            'current_price': self.current_price,
            'last_update': self.last_update
        }

    def calculate_stochastic(self) -> Dict[str, Any]:
        """Calcule l'oscillateur stochastique"""
        low_14 = self.data['3. low'].rolling(window=14).min()
        high_14 = self.data['2. high'].rolling(window=14).max()
        
        k = 100 * ((self.data['4. close'] - low_14) / (high_14 - low_14))
        d = k.rolling(window=3).mean()
        
        return {
            'k': float(k.iloc[-1]) if not pd.isna(k.iloc[-1]) else 50.0,
            'd': float(d.iloc[-1]) if not pd.isna(d.iloc[-1]) else 50.0,
            'current_price': self.current_price,
            'last_update': self.last_update
        }

    def calculate_volume_indicators(self) -> Dict[str, Any]:
        """Calcule les indicateurs de volume"""
        volume = self.data['5. volume']
        close = self.data['4. close']
        
        # On Balance Volume (OBV)
        daily_ret = close.pct_change()
        daily_direction = np.where(daily_ret > 0, 1, -1)
        daily_direction[daily_ret == 0] = 0
        obv = (volume * daily_direction).cumsum()
        
        # Volume Moving Average
        volume_ma = volume.rolling(window=20).mean()
        
        # Volume Weighted Average Price (VWAP)
        typical_price = (close + self.data['2. high'] + self.data['3. low']) / 3
        vwap = (typical_price * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()
        
        return {
            'obv': float(obv.iloc[-1]),
            'volume_ma': float(volume_ma.iloc[-1]),
            'vwap': float(vwap.iloc[-1]),
            'current_volume': float(volume.iloc[-1]),
            'current_price': self.current_price,
            'last_update': self.last_update
        }

    def calculate_support_resistance(self) -> Dict[str, Any]:
        """Calcule les niveaux de support et résistance"""
        high = float(self.data['2. high'].iloc[-1])
        low = float(self.data['3. low'].iloc[-1])
        close = self.current_price
        
        pivot = (high + low + close) / 3
        r1 = (pivot * 2) - low
        r2 = pivot + (high - low)
        s1 = (pivot * 2) - high
        s2 = pivot - (high - low)
        
        return {
            'pivot': float(pivot),
            'r1': float(r1),
            'r2': float(r2),
            's1': float(s1),
            's2': float(s2),
            'current_price': self.current_price,
            'current_high': high,
            'current_low': low,
            'last_update': self.last_update
        }

def calculate_indicators(data: pd.DataFrame) -> Dict[str, Any]:
    """Fonction principale pour calculer tous les indicateurs"""
    try:
        indicators = TechnicalIndicators(data)
        
        result = {
            'moving_averages': indicators.calculate_moving_averages(),
            'rsi': indicators.calculate_rsi(),
            'macd': indicators.calculate_macd(),
            'bollinger': indicators.calculate_bollinger_bands(),
            'stochastic': indicators.calculate_stochastic(),
            'volume': indicators.calculate_volume_indicators(),
            'support_resistance': indicators.calculate_support_resistance()
        }
        
        # Ajouter les signaux
        rsi_value = result['rsi']['rsi']
        macd_hist = result['macd']['histogram']
        bb_upper = result['bollinger']['upper']
        bb_lower = result['bollinger']['lower']
        
        result['signals'] = {
            'rsi_signal': 'SURACHETÉ' if rsi_value > 70 else 'SURVENDU' if rsi_value < 30 else 'NEUTRE',
            'macd_signal': 'ACHAT' if macd_hist > 0 else 'VENTE',
            'bollinger_signal': 'SURACHETÉ' if indicators.current_price > bb_upper else 'SURVENDU' if indicators.current_price < bb_lower else 'NEUTRE'
        }
        
        result['metadata'] = {
            'current_price': indicators.current_price,
            'last_update': indicators.last_update,
            'data_points': len(data)
        }
        
        return result
        
    except Exception as e:
        print(f"Erreur dans le calcul des indicateurs: {e}")
        return {'error': str(e)}