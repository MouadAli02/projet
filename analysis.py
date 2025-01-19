import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime

class MarketAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        for col in ['1. open', '2. high', '3. low', '4. close', '5. volume']:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        self.data = self.data.sort_index()
        self.current_price = float(self.data['4. close'].iloc[-1])
        self.prev_price = float(self.data['4. close'].iloc[-2])
        self.last_update = str(self.data.index[-1])

    def calculate_rsi(self, period: int = 14) -> float:
        close_price = self.data['4. close']
        deltas = close_price.diff()
        
        gains = deltas.copy()
        losses = deltas.copy()
        
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        avg_gains = gains.rolling(window=period, min_periods=1).mean()
        avg_losses = losses.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1])

    def analyze_trend(self) -> Dict[str, Any]:
        close_prices = self.data['4. close']
        min_periods = min(3, len(close_prices))
        
        sma20 = close_prices.rolling(window=20, min_periods=min_periods).mean().iloc[-1]
        sma50 = close_prices.rolling(window=50, min_periods=min_periods).mean().iloc[-1]
        sma200 = close_prices.rolling(window=200, min_periods=min_periods).mean().iloc[-1]
        
        rsi = self.calculate_rsi()
        momentum = ((self.current_price / self.prev_price) - 1) * 100
        
        direction_court = 'HAUSSIÈRE' if self.current_price > sma20 else 'BAISSIÈRE'
        direction_moyen = 'HAUSSIÈRE' if self.current_price > sma50 else 'BAISSIÈRE'
        direction_long = 'HAUSSIÈRE' if self.current_price > sma200 else 'BAISSIÈRE'
        
        strength_score = 0
        
        # Points basés sur les directions
        for direction in [direction_court, direction_moyen, direction_long]:
            if direction == 'BAISSIÈRE':
                strength_score -= 2
            else:
                strength_score += 1
                
        # Points basés sur le momentum
        if momentum < -2:
            strength_score -= 2
        elif momentum < 0:
            strength_score -= 1
        elif momentum > 2:
            strength_score += 2
        elif momentum > 0:
            strength_score += 1
            
        # Points basés sur le RSI
        if rsi < 30:
            strength_score -= 1  # Réduit l'impact du RSI bas
        elif rsi > 70:
            strength_score += 1
            
        # Mapping du score vers la force
        strength = ("TRÈS FAIBLE" if strength_score <= -6 else
                   "FAIBLE" if strength_score <= -3 else
                   "MODÉRÉE" if strength_score <= 1 else
                   "FORTE" if strength_score <= 4 else "TRÈS FORTE")
        
        return {
            'current_price': self.current_price,
            'direction': direction_court,
            'direction_moyen_terme': direction_moyen,
            'direction_long_terme': direction_long,
            'strength': strength,
            'strength_score': strength_score,
            'momentum': float(momentum),
            'rsi': float(rsi),
            'sma20': float(sma20),
            'sma50': float(sma50),
            'sma200': float(sma200)
        }

    def calculate_volatility(self) -> Dict[str, Any]:
        returns = self.data['4. close'].pct_change()
        min_periods = min(5, len(returns))
        
        current_vol = returns.tail(5).std() * np.sqrt(252)
        daily_vol = returns.std() * np.sqrt(252)
        weekly_vol = returns.rolling(5, min_periods=min_periods).std().mean() * np.sqrt(52)
        monthly_vol = returns.rolling(21, min_periods=min_periods).std().mean() * np.sqrt(12)
        
        vol_ratio = current_vol / daily_vol if daily_vol != 0 else 1
        
        trend = ('CROISSANTE' if vol_ratio > 1.1 else
                'DÉCROISSANTE' if vol_ratio < 0.9 else 'STABLE')
        
        return {
            'current': float(current_vol),
            'daily': float(daily_vol),
            'weekly': float(weekly_vol),
            'monthly': float(monthly_vol),
            'trend': trend,
            'volatility_ratio': float(vol_ratio)
        }

    def calculate_support_resistance(self) -> Dict[str, Any]:
        recent_data = self.data.tail(20)
        recent_high = float(recent_data['2. high'].max())
        recent_low = float(recent_data['3. low'].min())
        
        pivot = (recent_high + recent_low + self.current_price) / 3
        range_price = recent_high - recent_low
        
        resistance1 = pivot + (range_price * 0.382)
        resistance2 = pivot + (range_price * 0.618)
        support1 = pivot - (range_price * 0.382)
        support2 = pivot - (range_price * 0.618)
        
        return {
            'pivot_point': float(pivot),
            'resistance1': float(resistance1),
            'resistance2': float(resistance2),
            'support1': float(support1),
            'support2': float(support2),
            'recent_high': recent_high,
            'recent_low': recent_low
        }

    def generate_summary(self) -> Dict[str, Any]:
        trend = self.analyze_trend()
        volatility = self.calculate_volatility()
        
        volume = self.data['5. volume']
        current_volume = float(volume.iloc[-1])
        avg_volume = float(volume.rolling(window=20, min_periods=1).mean().iloc[-1])
        
        volume_trend = 'CROISSANT' if current_volume > avg_volume else 'DÉCROISSANT'
        change_percent = ((self.current_price - self.prev_price) / self.prev_price) * 100
        
        # Calcul des signaux
        buy_signals = 0
        sell_signals = 0
        
        # RSI
        if trend['rsi'] < 30:
            buy_signals += 2
        elif trend['rsi'] > 70:
            sell_signals += 2
            
        # Tendance
        if trend['direction'] == 'BAISSIÈRE':
            sell_signals += 2
        else:
            buy_signals += 1
            
        if trend['direction_moyen_terme'] == 'BAISSIÈRE':
            sell_signals += 1
        else:
            buy_signals += 1
            
        # Momentum
        if trend['momentum'] < -1:
            sell_signals += 2
        elif trend['momentum'] > 1:
            buy_signals += 2
            
        # Volume
        if volume_trend == 'CROISSANT':
            if change_percent > 0:
                buy_signals += 1
            else:
                sell_signals += 1
                
        # Volatilité
        if volatility['trend'] == 'CROISSANTE':
            sell_signals += 1

        # Nouvelle logique de recommandation
        if trend['strength_score'] <= -4:
            if trend['rsi'] < 30:
                if sell_signals > buy_signals * 2:
                    recommendation = 'VENTE (MALGRÉ SURVENTE)'
                elif sell_signals > buy_signals:
                    recommendation = 'NEUTRE (SURVENTE)'
                else:
                    recommendation = 'ACHAT (SURVENTE)'
            else:
                recommendation = 'VENTE FORTE'
        elif trend['strength_score'] <= -2:
            recommendation = 'VENTE'
        elif trend['strength_score'] >= 4:
            if trend['rsi'] > 70:
                recommendation = 'VENTE (SURACHAT)'
            else:
                recommendation = 'ACHAT FORT'
        elif trend['strength_score'] >= 2:
            recommendation = 'ACHAT'
        else:
            recommendation = 'NEUTRE'
        
        return {
            'current_price': self.current_price,
            'change_percent': float(change_percent),
            'volume': int(current_volume),
            'average_volume': float(avg_volume),
            'volume_trend': volume_trend,
            'recommendation': recommendation,
            'market_sentiment': 'POSITIF' if change_percent > 0 and volume_trend == 'CROISSANT' else 'NÉGATIF',
            'last_update': self.last_update,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'trend_strength': trend['strength'],
            'trend_consensus': 'FORTEMENT ' + trend['direction'] if all(d == trend['direction'] 
                for d in [trend['direction'], trend['direction_moyen_terme'], trend['direction_long_terme']]) 
                else 'MIXTE'
        }

def perform_detailed_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    try:
        analyzer = MarketAnalyzer(data)
        
        return {
            'trend': analyzer.analyze_trend(),
            'volatility': analyzer.calculate_volatility(),
            'support_resistance': analyzer.calculate_support_resistance(),
            'summary': analyzer.generate_summary()
        }
        
    except Exception as e:
        print(f"Erreur dans l'analyse détaillée: {e}")
        return {'error': str(e)}