from datetime import datetime

class PriceAlertSystem:
    def __init__(self):
        self.alerts = {}
        
    # Nouveaux champs dans la classe PriceAlertSystem
def add_alert(self, symbol, price, condition, user_id, percentage=None, expiry=None):
    try:
        if symbol not in self.alerts:
            self.alerts[symbol] = []
            
        alert = {
            'price': float(price),
            'condition': condition,  # 'above', 'below', 'percent_change'
            'user_id': user_id,
            'active': True,
            'percentage': percentage,  # Pour les alertes de variation en pourcentage
            'expiry': expiry,  # Date d'expiration de l'alerte
            'created_at': datetime.now().isoformat()
        }
        self.alerts[symbol].append(alert)
        return len(self.alerts[symbol]) - 1
    except Exception as e:
        print(f"Erreur dans add_alert: {e}")
        return -1
        
    def check_alerts(self, symbol, current_price):
        """Vérifie les alertes pour un symbole donné"""
        try:
            if symbol not in self.alerts:
                return []
                
            triggered_alerts = []
            current_price = float(current_price)
            
            for idx, alert in enumerate(self.alerts[symbol]):
                if not alert['active']:
                    continue
                    
                if ((alert['condition'] == 'above' and current_price > alert['price']) or
                    (alert['condition'] == 'below' and current_price < alert['price'])):
                    triggered_alerts.append({
                        'alert_id': idx,
                        'symbol': symbol,
                        'target_price': alert['price'],
                        'current_price': current_price,
                        'user_id': alert['user_id'],
                        'condition': alert['condition']
                    })
                    alert['active'] = False
                    
            return triggered_alerts
        except Exception as e:
            print(f"Erreur dans check_alerts: {e}")
            return []

    def get_active_alerts(self, symbol=None):
        """Récupère toutes les alertes actives"""
        try:
            if symbol:
                return [alert for alert in self.alerts.get(symbol, []) if alert['active']]
            
            active_alerts = {}
            for sym, alerts in self.alerts.items():
                active_alerts[sym] = [alert for alert in alerts if alert['active']]
            return active_alerts
        except Exception as e:
            print(f"Erreur dans get_active_alerts: {e}")
            return {}