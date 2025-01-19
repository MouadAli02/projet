from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from utils.indicators import calculate_indicators
from utils.analysis import perform_detailed_analysis
from utils.predictions import StockPredictor
from utils.alerts import PriceAlertSystem
import os
import json
from datetime import datetime, timedelta


app = Flask(__name__)
CORS(app)

API_KEY = 'BAC3H4KV8DQ9ZHK9'
ts = TimeSeries(key=API_KEY, output_format='pandas')

# Système d'alertes
alert_system = PriceAlertSystem()

@app.route('/test')
def test_connection():
    return jsonify({'status': 'ok'})

@app.route('/stock/<symbol>')
def get_stock_data(symbol):
    try:
        data, meta = ts.get_daily(symbol=symbol)
        return jsonify(data.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/historical/<symbol>')
def get_historical_data(symbol):
    try:
        data, meta = ts.get_daily(symbol=symbol, outputsize='full')
        result = data.reset_index().to_dict('records')
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/<symbol>')
def get_predictions(symbol):
    try:
        data, meta = ts.get_daily(symbol=symbol)
        predictor = StockPredictor(data)
        predictor.train_model()
        predictions = predictor.predict_next_days(30)
        future_dates = predictor.get_prediction_dates(30)
        
        last_price = float(data['4. close'].iloc[-1])
        changes = [(price - last_price) / last_price * 100 for price in predictions]
        
        return jsonify({
            'predictions': predictions,
            'dates': future_dates,
            'last_price': last_price,
            'percent_changes': changes
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/indicators/<symbol>')
def get_technical_indicators(symbol):
    try:
        data, meta = ts.get_daily(symbol=symbol)
        indicators = calculate_indicators(data)
        return jsonify(indicators)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analysis/<symbol>')
def get_detailed_analysis(symbol):
    try:
        data, meta = ts.get_daily(symbol=symbol)
        analysis = perform_detailed_analysis(data)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/alerts', methods=['POST'])
def create_alert():
    try:
        data = request.get_json()
        required_fields = ['symbol', 'price', 'condition', 'user_id']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        alert_id = alert_system.add_alert(
            data['symbol'],
            data['price'],
            data['condition'],
            data['user_id']
        )
        return jsonify({'alert_id': alert_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/alerts/check/<symbol>')
def check_alerts(symbol):
    try:
        data, meta = ts.get_daily(symbol=symbol)
        current_price = float(data['4. close'].iloc[-1])
        triggered = alert_system.check_alerts(symbol, current_price)
        return jsonify({
            'alerts': triggered,
            'current_price': current_price
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)