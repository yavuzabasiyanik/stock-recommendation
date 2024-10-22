from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import aiohttp
from functools import lru_cache
from threading import Lock
import logging

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

class Cache:
    def __init__(self, expiry_minutes=5):
        self.data = {}
        self.lock = Lock()
        self.expiry_minutes = expiry_minutes

    def set(self, key, value):
        with self.lock:
            self.data[key] = {
                'value': value,
                'timestamp': datetime.now()
            }

    def get(self, key):
        with self.lock:
            if key in self.data:
                entry = self.data[key]
                if datetime.now() - entry['timestamp'] < timedelta(minutes=self.expiry_minutes):
                    return entry['value']
                del self.data[key]
            return None

# Global cache instance
cache = Cache()

class MarketScanner:
    def __init__(self):
        self.min_price = 1.0
        self.min_volume = 50000
        self.batch_size = 20
        self.max_concurrent = 10
        self.session = None
        self._tickers = None
        self._last_ticker_update = None

    def _calculate_signals(self, hist):
        """Calculate trading signals with proper validation"""
        try:
            if len(hist) < 2:
                return {
                    'price_trending_up': False,
                    'volume_increasing': False,
                    'above_moving_avg': False,
                    'momentum_positive': False
                }

            closes = hist['Close'].values
            volumes = hist['Volume'].values
            
            # Ensure we have enough data for calculations
            min_periods = min(5, len(closes))
            sma_5 = np.mean(closes[-min_periods:])
            volume_avg = np.mean(volumes[-min_periods:])
            
            signals = {
                'price_trending_up': bool(closes[-1] > sma_5),
                'volume_increasing': bool(volumes[-1] > volume_avg),
                'above_moving_avg': bool(closes[-1] > np.mean(closes)),
                'momentum_positive': bool(closes[-1] > closes[-2])
            }
            
            return signals
        except Exception as e:
            logging.error(f"Error calculating signals: {e}")
            return {
                'price_trending_up': False,
                'volume_increasing': False,
                'above_moving_avg': False,
                'momentum_positive': False
            }

    def _calculate_trend_strength(self, hist):
        """Calculate trend strength with proper validation"""
        try:
            if len(hist) < 2:
                return 0

            closes = hist['Close'].values
            volumes = hist['Volume'].values
            
            score = 0
            
            # Only calculate if we have enough data
            if len(closes) >= 3:
                if closes[-1] > np.mean(closes[-3:]):
                    score += 1
            
            if len(closes) >= 2:
                if closes[-1] > closes[-2]:
                    score += 1
                    
            if len(volumes) >= 2:
                if volumes[-1] > volumes[-2]:
                    score += 1
                    
            if len(closes) >= 4:
                if closes[-1] >= np.percentile(closes, 75):
                    score += 1
                
                if closes[-1] > np.mean(closes):
                    score += 1
                    
            return score
                
        except Exception as e:
            logging.error(f"Error calculating trend strength: {e}")
            return 0

    def _calculate_confidence(self, analysis):
        """Calculate overall confidence score"""
        try:
            confidence = 50.0  # Base confidence
            
            # Add trend strength contribution
            confidence += float(analysis.get('trend_strength', 0) * 5)
            
            # Add RSI contribution
            rsi = analysis.get('rsi', 50.0)
            if rsi < 30:  # Oversold
                confidence += 20.0
            elif rsi > 70:  # Overbought
                confidence -= 20.0
                
            # Add recent performance contribution
            change_1d = analysis.get('change_1d', 0)
            if change_1d > 0:
                confidence += min(change_1d, 10)
                
            # Add technical signals contribution
            signals = analysis.get('signals', {})
            for signal in signals.values():
                if signal:
                    confidence += 5.0
                    
            # Ensure confidence stays within 0-100 range
            return float(min(max(confidence, 0.0), 100.0))
            
        except Exception as e:
            logging.error(f"Error calculating confidence: {e}")
            return 50.0

    def _get_empty_response(self):
        """Return empty response with proper structure"""
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'market_status': 'Market is open' if self._is_market_open() else 'Market is closed',
            'stats': {
                'total_stocks_analyzed': 0,
                'average_confidence': 0.0,
                'market_trend': 0.0
            },
            'recommendations': []
        }

    @lru_cache(maxsize=1)
    def get_all_tickers(self):
        """Get all tradeable stock tickers with caching"""
        now = datetime.now()
        if (self._tickers is None or 
            self._last_ticker_update is None or 
            (now - self._last_ticker_update).hours >= 1):
            
            try:
                # Get S&P 500 stocks
                sp500_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
                sp500_df = pd.read_csv(sp500_url)
                tickers = set(sp500_df['Symbol'].tolist())

                # Add NASDAQ 100 and other stocks
                nasdaq100_tickers = [
                    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META'
                    # Add more tickers as needed
                ]
                tickers.update(nasdaq100_tickers)

                self._tickers = sorted(list(tickers))
                self._last_ticker_update = now
                
            except Exception as e:
                logging.error(f"Error fetching tickers: {e}")
                self._tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
                
        return self._tickers

    async def init_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def fetch_stock_data(self, symbol):
        """Fetch stock data for a single symbol"""
        try:
            stock = yf.Ticker(symbol)
            # Use 5d to ensure we have enough data
            hist = stock.history(period="5d")
            
            # Validate we have enough data
            if hist.empty or len(hist) < 2:
                logging.info(f"Insufficient data for {symbol}")
                return None

            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) >= 2 else current_price
            
            if current_price < self.min_price or hist['Volume'].iloc[-1] < self.min_volume:
                return None

            analysis = {
                'symbol': symbol,
                'price': float(current_price),
                'volume': float(hist['Volume'].iloc[-1]),
                'change_1d': float(((current_price - prev_close) / prev_close) * 100)
            }

            # Add technical indicators only if we have enough data
            if len(hist) >= 2:
                analysis.update(self._calculate_technical_indicators(hist))
                analysis['signals'] = self._calculate_signals(hist)
                analysis['trend_strength'] = self._calculate_trend_strength(hist)
                analysis['confidence'] = self._calculate_confidence(analysis)
                analysis['action'] = 'Buy' if analysis['confidence'] > 60 else 'Wait'
            else:
                # Default values if insufficient data
                analysis.update({
                    'rsi': 50.0,
                    'sma_5': float(current_price),
                    'volatility': 0.0,
                    'signals': {
                        'price_trending_up': False,
                        'volume_increasing': False,
                        'above_moving_avg': False,
                        'momentum_positive': False
                    },
                    'trend_strength': 0,
                    'confidence': 50.0,
                    'action': 'Wait'
                })

            return analysis

        except Exception as e:
            logging.error(f"Error analyzing {symbol}: {e}")
            return None

    def _calculate_technical_indicators(self, hist):
        """Calculate technical indicators with proper validation"""
        try:
            if len(hist) < 2:
                return {'rsi': 50.0, 'sma_5': hist['Close'].iloc[-1], 'volatility': 0.0}

            closes = hist['Close'].values
            
            # Calculate deltas only if we have enough data
            deltas = np.diff(closes)
            gains = np.where(deltas >= 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # Handle division by zero
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            
            # Calculate RSI
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss if avg_loss != 0 else 0
                rsi = 100 - (100 / (1 + rs))

            # Calculate volatility with validation
            volatility = (np.std(deltas) / closes[-1] * 100) if len(deltas) > 0 else 0

            return {
                'rsi': float(rsi),
                'sma_5': float(np.mean(closes[-min(5, len(closes)):])),
                'volatility': float(volatility)
            }
        except Exception as e:
            logging.error(f"Error in technical indicators: {e}")
            return {'rsi': 50.0, 'sma_5': 0.0, 'volatility': 0.0}

    async def scan_market(self):
        """Main market scanning function"""
        cache_key = 'scan_results_daily'
        cached_results = cache.get(cache_key)
        if cached_results:
            return cached_results

        tickers = self.get_all_tickers()
        results = []
        
        # Process in batches
        for i in range(0, len(tickers), self.batch_size):
            batch = tickers[i:i + self.batch_size]
            tasks = [self.fetch_stock_data(symbol) for symbol in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend([r for r in batch_results if r])
            
            # Add small delay between batches
            await asyncio.sleep(0.1)

        # Sort and prepare response
        if results:
            results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            response_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'market_status': 'Market is open' if self._is_market_open() else 'Market is closed',
                'stats': self._calculate_stats(results),
                'recommendations': results[:50]  # Return top 50 recommendations
            }
            
            # Cache the results
            cache.set(cache_key, response_data)
            return response_data
        
        return self._get_empty_response()

    def _is_market_open(self):
        """Check if market is currently open"""
        now = datetime.now()
        return 9 <= now.hour < 16

    def _calculate_stats(self, results):
        """Calculate overall market statistics"""
        if not results:
            return {
                'total_stocks_analyzed': 0,
                'average_confidence': 0.0,
                'market_trend': 0.0
            }
            
        return {
            'total_stocks_analyzed': len(results),
            'average_confidence': float(np.mean([r.get('confidence', 50.0) for r in results])),
            'market_trend': float(np.mean([r.get('trend_strength', 0) for r in results]))
        }

@app.route('/scan/daily', methods=['GET'])
async def scan_market():
    """Endpoint for daily market scan"""
    try:
        scanner = MarketScanner()
        await scanner.init_session()
        results = await scanner.scan_market()
        await scanner.close_session()
        return jsonify(results)
    except Exception as e:
        logging.error(f"Error in scan_market endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/tickers', methods=['GET'])
def get_tickers():
    """Endpoint to get list of available tickers"""
    try:
        scanner = MarketScanner()
        tickers = scanner.get_all_tickers()
        return jsonify({
            'status': 'success',
            'count': len(tickers),
            'tickers': tickers
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'tickers': []
        }), 500

if __name__ == '__main__':
    app.run(debug=True)