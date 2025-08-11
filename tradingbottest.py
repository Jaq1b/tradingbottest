import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import os
import pytz
from pathlib import Path
from decimal import Decimal, ROUND_DOWN


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketHours:
    def __init__(self):
        self.cst = pytz.timezone('America/Chicago')
        self.est = pytz.timezone('America/New_York')
        
    def is_stock_market_open(self) -> bool:
        est_now = datetime.now(self.est)
        if est_now.weekday() >= 5 or self._is_market_holiday(est_now):
            return False
        
        market_open = est_now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = est_now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= est_now <= market_close
    
    def is_crypto_market_open(self) -> bool:
        return True
    
    def _is_market_holiday(self, est_time: datetime) -> bool:
        year, month, day = est_time.year, est_time.month, est_time.day
        holidays = [(1, 1), (7, 4), (12, 25)]
        
        # Thanksgiving (4th Thursday in November)
        if month == 11:
            first_day = datetime(year, 11, 1)
            first_thursday = 1 + (3 - first_day.weekday()) % 7
            if day == first_thursday + 21:
                return True
        
        return (month, day) in holidays

class OrderType(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending_new"
    FILLED = "filled"
    CANCELLED = "cancelled"

@dataclass
class Trade:
    symbol: str
    entry_price: float
    quantity: float
    timestamp: str
    stop_loss: float
    asset_class: str
    alpaca_order_id: Optional[str] = None
    exit_price: Optional[float] = None
    exit_timestamp: Optional[str] = None
    status: str = "open"
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: Optional[str] = None
    strategy_signals: Optional[str] = None
    actual_quantity: Optional[float] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Trade':
        trade_data = data.copy()
        trade_data.pop('trade_id', None)
        return cls(**trade_data)
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class Order:
    symbol: str
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    timestamp: datetime = None
    status: OrderStatus = OrderStatus.PENDING
    order_id: Optional[str] = None
    asset_class: str = "stock"
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class FileMemory:
    def __init__(self, data_dir: str = "trading_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.trades_file = self.data_dir / "trades.json"
        self.bot_state_file = self.data_dir / "bot_state.json"
        self.performance_file = self.data_dir / "performance.json"
        
        self._init_files()
        logger.info(f"File memory initialized: {self.data_dir}")
    
    def _init_files(self):
        if not self.trades_file.exists():
            self._save_json(self.trades_file, [])
        
        if not self.bot_state_file.exists():
            initial_state = {
                "total_pnl": 0.0, "total_trades": 0, "winning_trades": 0,
                "initial_capital": 1000.0, "peak_portfolio_value": 1000.0,
                "max_drawdown": 0.0, "last_update": datetime.now().isoformat()
            }
            self._save_json(self.bot_state_file, initial_state)
        
        if not self.performance_file.exists():
            self._save_json(self.performance_file, {})
    
    def _load_json(self, file_path: Path) -> Dict | List:
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading {file_path}: {e}")
            return [] if "trades" in str(file_path) else {}
    
    def _save_json(self, file_path: Path, data: Dict | List):
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving {file_path}: {e}")
    
    def save_trade(self, trade: Trade) -> str:
        trades = self._load_json(self.trades_file)
        trade_id = f"{trade.symbol}_{int(time.time())}_{len(trades)}"
        
        trade_data = trade.to_dict()
        trade_data['trade_id'] = trade_id
        
        trades.append(trade_data)
        self._save_json(self.trades_file, trades)
        return trade_id
    
    def update_trade(self, trade: Trade, trade_id: str):
        trades = self._load_json(self.trades_file)
        
        for i, existing_trade in enumerate(trades):
            if existing_trade.get('trade_id') == trade_id:
                updated_data = trade.to_dict()
                updated_data['trade_id'] = trade_id
                trades[i] = updated_data
                break
        
        self._save_json(self.trades_file, trades)
    
    def get_open_trades(self) -> Dict[str, Tuple[Trade, str]]:
        trades = self._load_json(self.trades_file)
        open_trades = {}
        
        for trade_data in trades:
            if trade_data.get('status') == 'open':
                trade = Trade.from_dict(trade_data)
                trade_id = trade_data.get('trade_id')
                open_trades[trade.symbol] = (trade, trade_id)
        
        return open_trades
    
    def get_trade_history(self, days: int = 30) -> List[Trade]:
        trades = self._load_json(self.trades_file)
        
        cutoff_datetime = datetime.now() - timedelta(days=days)
        cutoff_date_str = cutoff_datetime.strftime('%Y-%m-%d')
        
        filtered_trades = []
        for trade_data in trades:
            try:
                trade_timestamp = trade_data.get('timestamp', '')
                trade_date = trade_timestamp[:10] if len(trade_timestamp) >= 10 else ''
                
                if trade_date >= cutoff_date_str:
                    filtered_trades.append(Trade.from_dict(trade_data))
            except Exception as e:
                continue
    
        return sorted(filtered_trades, key=lambda x: x.timestamp, reverse=True)

    def get_bot_state(self) -> Dict:
        return self._load_json(self.bot_state_file)
    
    def update_bot_state(self, **kwargs):
        if not kwargs:
            return
        
        state = self._load_json(self.bot_state_file)
        state.update(kwargs)
        state['last_update'] = datetime.now().isoformat()
        self._save_json(self.bot_state_file, state)
    
    def get_recent_performance(self, symbol: str, days: int = 7) -> List[Trade]:
        trades = self._load_json(self.trades_file)
        since_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        filtered_trades = []
        for trade_data in trades:
            if (trade_data.get('symbol') == symbol and 
                trade_data.get('timestamp', '') >= since_date and 
                trade_data.get('status') != 'open'):
                filtered_trades.append(Trade.from_dict(trade_data))
        
        return sorted(filtered_trades, key=lambda x: x.timestamp, reverse=True)[:10]

class TechnicalIndicators:
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

class AggressiveStrategy:
    def __init__(self, memory: FileMemory):
        self.memory = memory
        self.last_signal_time = {}
        self.signal_cooldown = 120  # 2 minutes - more aggressive
    
    def analyze_market(self, data: pd.DataFrame, symbol: str, asset_class: str = "stock") -> Dict:
        if len(data) < 30:  # Reduced from 50 for more aggressive trading
            return {"insufficient_data": True}
        
        current_price = data['Close'].iloc[-1]
        
        # More aggressive RSI thresholds
        rsi = TechnicalIndicators.rsi(data['Close'], 10)  # Faster RSI
        oversold_threshold = 35 if asset_class == "crypto" else 35
        overbought_threshold = 65 if asset_class == "crypto" else 65
            
        analysis = {
            'symbol': symbol, 'asset_class': asset_class, 'price': current_price,
            'rsi': {
                'value': rsi.iloc[-1] if not rsi.empty else 50,
                'oversold': rsi.iloc[-1] < oversold_threshold if not rsi.empty else False,
                'overbought': rsi.iloc[-1] > overbought_threshold if not rsi.empty else False
            }
        }
        
        # Faster MACD
        macd_line, signal_line, histogram = TechnicalIndicators.macd(data['Close'], 8, 21, 5)
        analysis['macd'] = {
            'line': macd_line.iloc[-1] if not macd_line.empty else 0,
            'signal': signal_line.iloc[-1] if not signal_line.empty else 0,
            'histogram': histogram.iloc[-1] if not histogram.empty else 0,
            'bullish_cross': False, 'bearish_cross': False
        }
        
        if len(macd_line) >= 2 and len(signal_line) >= 2:
            prev_macd, curr_macd = macd_line.iloc[-2], macd_line.iloc[-1]
            prev_signal, curr_signal = signal_line.iloc[-2], signal_line.iloc[-1]
            
            analysis['macd']['bullish_cross'] = (prev_macd <= prev_signal) and (curr_macd > curr_signal)
            analysis['macd']['bearish_cross'] = (prev_macd >= prev_signal) and (curr_macd < curr_signal)
        
        # Faster moving averages
        sma_10 = TechnicalIndicators.sma(data['Close'], 10)
        sma_20 = TechnicalIndicators.sma(data['Close'], 20)
        
        analysis['sma'] = {
            'sma_10': sma_10.iloc[-1] if not sma_10.empty else current_price,
            'sma_20': sma_20.iloc[-1] if not sma_20.empty else current_price,
            'above_sma_10': current_price > sma_10.iloc[-1] if not sma_10.empty else False,
            'above_sma_20': current_price > sma_20.iloc[-1] if not sma_20.empty else False,
            'sma_cross_up': False, 'sma_cross_down': False
        }
        
        if len(sma_10) >= 2 and len(sma_20) >= 2:
            analysis['sma']['sma_cross_up'] = (sma_10.iloc[-2] <= sma_20.iloc[-2]) and (sma_10.iloc[-1] > sma_20.iloc[-1])
            analysis['sma']['sma_cross_down'] = (sma_10.iloc[-2] >= sma_20.iloc[-2]) and (sma_10.iloc[-1] < sma_20.iloc[-1])
        
        # Volume analysis
        if len(data) >= 10:
            avg_volume = data['Volume'].rolling(10).mean().iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            volume_multiplier = 1.3 if asset_class == "crypto" else 1.2  # Lower threshold
            analysis['volume'] = {
                'current': current_volume, 'average': avg_volume,
                'high_volume': current_volume > (avg_volume * volume_multiplier)
            }
        else:
            analysis['volume'] = {'high_volume': False}
        
        return analysis
    
    def should_buy(self, data: pd.DataFrame, symbol: str, asset_class: str = "stock") -> Tuple[bool, str, float]:
        analysis = self.analyze_market(data, symbol, asset_class)
        
        if analysis.get('insufficient_data'):
            return False, "Insufficient data", 0.0
        
        # Less restrictive recent performance check
        recent_trades = self.memory.get_recent_performance(symbol, days=1)
        recent_losses = sum(1 for trade in recent_trades if trade.pnl < 0)
        if recent_losses >= 5:  # Allow more losses before stopping
            return False, f"Too many recent losses ({recent_losses})", 0.0
        
        # Signal cooldown
        current_time = time.time()
        if symbol in self.last_signal_time:
            if current_time - self.last_signal_time[symbol] < self.signal_cooldown:
                return False, "Signal cooldown active", 0.0
        
        buy_signals = []
        buy_score = 0
        
        # More aggressive RSI signals
        if analysis['rsi']['oversold']:
            buy_signals.append("RSI oversold")
            buy_score += 2
        elif analysis['rsi']['value'] < 45:
            buy_signals.append("RSI moderate")
            buy_score += 1
        
        # MACD signals
        if analysis['macd']['bullish_cross']:
            buy_signals.append("MACD bullish cross")
            buy_score += 3
        elif analysis['macd']['histogram'] > 0:
            buy_signals.append("MACD positive")
            buy_score += 1
        
        # Moving average signals
        if analysis['sma']['above_sma_10']:
            buy_signals.append("Above SMA 10")
            buy_score += 1
        
        if analysis['sma']['sma_cross_up']:
            buy_signals.append("SMA cross up")
            buy_score += 2
        
        # Volume confirmation
        if analysis['volume']['high_volume']:
            buy_signals.append("High volume")
            buy_score += 1
        
        # Calculate tighter stop loss for smaller positions
        current_price = analysis['price']
        stop_loss = current_price * (0.97 if asset_class == "crypto" else 0.98)  # Tighter stops
        
        required_score = 2  # Lower threshold for more trades
        
        if buy_score >= required_score:
            self.last_signal_time[symbol] = current_time
            return True, f"BUY: {', '.join(buy_signals)} (Score: {buy_score})", stop_loss
        
        return False, f"Weak buy: {', '.join(buy_signals)} (Score: {buy_score})", 0.0
    
    def should_sell(self, data: pd.DataFrame, symbol: str, entry_price: float = None, asset_class: str = "stock") -> Tuple[bool, str]:
        analysis = self.analyze_market(data, symbol, asset_class)
        
        if analysis.get('insufficient_data'):
            return False, "Insufficient data"
        
        # Signal cooldown
        current_time = time.time()
        if symbol in self.last_signal_time:
            if current_time - self.last_signal_time[symbol] < self.signal_cooldown:
                return False, "Signal cooldown active"
        
        sell_signals = []
        sell_score = 0
        
        # RSI signals
        if analysis['rsi']['overbought']:
            sell_signals.append("RSI overbought")
            sell_score += 2
        elif analysis['rsi']['value'] > 55:
            sell_signals.append("RSI high")
            sell_score += 1
        
        # MACD signals
        if analysis['macd']['bearish_cross']:
            sell_signals.append("MACD bearish cross")
            sell_score += 3
        elif analysis['macd']['histogram'] < 0:
            sell_signals.append("MACD negative")
            sell_score += 1
        
        # Moving average signals
        if not analysis['sma']['above_sma_10']:
            sell_signals.append("Below SMA 10")
            sell_score += 1
        
        if analysis['sma']['sma_cross_down']:
            sell_signals.append("SMA cross down")
            sell_score += 2
        
        # Quick profit taking for small positions
        if entry_price:
            current_price = analysis['price']
            profit_pct = (current_price - entry_price) / entry_price * 100
            
            if asset_class == "crypto":
                if profit_pct > 4:  # Take profits faster
                    sell_signals.append(f"Take profit ({profit_pct:.1f}%)")
                    sell_score += 2
                elif profit_pct > 2:
                    sell_signals.append(f"Small profit ({profit_pct:.1f}%)")
                    sell_score += 1
            else:
                if profit_pct > 3:
                    sell_signals.append(f"Take profit ({profit_pct:.1f}%)")
                    sell_score += 2
                elif profit_pct > 1.5:
                    sell_signals.append(f"Small profit ({profit_pct:.1f}%)")
                    sell_score += 1
        
        required_score = 2  # Lower threshold
        
        if sell_score >= required_score:
            self.last_signal_time[symbol] = current_time
            return True, f"SELL: {', '.join(sell_signals)} (Score: {sell_score})"
        
        return False, f"Weak sell: {', '.join(sell_signals)} (Score: {sell_score})"

class AlpacaBroker:
    def __init__(self, api_key: str, secret_key: str, paper_trading: bool = True):
        self.api_key = api_key.strip()
        self.secret_key = secret_key.strip()
        self.paper_trading = paper_trading
        
        if paper_trading:
            self.base_url = "https://paper-api.alpaca.markets"
            logger.info("Using PAPER TRADING mode")
        else:
            self.base_url = "https://api.alpaca.markets"
            logger.warning("Using LIVE TRADING mode")
        
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        self._test_connection()
      
    def get_actual_position_quantity(self, symbol: str) -> float:
            """Get the actual position quantity from Alpaca, handling crypto precision"""
            try:
                positions = self.get_positions()
                for pos in positions:
                    if pos['symbol'] == symbol:
                        qty = float(pos['qty'])
                        # For crypto, ensure we have enough precision
                        if symbol.endswith('/USD') and qty > 0:
                            # Round down to avoid precision issues
                            if qty < 1:
                                return float(Decimal(str(qty)).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN))
                            else:
                                return float(Decimal(str(qty)).quantize(Decimal('0.00001'), rounding=ROUND_DOWN))
                        return qty
                return 0.0
            except Exception as e:
                logger.error(f"Error getting actual position for {symbol}: {e}")
                return 0.0
        
    def _test_connection(self):
        try:
            account = self.get_account()
            if account and 'account_number' in account:
                logger.info(f"Connected to Alpaca successfully")
                logger.info(f"Buying power: ${float(account.get('buying_power', 0)):,.2f}")
                return True
            else:
                logger.error("Account data missing or invalid")
                return False
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_account(self) -> Dict:
        try:
            response = requests.get(f"{self.base_url}/v2/account", headers=self.headers, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Account request failed: {response.status_code}")
                response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    def get_current_price(self, symbol: str) -> float:
        try:
            yf_symbol = symbol.replace('/USD', '-USD') if symbol.endswith('/USD') else symbol
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                return float(hist.iloc[-1]['Close'])
        except Exception as e:
            logger.debug(f"Price lookup failed for {symbol}: {e}")
        return 0.0
    
    def get_bars(self, symbol: str, timeframe: str = "5Min", limit: int = 100) -> pd.DataFrame:
        try:
            yf_symbol = symbol.replace('/USD', '-USD') if symbol.endswith('/USD') else symbol
            ticker = yf.Ticker(yf_symbol)
            period = "10d" if symbol.endswith('/USD') else "30d"
            data = ticker.history(period=period, interval="5m")  # 5 minute bars for more aggressive
            if not data.empty:
                return data.tail(limit)
        except Exception as e:
            logger.debug(f"Bars failed for {symbol}: {e}")
        return pd.DataFrame()
    def get_order_details(self, order_id: str) -> Dict:
        """Get details of a specific order"""
        try:
            response = requests.get(f"{self.base_url}/v2/orders/{order_id}", headers=self.headers, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Order details request failed: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Error getting order details: {e}")
            return {}
    
    def wait_for_order_fill(self, order_id: str, timeout: int = 30) -> Dict:
        """Wait for order to fill and return final status"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            order_details = self.get_order_details(order_id)
            if order_details:
                status = order_details.get('status', '')
                if status in ['filled', 'partially_filled', 'cancelled', 'rejected']:
                    return order_details
            time.sleep(1)
        
        # Return last known status
        return self.get_order_details(order_id)
    

    def place_order(self, order: Order) -> bool:
            """Place order with improved crypto handling"""
            try:
                time_in_force = "gtc" if order.asset_class == "crypto" else "day"
                
                if order.asset_class == "crypto":
                    # For crypto, always use notional orders for more precise control
                    current_price = self.get_current_price(order.symbol)
                    if current_price <= 0:
                        logger.error(f"Cannot get price for {order.symbol}")
                        return False
                    
                    # Calculate notional value
                    if order.notional:
                        notional = order.notional
                    else:
                        notional = order.quantity * current_price
                    
                    # Ensure minimum notional ($1 for most cryptos)
                    if notional < 1.0:
                        logger.error(f"Notional value too small: ${notional:.2f}")
                        return False
                    
                    order_data = {
                        "symbol": order.symbol,
                        "notional": str(round(notional, 2)),
                        "side": order.order_type.value,
                        "type": "market",
                        "time_in_force": time_in_force
                    }
                else:
                    # For stocks, use quantity-based orders
                    order_data = {
                        "symbol": order.symbol,
                        "qty": str(int(order.quantity)),
                        "side": order.order_type.value,
                        "type": "market",
                        "time_in_force": time_in_force
                    }
                
                logger.info(f"Placing {order.order_type.value.upper()} order for {order.symbol}: {order_data}")
                
                response = requests.post(f"{self.base_url}/v2/orders", headers=self.headers, json=order_data, timeout=10)
                
                if response.status_code in [200, 201]:
                    result = response.json()
                    order.order_id = result['id']
                    order.status = OrderStatus.PENDING
                    logger.info(f"Order placed successfully: {order.order_id}")
                    
                    # For crypto, wait for fill to get actual quantity
                    if order.asset_class == "crypto":
                        fill_details = self.wait_for_order_fill(order.order_id, timeout=15)
                        if fill_details and fill_details.get('status') == 'filled':
                            filled_qty = float(fill_details.get('filled_qty', 0))
                            if filled_qty > 0:
                                order.quantity = filled_qty
                                logger.info(f"Crypto order filled: {filled_qty} {order.symbol}")
                            else:
                                logger.warning(f"Crypto order filled but no quantity returned")
                    
                    return True
                else:
                    logger.error(f"Order failed: {response.status_code} - {response.text}")
                    return False
                
            except Exception as e:
                logger.error(f"Error placing order: {e}")
                return False
        
    def get_positions(self) -> List[Dict]:
        try:
            response = requests.get(f"{self.base_url}/v2/positions", headers=self.headers, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Positions request failed: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
   

class AggressiveTradingBot:
    def __init__(self, broker: AlpacaBroker, initial_capital: float = 1000.0, data_dir: str = "trading_data"):
        self.broker = broker
        self.initial_capital = initial_capital
        self.market_hours = MarketHours()
        self.memory = FileMemory(data_dir)
        
        # HIGHER DAILY LIMITS - MORE AGGRESSIVE
        self.daily_crypto_limit = 2500.0   # $2500 per day for crypto (up from $1000)
        self.daily_stock_limit = 5000.0    # $5000 per day for stocks (up from $2000)
        
        # More diverse and aggressive symbol selection
        self.stock_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "AMD", 
                             "CRM", "UBER", "SPY", "QQQ", "IWM", "NFLX", "BABA", "PYPL", "SQ", "ROKU"]
        self.crypto_symbols = ["BTC/USD", "ETH/USD", "LTC/USD", "BCH/USD", "LINK/USD", "AAVE/USD"]
        self.symbols = self.stock_symbols + self.crypto_symbols
        
        self.strategy = AggressiveStrategy(self.memory)
        
        # AGGRESSIVE PARAMETERS
        self.check_interval = 60        # 1 minute checks
        self.max_positions = 12         # More positions
        self.min_hold_time = 60         # 1 minute minimum hold
        self.crypto_risk_per_trade = 0.01   # 1% risk per trade - SMALLER AMOUNTS
        self.stock_risk_per_trade = 0.01    # 1% risk per trade - SMALLER AMOUNTS
        
        self.active_positions = {}
        self._restore_state()
        
        bot_state = self.memory.get_bot_state()
        self.total_pnl = bot_state.get('total_pnl', 0.0)
        self.total_trades = bot_state.get('total_trades', 0)
        self.winning_trades = bot_state.get('winning_trades', 0)
        
        logger.info(f"🚀 AGGRESSIVE TRADING BOT INITIALIZED 🚀")
        logger.info(f"Symbols: {len(self.stock_symbols)} stocks + {len(self.crypto_symbols)} cryptos")
        logger.info(f"Daily limits: Crypto ${self.daily_crypto_limit:,}, Stocks ${self.daily_stock_limit:,}")
        logger.info(f"Risk per trade: {self.crypto_risk_per_trade*100:.0f}% (SMALL POSITIONS)")
    
    def _get_asset_class(self, symbol: str) -> str:
        return "crypto" if symbol in self.crypto_symbols else "stock"
    
    def _restore_state(self):
            """Improved state restoration with better crypto position handling"""
            memory_positions = self.memory.get_open_trades()
            alpaca_positions = self.broker.get_positions()
            alpaca_symbols = {pos['symbol']: pos for pos in alpaca_positions if float(pos['qty']) != 0}
            
            self.active_positions = {}
            ghost_positions = []
            restored_positions = []
            
            # Clean restore - only load positions that exist in both places
            for symbol, (trade, trade_id) in memory_positions.items():
                if symbol in alpaca_symbols:
                    alpaca_pos = alpaca_symbols[symbol]
                    actual_qty = float(alpaca_pos['qty'])
                    
                    # For crypto, update the trade quantity to match actual position
                    if self._get_asset_class(symbol) == "crypto":
                        if abs(actual_qty - trade.quantity) > 0.0001:  # Allow for small differences
                            logger.info(f"Updating {symbol} quantity: {trade.quantity:.6f} -> {actual_qty:.6f}")
                            trade.quantity = actual_qty
                            trade.actual_quantity = actual_qty
                            self.memory.update_trade(trade, trade_id)
                    
                    self.active_positions[symbol] = {
                        'trade': trade, 'trade_id': trade_id, 'quantity': actual_qty,
                        'entry_price': trade.entry_price, 'stop_loss': trade.stop_loss,
                        'timestamp': datetime.fromisoformat(trade.timestamp),
                        'asset_class': trade.asset_class
                    }
                    restored_positions.append(symbol)
                else:
                    ghost_positions.append((symbol, trade, trade_id))
            
            # Auto-close ghost positions (positions in memory but not in Alpaca)
            for symbol, trade, trade_id in ghost_positions:
                trade.exit_price = trade.entry_price
                trade.exit_timestamp = datetime.now().isoformat()
                trade.status = "closed"
                trade.pnl = 0.0
                trade.pnl_pct = 0.0
                trade.exit_reason = "Position not found - auto-closed"
                self.memory.update_trade(trade, trade_id)
            
            logger.info(f"Restored {len(restored_positions)} positions, cleaned {len(ghost_positions)} ghosts")
            if restored_positions:
                logger.info(f"Active positions: {', '.join(restored_positions)}")
    
    def get_daily_spending(self, asset_class: str) -> float:
        today = datetime.now().strftime('%Y-%m-%d')
        all_trades = self.memory.get_trade_history(days=1)
        
        daily_spending = 0.0
        for trade in all_trades:
            try:
                if hasattr(trade, 'timestamp') and trade.timestamp:
                    trade_date = trade.timestamp.split('T')[0] if 'T' in trade.timestamp else trade.timestamp[:10]
                    
                    if (trade_date == today and 
                        trade.asset_class == asset_class and 
                        trade.status in ['open', 'closed']):
                        
                        trade_value = trade.entry_price * trade.quantity
                        daily_spending += trade_value
            except Exception:
                continue
        
        return daily_spending
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float, asset_class: str) -> float:
        account = self.broker.get_account()
        buying_power = float(account.get('buying_power', self.initial_capital))
        
        # Small risk per trade
        risk_per_trade = self.crypto_risk_per_trade if asset_class == "crypto" else self.stock_risk_per_trade
        risk_amount = buying_power * risk_per_trade
        
        stop_distance = entry_price - stop_loss
        if stop_distance <= 0:
            return 0
        
        position_size = risk_amount / stop_distance
        
        # Cap by available budget
        daily_spending = self.get_daily_spending(asset_class)
        daily_limit = self.daily_crypto_limit if asset_class == "crypto" else self.daily_stock_limit
        remaining_budget = daily_limit - daily_spending
        
        if remaining_budget < 20:  # Minimum $20 to trade
            return 0
        
        # Ensure position fits in remaining budget
        position_value = position_size * entry_price
        if position_value > remaining_budget:
            position_size = remaining_budget / entry_price
        
        # Final size constraints
        if asset_class == "crypto":
            min_shares = 15.0 / entry_price  # $15 minimum
            position_size = max(position_size, min_shares)
            position_size = min(position_size, 200.0 / entry_price)  # $200 max per crypto trade
        else:
            position_size = max(1, int(position_size))
            position_size = min(position_size, int(300.0 / entry_price))  # $300 max per stock trade
        
        return position_size
    
    def execute_buy(self, symbol: str, current_price: float, stop_loss: float, reason: str):
        if len(self.active_positions) >= self.max_positions:
            return False
        
        asset_class = self._get_asset_class(symbol)
        
        if asset_class == "stock" and not self.market_hours.is_stock_market_open():
            return False
        
        # Check daily limits first
        daily_spending = self.get_daily_spending(asset_class)
        daily_limit = self.daily_crypto_limit if asset_class == "crypto" else self.daily_stock_limit
        
        if daily_spending >= daily_limit:
            return False
        
        position_size = self.calculate_position_size(symbol, current_price, stop_loss, asset_class)
        
        if position_size <= 0:
            return False
        
        order = Order(symbol=symbol, order_type=OrderType.BUY, quantity=position_size, asset_class=asset_class)
        
        logger.info(f"BUY {symbol}: {position_size:.4f} @ ${current_price:.2f} | {reason}")
        
        if self.broker.place_order(order):
            # Use actual filled quantity for crypto
            actual_quantity = order.quantity
            
            trade = Trade(
                symbol=symbol, entry_price=current_price, quantity=actual_quantity,
                timestamp=datetime.now().isoformat(), stop_loss=stop_loss,
                asset_class=asset_class, alpaca_order_id=order.order_id,
                status="open", strategy_signals=reason, actual_quantity=actual_quantity
            )
            
            trade_id = self.memory.save_trade(trade)
            
            self.active_positions[symbol] = {
                'trade': trade, 'trade_id': trade_id, 'quantity': actual_quantity,
                'entry_price': current_price, 'stop_loss': stop_loss,
                'timestamp': datetime.now(), 'asset_class': asset_class
            }
            
            return True
        
        return False
    
    def execute_sell(self, symbol: str, reason: str):
        """Improved crypto selling with better position handling"""
        if symbol not in self.active_positions:
            return False
        
        position = self.active_positions[symbol]
        asset_class = position['asset_class']
        
        if asset_class == "stock" and not self.market_hours.is_stock_market_open():
            if "stop loss" not in reason.lower():
                return False
        
        current_price = self.broker.get_current_price(symbol)
        if current_price <= 0:
            logger.error(f"Cannot get current price for {symbol}")
            return False
        
        # Check minimum hold time
        hold_time = (datetime.now() - position['timestamp']).seconds
        if hold_time < self.min_hold_time:
            return False
        
        # Get the actual position quantity from Alpaca
        actual_position = self.broker.get_actual_position_quantity(symbol)
        
        if actual_position <= 0:
            logger.warning(f"No actual {symbol} position found in Alpaca, cleaning up memory")
            # Clean up the position from memory
            trade = position['trade']
            trade.exit_price = current_price
            trade.exit_timestamp = datetime.now().isoformat()
            trade.status = "closed"
            trade.pnl = 0.0
            trade.pnl_pct = 0.0
            trade.exit_reason = "Position not found - cleaned up"
            self.memory.update_trade(trade, position['trade_id'])
            del self.active_positions[symbol]
            return False
        
        sell_quantity = actual_position
        
        # For crypto, create a notional sell order if the quantity is very small
        if asset_class == "crypto" and sell_quantity * current_price < 1.0:
            notional_value = sell_quantity * current_price
            order = Order(
                symbol=symbol, 
                order_type=OrderType.SELL, 
                quantity=sell_quantity,
                notional=notional_value,
                asset_class=asset_class
            )
        else:
            order = Order(
                symbol=symbol, 
                order_type=OrderType.SELL, 
                quantity=sell_quantity, 
                asset_class=asset_class
            )
        
        # Calculate P&L using position entry price and actual sell quantity
        pnl = (current_price - position['entry_price']) * sell_quantity
        pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
        
        logger.info(f"SELL {symbol}: {sell_quantity:.6f} @ ${current_price:.2f} | "
                   f"P&L: ${pnl:.2f} ({pnl_pct:.1f}%) | {reason}")
        
        if self.broker.place_order(order):
            trade = position['trade']
            trade.exit_price = current_price
            trade.exit_timestamp = datetime.now().isoformat()
            trade.status = "closed"
            trade.pnl = pnl
            trade.pnl_pct = pnl_pct
            trade.exit_reason = reason
            trade.quantity = sell_quantity  # Update with actual sold quantity
            trade.actual_quantity = sell_quantity
            
            self.memory.update_trade(trade, position['trade_id'])
            
            self.total_pnl += pnl
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1
            
            self.memory.update_bot_state(
                total_pnl=self.total_pnl,
                total_trades=self.total_trades,
                winning_trades=self.winning_trades
            )
            
            del self.active_positions[symbol]
            return True
        else:
            logger.error(f"Failed to place sell order for {symbol}")
            return False

    
    def check_stop_losses(self):
        for symbol, position in list(self.active_positions.items()):
            try:
                current_price = self.broker.get_current_price(symbol)
                if current_price <= 0:
                    continue
                
                # Tight stop loss
                if current_price <= position['stop_loss']:
                    logger.warning(f"STOP LOSS: {symbol} @ ${current_price:.2f}")
                    self.execute_sell(symbol, "Stop loss")
                    continue
                
                # Quick trailing stop for profitable positions
                entry_price = position['entry_price']
                profit_pct = (current_price - entry_price) / entry_price * 100
                
                if profit_pct > 2:  # Trail after 2% profit
                    asset_class = position['asset_class']
                    trail_pct = 0.04 if asset_class == "crypto" else 0.03  # Tighter trailing
                    new_stop = current_price * (1 - trail_pct)
                    
                    if new_stop > position['stop_loss']:
                        position['stop_loss'] = new_stop
                        trade = position['trade']
                        trade.stop_loss = new_stop
                        self.memory.update_trade(trade, position['trade_id'])
                
            except Exception as e:
                logger.error(f"Error checking stop loss for {symbol}: {e}")
    
    def scan_markets(self):
        stock_market_open = self.market_hours.is_stock_market_open()
        
        for symbol in self.symbols:
            try:
                asset_class = self._get_asset_class(symbol)
                
                if asset_class == "stock" and not stock_market_open:
                    continue
                
                data = self.broker.get_bars(symbol, "5Min", 60)  # Faster bars
                has_position = symbol in self.active_positions
                asset_class = self._get_asset_class(symbol)
                min_bars = 3 if (has_position and asset_class == "crypto") else (5 if has_position else 10)
                if data.empty or len(data) < min_bars:
                    continue
                current_price = self.broker.get_current_price(symbol)
                if current_price <= 0:
                    continue
                
                # Check for buy signals
                if symbol not in self.active_positions:
                    should_buy, buy_reason, stop_loss = self.strategy.should_buy(data, symbol, asset_class)
                    if should_buy:
                        self.execute_buy(symbol, current_price, stop_loss, buy_reason)
                
                # Check for sell signals
                elif symbol in self.active_positions:
                    position = self.active_positions[symbol]
                    should_sell, sell_reason = self.strategy.should_sell(data, symbol, position['entry_price'], asset_class)
                    if should_sell:
                        self.execute_sell(symbol, sell_reason)
            
            except Exception as e:
                logger.debug(f"Error scanning {symbol}: {e}")
    
    def print_status(self):
        account = self.broker.get_account()
        portfolio_value = float(account.get('portfolio_value', self.initial_capital))
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        stock_positions = {k: v for k, v in self.active_positions.items() if v['asset_class'] == 'stock'}
        crypto_positions = {k: v for k, v in self.active_positions.items() if v['asset_class'] == 'crypto'}
        
        logger.info("=" * 60)
        logger.info(f"🚀 AGGRESSIVE BOT STATUS - {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"Portfolio: ${portfolio_value:,.2f} | P&L: ${self.total_pnl:.2f} | Win Rate: {win_rate:.1%} ({self.winning_trades}/{self.total_trades})")
        logger.info(f"Positions: {len(self.active_positions)}/{self.max_positions} | Stocks: {len(stock_positions)} | Crypto: {len(crypto_positions)}")
        
        if self.active_positions:
            logger.info("Active positions:")
            
            if stock_positions:
                logger.info("  STOCKS:")
                for symbol, position in stock_positions.items():
                    current_price = self.broker.get_current_price(symbol)
                    unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
                    hold_time = datetime.now() - position['timestamp']
                    
                    logger.info(f"    {symbol}: {position['quantity']:.0f} @ ${position['entry_price']:.2f} | "
                              f"Current: ${current_price:.2f} | P&L: ${unrealized_pnl:.2f} ({pnl_pct:.1f}%) | "
                              f"Stop: ${position['stop_loss']:.2f} | Hold: {str(hold_time).split('.')[0]}")
            
            if crypto_positions:
                logger.info("  CRYPTO:")
                for symbol, position in crypto_positions.items():
                    current_price = self.broker.get_current_price(symbol)
                    actual_qty = self.broker.get_actual_position_quantity(symbol)  # Use actual quantity
                    unrealized_pnl = (current_price - position['entry_price']) * actual_qty
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
                    hold_time = datetime.now() - position['timestamp']
                    
                    logger.info(f"    {symbol}: {actual_qty:.6f} @ ${position['entry_price']:.2f} | "
                              f"Current: ${current_price:.2f} | P&L: ${unrealized_pnl:.2f} ({pnl_pct:.1f}%) | "
                              f"Stop: ${position['stop_loss']:.2f} | Hold: {str(hold_time).split('.')[0]}")
        
        logger.info("=" * 60)
    
    def run(self):
        logger.info("🚀 STARTING AGGRESSIVE HIGH-FREQUENCY TRADING BOT 🚀")
        logger.info(f"SMALL POSITIONS | HIGH LIMITS | 1% RISK | {self.max_positions} MAX POSITIONS | {self.check_interval}s INTERVALS")
        
        iteration = 0
        last_status_print = time.time()
        
        try:
            while True:
                iteration += 1
                start_time = time.time()
                
                # Print status every 5 minutes
                if time.time() - last_status_print >= 300:
                    self.print_status()
                    last_status_print = time.time()
                
                self.check_stop_losses()
                self.scan_markets()
                
                elapsed = time.time() - start_time
                sleep_time = max(0, self.check_interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("🛑 Trading stopped by user")
        except Exception as e:
            logger.error(f"💥 Fatal error: {e}")
        finally:
            self.print_status()
            logger.info("🏁 Bot shutdown complete")

def main():
    API_KEY = os.getenv("ALPACA_API_KEY")
    SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    PAPER_TRADING = os.getenv("ALPACA_PAPER_TRADING", "true").lower() == "true"
    DATA_DIR = os.getenv("TRADING_DATA_DIR", "trading_data")
    
    if not API_KEY or not SECRET_KEY:
        logger.error("Missing API keys! Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        return
    
    try:
        logger.info("🚀 Initializing AGGRESSIVE trading bot...")
        broker = AlpacaBroker(API_KEY, SECRET_KEY, paper_trading=PAPER_TRADING)
        
        account = broker.get_account()
        if not account or 'account_number' not in account:
            logger.error("❌ Failed to connect to Alpaca API")
            return
        
        bot = AggressiveTradingBot(broker, data_dir=DATA_DIR)
        bot.run()
        
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"💥 Fatal error in main: {e}")

if __name__ == "__main__":
    main()
