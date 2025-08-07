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
    
    def get_market_status_message(self) -> str:
        cst_now = datetime.now(self.cst)
        stock_open = self.is_stock_market_open()
        
        status = f"Market Status (CST: {cst_now.strftime('%I:%M %p, %A %m/%d/%Y')}):\n"
        status += f"  📈 Stocks: {'OPEN' if stock_open else 'CLOSED'}\n"
        status += f"  ₿ Crypto: OPEN"
        return status

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
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Trade':
        # Remove trade_id if it exists since it's not a field in the Trade class
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
        
        # Get cutoff datetime
        cutoff_datetime = datetime.now() - timedelta(days=days)
        cutoff_date_str = cutoff_datetime.strftime('%Y-%m-%d')
        
        filtered_trades = []
        for trade_data in trades:
            try:
                trade_timestamp = trade_data.get('timestamp', '')
                # Extract date part from ISO timestamp (YYYY-MM-DD)
                trade_date = trade_timestamp[:10] if len(trade_timestamp) >= 10 else ''
                
                if trade_date >= cutoff_date_str:
                    filtered_trades.append(Trade.from_dict(trade_data))
            except Exception as e:
                # Skip malformed trade data
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
    
    def save_daily_performance(self, date: str, daily_pnl: float, daily_trades: int, 
                             win_rate: float, portfolio_value: float, drawdown: float):
        performance = self._load_json(self.performance_file)
        performance[date] = {
            'daily_pnl': daily_pnl, 'daily_trades': daily_trades, 'win_rate': win_rate,
            'portfolio_value': portfolio_value, 'drawdown': drawdown,
            'timestamp': datetime.now().isoformat()
        }
        self._save_json(self.performance_file, performance)
    
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

class TradingStrategy:
    def __init__(self, memory: FileMemory):
        self.memory = memory
        self.last_signal_time = {}
        self.signal_cooldown = 900  # 15 minutes
    
    def analyze_market(self, data: pd.DataFrame, symbol: str, asset_class: str = "stock") -> Dict:
        if len(data) < 50:
            return {"insufficient_data": True}
        
        current_price = data['Close'].iloc[-1]
        
        # RSI with asset class thresholds
        rsi = TechnicalIndicators.rsi(data['Close'])
        oversold_threshold = 25 if asset_class == "crypto" else 30
        overbought_threshold = 75 if asset_class == "crypto" else 70
            
        analysis = {
            'symbol': symbol, 'asset_class': asset_class, 'price': current_price,
            'rsi': {
                'value': rsi.iloc[-1] if not rsi.empty else 50,
                'oversold': rsi.iloc[-1] < oversold_threshold if not rsi.empty else False,
                'overbought': rsi.iloc[-1] > overbought_threshold if not rsi.empty else False
            }
        }
        
        # MACD
        macd_line, signal_line, histogram = TechnicalIndicators.macd(data['Close'])
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
        
        # Moving Averages
        sma_20 = TechnicalIndicators.sma(data['Close'], 20)
        sma_50 = TechnicalIndicators.sma(data['Close'], 50)
        
        analysis['sma'] = {
            'sma_20': sma_20.iloc[-1] if not sma_20.empty else current_price,
            'sma_50': sma_50.iloc[-1] if not sma_50.empty else current_price,
            'above_sma_20': current_price > sma_20.iloc[-1] if not sma_20.empty else False,
            'above_sma_50': current_price > sma_50.iloc[-1] if not sma_50.empty else False,
            'sma_cross_up': False, 'sma_cross_down': False
        }
        
        if len(sma_20) >= 2 and len(sma_50) >= 2:
            analysis['sma']['sma_cross_up'] = (sma_20.iloc[-2] <= sma_50.iloc[-2]) and (sma_20.iloc[-1] > sma_50.iloc[-1])
            analysis['sma']['sma_cross_down'] = (sma_20.iloc[-2] >= sma_50.iloc[-2]) and (sma_20.iloc[-1] < sma_50.iloc[-1])
        
        # Volume analysis
        if len(data) >= 20:
            avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            volume_multiplier = 2.0 if asset_class == "crypto" else 1.5
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
        
        # Check recent performance
        recent_trades = self.memory.get_recent_performance(symbol, days=1)
        recent_losses = sum(1 for trade in recent_trades if trade.pnl < 0)
        if recent_losses >= 3:
            return False, f"Too many recent losses ({recent_losses})", 0.0
        
        # Signal cooldown
        current_time = time.time()
        if symbol in self.last_signal_time:
            if current_time - self.last_signal_time[symbol] < self.signal_cooldown:
                return False, "Signal cooldown active", 0.0
        
        buy_signals = []
        buy_score = 0
        
        # RSI signals
        if analysis['rsi']['oversold']:
            buy_signals.append("RSI oversold")
            buy_score += 3
        elif analysis['rsi']['value'] < 40:
            buy_signals.append("RSI trending down")
            buy_score += 1
        
        # MACD signals
        if analysis['macd']['bullish_cross']:
            buy_signals.append("MACD bullish crossover")
            buy_score += 4
        elif analysis['macd']['histogram'] > 0:
            buy_signals.append("MACD positive momentum")
            buy_score += 1
        
        # Moving average signals
        if analysis['sma']['above_sma_20']:
            buy_signals.append("Above SMA 20")
            buy_score += 2
        
        if analysis['sma']['sma_cross_up']:
            buy_signals.append("SMA golden cross")
            buy_score += 3
        
        # Volume confirmation
        if analysis['volume']['high_volume']:
            buy_signals.append("High volume")
            weight = 2 if asset_class == "crypto" else 1
            buy_score += weight
        
        # Calculate stop loss
        current_price = analysis['price']
        stop_loss = current_price * (0.95 if asset_class == "crypto" else 0.96)
        
        required_score = 2 if asset_class == "crypto" else 3
        
        if buy_score >= required_score:
            self.last_signal_time[symbol] = current_time
            return True, f"BUY signals: {', '.join(buy_signals)} (Score: {buy_score})", stop_loss
        
        return False, f"Weak buy signals: {', '.join(buy_signals)} (Score: {buy_score})", 0.0
    
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
            sell_score += 3
        elif analysis['rsi']['value'] > 60:
            sell_signals.append("RSI trending up")
            sell_score += 1
        
        # MACD signals
        if analysis['macd']['bearish_cross']:
            sell_signals.append("MACD bearish crossover")
            sell_score += 4
        elif analysis['macd']['histogram'] < 0:
            sell_signals.append("MACD negative momentum")
            sell_score += 1
        
        # Moving average signals
        if not analysis['sma']['above_sma_20']:
            sell_signals.append("Below SMA 20")
            sell_score += 2
        
        if analysis['sma']['sma_cross_down']:
            sell_signals.append("SMA death cross")
            sell_score += 3
        
        # Profit taking
        if entry_price:
            current_price = analysis['price']
            profit_pct = (current_price - entry_price) / entry_price * 100
            
            if asset_class == "crypto":
                if profit_pct > 8:
                    sell_signals.append(f"Take profit ({profit_pct:.1f}%)")
                    sell_score += 3
                elif profit_pct > 4:
                    sell_signals.append(f"Partial profit ({profit_pct:.1f}%)")
                    sell_score += 1
            else:
                if profit_pct > 6:
                    sell_signals.append(f"Take profit ({profit_pct:.1f}%)")
                    sell_score += 3
                elif profit_pct > 3:
                    sell_signals.append(f"Partial profit ({profit_pct:.1f}%)")
                    sell_score += 1
        
        required_score = 2 if asset_class == "crypto" else 3
        
        if sell_score >= required_score:
            self.last_signal_time[symbol] = current_time
            return True, f"SELL signals: {', '.join(sell_signals)} (Score: {sell_score})"
        
        return False, f"Weak sell signals: {', '.join(sell_signals)} (Score: {sell_score})"

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
        """Get the exact quantity we own for a symbol from Alpaca"""
        try:
            positions = self.get_positions()
            for pos in positions:
                if pos['symbol'] == symbol:
                    return float(pos['qty'])
            return 0.0
        except Exception as e:
            logger.error(f"Error getting actual position for {symbol}: {e}")
            return 0.0
        
    def _test_connection(self):
        try:
            account = self.get_account()
            if account and 'account_number' in account:
                logger.info(f"Connected to Alpaca successfully")
                logger.info(f"Account: {account.get('account_number', 'N/A')}")
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
                logger.error(f"Account request failed: {response.status_code} - {response.text}")
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
    
    def get_bars(self, symbol: str, timeframe: str = "15Min", limit: int = 200) -> pd.DataFrame:
        try:
            yf_symbol = symbol.replace('/USD', '-USD') if symbol.endswith('/USD') else symbol
            ticker = yf.Ticker(yf_symbol)
            # Use shorter period for crypto to avoid delisting issues
            period = "30d" if symbol.endswith('/USD') else "60d"
            data = ticker.history(period=period, interval="15m")
            if not data.empty:
                return data.tail(limit)
        except Exception as e:
            logger.debug(f"Bars failed for {symbol}: {e}")
        return pd.DataFrame()
    
    def place_order(self, order: Order) -> bool:
        try:
            # Crypto orders need different time_in_force
            time_in_force = "gtc" if order.asset_class == "crypto" else "day"
            
            order_data = {
                "symbol": order.symbol,
                "qty": str(order.quantity),
                "side": order.order_type.value,
                "type": "market",
                "time_in_force": time_in_force
            }
            
            # Handle fractional crypto orders
            if order.asset_class == "crypto" and order.quantity < 1:
                current_price = self.get_current_price(order.symbol)
                if current_price > 0:
                    notional = order.quantity * current_price
                    order_data = {
                        "symbol": order.symbol,
                        "notional": str(round(notional, 2)),
                        "side": order.order_type.value,
                        "type": "market",
                        "time_in_force": time_in_force
                    }
            
            response = requests.post(f"{self.base_url}/v2/orders", headers=self.headers, json=order_data, timeout=10)
            
            if response.status_code in [200, 201]:
                result = response.json()
                order.order_id = result['id']
                order.status = OrderStatus.PENDING
                logger.info(f"Order placed: {order.order_type.value.upper()} {order.quantity} {order.symbol}")
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

class TradingBot:
    def __init__(self, broker: AlpacaBroker, initial_capital: float = 1000.0, data_dir: str = "trading_data"):
        self.broker = broker
        self.initial_capital = initial_capital
        self.market_hours = MarketHours()
        self.memory = FileMemory(data_dir)
        
        # Daily spending limits (hardcoded)
        self.daily_crypto_limit = 1000.0  # $1000 per day for crypto
        self.daily_stock_limit = 2000.0   # $2000 per day for stocks
        
        # Trading symbols
        self.stock_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "SPY", "QQQ", "IWM", "NFLX", "AMD", "CRM", "UBER"]
        # Updated crypto symbols - more conservative list of stable cryptos on Alpaca
        self.crypto_symbols = [
            "BTC/USD",   # Bitcoin
            "ETH/USD",   # Ethereum  
            "LTC/USD",   # Litecoin
            "BCH/USD",   # Bitcoin Cash
            "LINK/USD",  # Chainlink
            "AAVE/USD"   # Aave - removing others that may be delisted
        ]
        self.symbols = self.stock_symbols + self.crypto_symbols
        
        self.strategy = TradingStrategy(self.memory)
        
        # Trading parameters
        self.check_interval = 180  # 3 minutes
        self.max_positions = 8
        self.min_hold_time = 180  # 3 minutes
        # Reduced risk by 1% for both asset classes
        self.crypto_risk_per_trade = 0.02  # 2% risk per crypto trade (down from 3%)
        self.stock_risk_per_trade = 0.02   # 2% risk per stock trade (down from 3%)
        
        self.active_positions = {}
        self._restore_state()
        
        # Performance tracking
        bot_state = self.memory.get_bot_state()
        self.total_pnl = bot_state.get('total_pnl', 0.0)
        self.total_trades = bot_state.get('total_trades', 0)
        self.winning_trades = bot_state.get('winning_trades', 0)
        
        logger.info(f"Trading bot initialized: {len(self.stock_symbols)} stocks + {len(self.crypto_symbols)} cryptos")
        logger.info(f"Daily limits: Crypto ${self.daily_crypto_limit:,}, Stocks ${self.daily_stock_limit:,}")
        logger.info(f"Risk per trade: Crypto {self.crypto_risk_per_trade*100:.0f}%, Stocks {self.stock_risk_per_trade*100:.0f}%")
    
    def _get_asset_class(self, symbol: str) -> str:
        return "crypto" if symbol in self.crypto_symbols else "stock"
    
    def _restore_state(self):
        memory_positions = self.memory.get_open_trades()
        alpaca_positions = self.broker.get_positions()
        alpaca_symbols = {pos['symbol']: pos for pos in alpaca_positions if float(pos['qty']) != 0}
        
        self.active_positions = {}
        
        for symbol, (trade, trade_id) in memory_positions.items():
            if symbol in alpaca_symbols:
                self.active_positions[symbol] = {
                    'trade': trade, 'trade_id': trade_id, 'quantity': trade.quantity,
                    'entry_price': trade.entry_price, 'stop_loss': trade.stop_loss,
                    'timestamp': datetime.fromisoformat(trade.timestamp),
                    'asset_class': trade.asset_class
                }
        
        logger.info(f"Restored {len(self.active_positions)} positions")
    
    def get_daily_spending(self, asset_class: str) -> float:
        """Calculate how much has been spent today on a specific asset class"""
        from datetime import datetime
        
        today = datetime.now().strftime('%Y-%m-%d')  # "2025-08-04"
        all_trades = self.memory.get_trade_history(days=1)
        
        daily_spending = 0.0
        today_trades = []
        
        for trade in all_trades:
            try:
                # Parse the ISO timestamp and extract date
                if hasattr(trade, 'timestamp') and trade.timestamp:
                    # Handle both ISO format "2025-08-04T01:20:10.123456" and simple format
                    trade_date = trade.timestamp.split('T')[0] if 'T' in trade.timestamp else trade.timestamp[:10]
                    
                    if (trade_date == today and 
                        trade.asset_class == asset_class and 
                        trade.status in ['open', 'closed']):
                        
                        trade_value = trade.entry_price * trade.quantity
                        daily_spending += trade_value
                        today_trades.append({
                            'symbol': trade.symbol,
                            'value': trade_value,
                            'timestamp': trade.timestamp
                        })
            except Exception as e:
                # Skip malformed trades
                continue
        
        # Debug logging to see what's being counted
        if today_trades:
            logger.info(f"Today's {asset_class} trades found: {len(today_trades)} trades totaling ${daily_spending:.0f}")
            for t in today_trades[:3]:  # Show first 3 trades
                logger.info(f"  {t['symbol']}: ${t['value']:.0f} at {t['timestamp']}")
        else:
            logger.info(f"No {asset_class} trades found for today ({today})")
        
        return daily_spending
    
    def can_afford_trade(self, symbol: str, position_value: float, asset_class: str) -> Tuple[bool, str]:
        """Check if trade is within daily spending limits"""
        daily_spending = self.get_daily_spending(asset_class)
        
        if asset_class == "crypto":
            limit = self.daily_crypto_limit
            if daily_spending + position_value > limit:
                return False, f"Crypto daily limit exceeded: ${daily_spending:.0f}/${limit:.0f} used"
        else:  # stocks
            limit = self.daily_stock_limit
            if daily_spending + position_value > limit:
                return False, f"Stock daily limit exceeded: ${daily_spending:.0f}/${limit:.0f} used"
        
        return True, f"Within limits: ${daily_spending:.0f}/${limit:.0f} used"
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float, asset_class: str) -> float:
        """Calculate position size using different risk levels for crypto vs stocks"""
        account = self.broker.get_account()
        buying_power = float(account.get('buying_power', self.initial_capital))
        
        # Use different risk per trade based on asset class
        risk_per_trade = self.crypto_risk_per_trade if asset_class == "crypto" else self.stock_risk_per_trade
        risk_amount = buying_power * risk_per_trade
        
        stop_distance = entry_price - stop_loss
        if stop_distance <= 0:
            return 0
        
        position_size = risk_amount / stop_distance
        max_position_value = buying_power * 0.25
        max_shares_by_value = max_position_value / entry_price
        
        if asset_class == "crypto":
            position_size = min(position_size, max_shares_by_value)
            min_shares = 10.0 / entry_price  # $10 minimum
            position_size = max(position_size, min_shares)
        else:
            position_size = max(1, int(min(position_size, max_shares_by_value)))
            position_size = min(position_size, 200)
        
        # *** NEW: ENFORCE DAILY LIMITS IN POSITION SIZING ***
        daily_spending = self.get_daily_spending(asset_class)
        daily_limit = self.daily_crypto_limit if asset_class == "crypto" else self.daily_stock_limit
        remaining_budget = daily_limit - daily_spending
        
        # Cap position size by remaining daily budget
        position_value = position_size * entry_price
        if position_value > remaining_budget:
            if remaining_budget < 10:  # Not worth trading
                return 0
            
            # Reduce position size to fit remaining budget
            position_size = remaining_budget / entry_price
            
            # Ensure minimum viable position
            if asset_class == "stock":
                position_size = max(1, int(position_size))
        
        return position_size
    
    def execute_buy(self, symbol: str, current_price: float, stop_loss: float, reason: str):
        if len(self.active_positions) >= self.max_positions:
            return False
        
        asset_class = self._get_asset_class(symbol)
        
        if asset_class == "stock" and not self.market_hours.is_stock_market_open():
            return False
        
        position_size = self.calculate_position_size(symbol, current_price, stop_loss, asset_class)
        
        if position_size <= 0:
            return False
        
        # *** SIMPLE DAILY LIMIT CHECK - Use original get_daily_spending method ***
        daily_spending = self.get_daily_spending(asset_class)
        daily_limit = self.daily_crypto_limit if asset_class == "crypto" else self.daily_stock_limit
        
        if daily_spending >= daily_limit:
            logger.info(f"Daily {asset_class} limit reached: ${daily_spending:.0f}/${daily_limit:.0f}")
            return False
        
        order = Order(symbol=symbol, order_type=OrderType.BUY, quantity=position_size, asset_class=asset_class)
        
        logger.info(f"BUY {symbol} ({asset_class}): {position_size:.4f} @ ${current_price:.2f} | {reason}")
        logger.info(f"Daily spending: ${daily_spending:.0f}/${daily_limit:.0f}")
        
        if self.broker.place_order(order):
            trade = Trade(
                symbol=symbol, entry_price=current_price, quantity=position_size,
                timestamp=datetime.now().isoformat(), stop_loss=stop_loss,
                asset_class=asset_class, alpaca_order_id=order.order_id,
                status="open", strategy_signals=reason
            )
            
            trade_id = self.memory.save_trade(trade)
            
            self.active_positions[symbol] = {
                'trade': trade, 'trade_id': trade_id, 'quantity': position_size,
                'entry_price': current_price, 'stop_loss': stop_loss,
                'timestamp': datetime.now(), 'asset_class': asset_class
            }
            
            return True
        
        return False



    
   # Fix for crypto precision issues in execute_sell method

    def execute_sell(self, symbol: str, reason: str):
        if symbol not in self.active_positions:
            return False
        
        position = self.active_positions[symbol]
        asset_class = position['asset_class']
        
        if asset_class == "stock" and not self.market_hours.is_stock_market_open():
            if "stop loss" not in reason.lower():
                return False
        
        current_price = self.broker.get_current_price(symbol)
        if current_price <= 0:
            return False
        
        # Check minimum hold time
        hold_time = (datetime.now() - position['timestamp']).seconds
        if hold_time < self.min_hold_time:
            return False
        
        # *** NEW: GET ACTUAL ALPACA POSITION FOR CRYPTO ***
        sell_quantity = position['quantity']
        
        if asset_class == "crypto":
            # Get actual position from Alpaca for crypto
            alpaca_positions = self.broker.get_positions()
            actual_position = None
            
            for pos in alpaca_positions:
                if pos['symbol'] == symbol:
                    actual_position = float(pos['qty'])
                    break
            
            if actual_position is None or actual_position <= 0:
                logger.warning(f"No actual {symbol} position found in Alpaca account")
                # Clean up our tracking
                del self.active_positions[symbol]
                return False
            
            # Use actual position (which accounts for fees, rounding, etc.)
            sell_quantity = actual_position
            
            # Log the difference if significant
            tracked_qty = position['quantity']
            diff = abs(tracked_qty - actual_position)
            if diff > 0.001:  # More than 0.001 difference
                diff_value = diff * current_price
                logger.info(f"Position adjustment for {symbol}: "
                        f"Tracked={tracked_qty:.6f}, Actual={actual_position:.6f}, "
                        f"Diff=${diff_value:.2f}")
        
        order = Order(symbol=symbol, order_type=OrderType.SELL, 
                    quantity=sell_quantity, asset_class=asset_class)
        
        pnl = (current_price - position['entry_price']) * sell_quantity  # Use actual quantity for P&L
        pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
        
        logger.info(f"SELL {symbol} ({asset_class}): {sell_quantity:.4f} @ ${current_price:.2f} | "
                f"P&L: ${pnl:.2f} ({pnl_pct:.1f}%) | {reason}")
        
        if self.broker.place_order(order):
            trade = position['trade']
            trade.exit_price = current_price
            trade.exit_timestamp = datetime.now().isoformat()
            trade.status = "closed"
            trade.pnl = pnl
            trade.pnl_pct = pnl_pct
            trade.exit_reason = reason
            
            # Update trade with actual sold quantity
            trade.quantity = sell_quantity
            
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
        
        return False


    
    def check_stop_losses(self):
        for symbol, position in list(self.active_positions.items()):
            try:
                current_price = self.broker.get_current_price(symbol)
                if current_price <= 0:
                    continue
                
                # Basic stop loss
                if current_price <= position['stop_loss']:
                    logger.warning(f"STOP LOSS triggered for {symbol} @ ${current_price:.2f}")
                    self.execute_sell(symbol, "Stop loss triggered")
                    continue
                
                # Trailing stop for profitable positions
                entry_price = position['entry_price']
                profit_pct = (current_price - entry_price) / entry_price * 100
                
                if profit_pct > 5:  # Trail after 5% profit
                    asset_class = position['asset_class']
                    trail_pct = 0.08 if asset_class == "crypto" else 0.06
                    new_stop = current_price * (1 - trail_pct)
                    
                    if new_stop > position['stop_loss']:
                        position['stop_loss'] = new_stop
                        trade = position['trade']
                        trade.stop_loss = new_stop
                        self.memory.update_trade(trade, position['trade_id'])
                        logger.info(f"Trailing stop updated for {symbol}: ${new_stop:.2f}")
                
            except Exception as e:
                logger.error(f"Error checking stop loss for {symbol}: {e}")
    
    def scan_markets(self):
        stock_market_open = self.market_hours.is_stock_market_open()
        
        for symbol in self.symbols:
            try:
                asset_class = self._get_asset_class(symbol)
                
                if asset_class == "stock" and not stock_market_open:
                    continue
                
                data = self.broker.get_bars(symbol, "15Min", 100)
                if data.empty or len(data) < 50:
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
        
        logger.info("=" * 70)
        logger.info(f"TRADING BOT STATUS - {datetime.now().strftime('%H:%M:%S')}")
        logger.info(self.market_hours.get_market_status_message())
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
                    unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
                    hold_time = datetime.now() - position['timestamp']
                    
                    logger.info(f"    {symbol}: {position['quantity']:.4f} @ ${position['entry_price']:.2f} | "
                              f"Current: ${current_price:.2f} | P&L: ${unrealized_pnl:.2f} ({pnl_pct:.1f}%) | "
                              f"Stop: ${position['stop_loss']:.2f} | Hold: {str(hold_time).split('.')[0]}")
        
        logger.info("=" * 70)
    
    def save_daily_performance(self):
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            account = self.broker.get_account()
            portfolio_value = float(account.get('portfolio_value', self.initial_capital))
            
            today_trades = []
            all_trades = self.memory.get_trade_history(days=1)
            for trade in all_trades:
                if trade.exit_timestamp and trade.exit_timestamp.startswith(today):
                    today_trades.append(trade)
            
            daily_pnl = sum(trade.pnl for trade in today_trades)
            daily_trade_count = len(today_trades)
            win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
            
            bot_state = self.memory.get_bot_state()
            peak_value = max(bot_state.get('peak_portfolio_value', self.initial_capital), portfolio_value)
            drawdown = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0
            
            if portfolio_value > bot_state.get('peak_portfolio_value', 0):
                self.memory.update_bot_state(peak_portfolio_value=portfolio_value)
            
            self.memory.save_daily_performance(today, daily_pnl, daily_trade_count, win_rate, portfolio_value, drawdown)
            
            crypto_trades = len([t for t in today_trades if t.asset_class == "crypto"])
            stock_trades = len([t for t in today_trades if t.asset_class == "stock"])
            
            logger.info(f"Daily performance saved: P&L=${daily_pnl:.2f}, Trades={daily_trade_count} (Stocks: {stock_trades}, Crypto: {crypto_trades})")
            
        except Exception as e:
            logger.error(f"Error saving daily performance: {e}")
    
    def run(self):
        logger.info("STARTING AGGRESSIVE CRYPTO+STOCK TRADING BOT")
        logger.info("Using JSON file-based memory system")
        logger.info(f"3% risk per trade | {self.max_positions} max positions | {self.check_interval}s intervals")
        logger.info("Market hours checking enabled for CST timezone")
        
        logger.info(self.market_hours.get_market_status_message())
        
        iteration = 0
        last_daily_save = datetime.now().date()
        last_market_status_log = datetime.now()
        
        try:
            while True:
                iteration += 1
                start_time = time.time()
                
                # Log market status every hour
                if (datetime.now() - last_market_status_log).total_seconds() >= 3600:
                    logger.info(self.market_hours.get_market_status_message())
                    last_market_status_log = datetime.now()
                
                if iteration % 5 == 0:  # Every 15 minutes
                    self.print_status()
                
                if datetime.now().date() > last_daily_save:
                    self.save_daily_performance()
                    last_daily_save = datetime.now().date()
                
                self.check_stop_losses()
                self.scan_markets()
                
                elapsed = time.time() - start_time
                sleep_time = max(0, self.check_interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Trading stopped by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
        finally:
            self.print_status()
            self.save_daily_performance()
            logger.info("Bot shutdown complete")

def main():
    API_KEY = os.getenv("ALPACA_API_KEY")
    SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    PAPER_TRADING = os.getenv("ALPACA_PAPER_TRADING", "true").lower() == "true"
    DATA_DIR = os.getenv("TRADING_DATA_DIR", "trading_data")
    
    if not API_KEY or not SECRET_KEY:
        logger.error("Missing API keys! Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        return
    
    if PAPER_TRADING and not API_KEY.startswith('PK'):
        logger.warning("Warning: Paper trading enabled but API key doesn't start with 'PK'")
    
    try:
        logger.info("Initializing aggressive crypto+stock trading bot...")
        broker = AlpacaBroker(API_KEY, SECRET_KEY, paper_trading=PAPER_TRADING)
        
        account = broker.get_account()
        if not account or 'account_number' not in account:
            logger.error("Failed to connect to Alpaca API. Cannot start bot.")
            return
        
        bot = TradingBot(broker, data_dir=DATA_DIR)
        bot.run()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")

if __name__ == "__main__":
    main()
