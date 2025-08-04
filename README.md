# tradingbottest
messing around with algo trading in a simple codebase, please dont use real money with this bot that is not responsible and I am sure there are better bots for that than this one.


API Configuration
Get Alpaca API Keys

Create an account at Alpaca Markets
For Paper Trading (Recommended):

Go to Paper Trading Dashboard
Generate API Key & Secret (keys will start with 'PK' for paper trading)


For Live Trading:

Fund your account and generate live API keys
⚠️ WARNING: Live trading involves real money and risk



Set Environment Variables
Windows (PowerShell):
powershell$env:ALPACA_API_KEY = "your_api_key_here"
$env:ALPACA_SECRET_KEY = "your_secret_key_here"
$env:ALPACA_PAPER_TRADING = "true"

Linux/Mac (Bash):
bashexport ALPACA_API_KEY="your_api_key_here"
export ALPACA_SECRET_KEY="your_secret_key_here"
export ALPACA_PAPER_TRADING="true"

Or create a .env file:
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_PAPER_TRADING=true
