# backend/trading_bot.py

import asyncio
import os
import httpx
import asyncpg
import json
import numpy as np
from datetime import datetime
from kucoin.client import Client as KuCoinClient

# --- Configuration ---
# Load environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
KUCOIN_KEY = os.getenv("KUCOIN_KEY")
KUCOIN_SECRET = os.getenv("KUCOIN_SECRET")
KUCOIN_PASSPHRASE = os.getenv("KUCOIN_PASSPHRASE")

# KuCoin uses different symbols, e.g., SHIB-USDT
TRADING_SYMBOL = 'SHIB-USDT' 
# The asset we are trading
BASE_CURRENCY = 'SHIB'
# The currency we are using to trade
QUOTE_CURRENCY = 'USDT'

# --- KuCoin Sandbox Client ---
# is_sandbox=True connects to the test environment
kucoin_client = KuCoinClient(KUCOIN_KEY, KUCOIN_SECRET, KUCOIN_PASSPHRASE, is_sandbox=True)

db_pool = None

# --- Database Functions ---
async def connect_to_db():
    global db_pool
    if not DATABASE_URL:
        print("ERROR: DATABASE_URL not set.")
        return
    try:
        db_pool = await asyncpg.create_pool(DATABASE_URL, timeout=10)
        print("Bot successfully connected to PostgreSQL database.")
    except Exception as e:
        print(f"CRITICAL ERROR: Bot could not connect to database: {e}")
        db_pool = None

async def save_trade_to_db(trade_data):
    if not db_pool: return
    try:
        await db_pool.execute('''
            INSERT INTO simulated_trades (id, signal_type, entry_price, take_profit_price, stop_loss_price, position_size, status, timestamp, ai_reasoning)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ''',
        trade_data['id'], trade_data['signalType'], float(trade_data['entryPrice']), float(trade_data['takeProfitPrice']),
        float(trade_data['stopLossPrice']), float(trade_data['positionSize']), 'open', # Trades are 'open' when first saved
        trade_data['timestamp'], trade_data['aiReasoning']
        )
        print(f"Saved new OPEN trade {trade_data['id']} to database.")
    except Exception as e:
        print(f"Error saving trade to DB: {e}")

async def update_trade_status_in_db(trade_id, status, outcome_price):
    if not db_pool: return
    try:
        await db_pool.execute(
            'UPDATE simulated_trades SET status = $1, outcome_price = $2 WHERE id = $3;',
            status, outcome_price, trade_id
        )
        print(f"Updated trade {trade_id} status to '{status}' in database.")
    except Exception as e:
        print(f"Error updating trade status in DB: {e}")

async def get_last_n_trades(n=3):
    if not db_pool: return []
    try:
        records = await db_pool.fetch('SELECT * FROM simulated_trades WHERE status != $1 ORDER BY timestamp DESC LIMIT $2;', 'open', n)
        return [dict(record) for record in records]
    except Exception as e:
        print(f"Error fetching last trades: {e}")
        return []

# --- AI and Data Fetching Functions ---
async def get_shib_data():
    # This function combines price and historical data fetching
    try:
        async with httpx.AsyncClient() as client:
            headers = {"x-cg-demo-api-key": COINGECKO_API_KEY} if COINGECKO_API_KEY else {}
            
            # Fetch current price
            price_url = 'https://api.coingecko.com/api/v3/coins/shiba-inu?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false&sparkline=false'
            price_res = await client.get(price_url, headers=headers, timeout=10)
            price_res.raise_for_status()
            current_price = price_res.json()['market_data']['current_price']['usd']

            # Fetch historical data
            ohlc_url = f"https://api.coingecko.com/api/v3/coins/shiba-inu/ohlc?vs_currency=usd&days=30"
            ohlc_res = await client.get(ohlc_url, headers=headers, timeout=10)
            ohlc_res.raise_for_status()
            
            return {"current_price": current_price, "historical_ohlc": ohlc_res.json()}
    except Exception as e:
        print(f"Error fetching data from CoinGecko: {e}")
        return None

async def get_ai_trade_signal(market_data):
    # This is the same logic as in main.py, but self-contained for the bot
    def calculate_sma(series, period):
        if len(series) < period: return None
        return sum(series[-period:]) / period

    def calculate_atr(ohlc_data, period=14):
        if len(ohlc_data) < period + 1: return None
        true_ranges = [max(d[2] - d[3], abs(d[2] - ohlc_data[i-1][4]), abs(d[3] - ohlc_data[i-1][4])) for i, d in enumerate(ohlc_data) if i > 0]
        if not true_ranges: return None
        return np.mean(true_ranges[-period:])

    closing_prices = [d[4] for d in market_data['historical_ohlc']]
    price_sma_10 = calculate_sma(closing_prices, 10)
    atr_14 = calculate_atr(market_data['historical_ohlc'], 14)

    if not all([price_sma_10, atr_14]):
        print("Could not calculate indicators.")
        return None

    yesterday_ohlc = market_data['historical_ohlc'][-1]
    yesterday_open, yesterday_close = yesterday_ohlc[1], yesterday_ohlc[4]

    prompt = f"""
    You are a trading analyst. Evaluate a simple momentum strategy.
    --- Rules ---
    1. Regime Filter: LONG if Current Price > 10-Day SMA; SHORT if Current Price < 10-Day SMA.
    2. Entry Signal: Previous Green Candle (Close > Open) for LONG; Previous Red Candle for SHORT.
    3. Exits: TP = Entry +/- (1.5 * ATR), SL = Entry -/+ (1.0 * ATR).
    4. If conditions fail, return NEUTRAL.
    --- Data ---
    - Current Price: ${market_data['current_price']}
    - Previous Day's Open: ${yesterday_open}
    - Previous Day's Close: ${yesterday_close}
    - 10-Day Price SMA: ${price_sma_10:.8f}
    - 14-Day ATR: ${atr_14:.8f}
    --- Task ---
    Generate the signal and exits in the specified JSON format.
    """
    
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT", "properties": {
                    "signal_type": {"type": "STRING", "enum": ["LONG", "SHORT", "NEUTRAL"]},
                    "reasoning": {"type": "STRING"},
                    "take_profit_price": {"type": "NUMBER"},
                    "stop_loss_price": {"type": "NUMBER"}
                }, "required": ["signal_type", "reasoning"]
            }
        }
    }
    
    gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(gemini_api_url, json=payload, timeout=45.0)
            response.raise_for_status()
            result = response.json()
        return json.loads(result['candidates'][0]['content']['parts'][0]['text'])
    except Exception as e:
        print(f"Error getting AI signal: {e}")
        return None

async def analyze_strategy(trades):
    # This is the same logic as the review endpoint in main.py
    print("Analyzing strategy after 3 consecutive losses...")
    # ... (omitting full prompt for brevity, but it's the same as in main.py)
    # In a real scenario, you would call the same review logic here.
    # For this simulation, we will just log a message.
    print("--- AI STRATEGY REVIEW ---")
    print("OBSERVATIONS: The strategy has experienced 3 consecutive losses. This may indicate the current parameters are not suited for the market conditions.")
    print("RECOMMENDATIONS: Consider widening the stop-loss from 1.0*ATR to 1.5*ATR or adding a stronger trend confirmation filter.")
    print("--- TRADING PAUSED ---")


# --- Main Trading Logic ---
async def run_trading_loop():
    trading_paused = False
    
    while True:
        if trading_paused:
            print("Trading is paused due to consecutive losses. Please review strategy and restart bot.")
            await asyncio.sleep(3600) # Sleep for an hour before checking again
            continue

        print(f"\n--- New Cycle Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        
        # 1. Check for existing open positions
        # In a real bot, you would check KuCoin for open orders. We'll simulate by checking our DB.
        open_trades = await db_pool.fetch("SELECT * FROM simulated_trades WHERE status = 'open'")
        if open_trades:
            print(f"Found {len(open_trades)} open position(s). Monitoring existing trades instead of creating a new one.")
            # The monitoring task (started separately) will handle these.
            await asyncio.sleep(1800) # 30 min sleep
            continue

        # 2. Fetch data and get AI signal
        market_data = await get_shib_data()
        if not market_data:
            await asyncio.sleep(60)
            continue
        
        ai_signal = await get_ai_trade_signal(market_data)
        if not ai_signal or ai_signal.get('signal_type') == 'NEUTRAL':
            print("AI signal is NEUTRAL. No trade executed.")
            await asyncio.sleep(1800) # 30 min sleep
            continue

        # 3. Execute Trade
        print(f"AI signal is {ai_signal['signal_type']}. Preparing to execute trade.")
        try:
            # Get account balance
            accounts = kucoin_client.get_accounts(currency=QUOTE_CURRENCY)
            usdt_balance = float(accounts[0]['available'])
            
            # Calculate position size (2% of bankroll)
            position_size_usd = usdt_balance * 0.02
            position_size_shib = position_size_usd / market_data['current_price']
            
            print(f"Bankroll: {usdt_balance:.2f} USDT. Position size: {position_size_usd:.2f} USDT ({position_size_shib:,.0f} SHIB).")

            # Place market order
            order_result = kucoin_client.create_market_order(
                symbol=TRADING_SYMBOL,
                side='buy' if ai_signal['signal_type'] == 'LONG' else 'sell',
                size=int(position_size_shib) # KuCoin requires integer size for SHIB
            )
            print(f"Trade executed on KuCoin. Order ID: {order_result['orderId']}")

            # Save trade to our database for monitoring
            trade_to_save = {
                "id": int(order_result['orderId'].replace('-', '')), # Use orderId as unique ID
                "signalType": ai_signal['signal_type'],
                "entryPrice": market_data['current_price'],
                "takeProfitPrice": ai_signal['take_profit_price'],
                "stopLossPrice": ai_signal['stop_loss_price'],
                "positionSize": position_size_shib,
                "timestamp": int(datetime.now().timestamp() * 1000),
                "aiReasoning": ai_signal['reasoning'],
            }
            await save_trade_to_db(trade_to_save)

        except Exception as e:
            print(f"CRITICAL ERROR during trade execution: {e}")

        await asyncio.sleep(1800) # 30 min sleep


async def monitor_open_trades():
    while True:
        await asyncio.sleep(30) # Check every 30 seconds
        try:
            open_trades = await db_pool.fetch("SELECT * FROM simulated_trades WHERE status = 'open'")
            if not open_trades:
                continue

            print(f"Monitoring {len(open_trades)} open trade(s)...")
            
            # Get current price
            ticker = kucoin_client.get_ticker(TRADING_SYMBOL)
            current_price = float(ticker['price'])

            for trade in open_trades:
                trade = dict(trade)
                should_close = False
                status = 'open'

                if trade['signal_type'] == 'LONG':
                    if current_price >= trade['take_profit_price']:
                        should_close, status = True, 'win'
                    elif current_price <= trade['stop_loss_price']:
                        should_close, status = True, 'loss'
                elif trade['signal_type'] == 'SHORT':
                    if current_price <= trade['take_profit_price']:
                        should_close, status = True, 'win'
                    elif current_price >= trade['stop_loss_price']:
                        should_close, status = True, 'loss'
                
                if should_close:
                    print(f"Closing trade {trade['id']} with status: {status}")
                    # In a real bot, you'd place the closing order on KuCoin here
                    # For sandbox, we just update our DB
                    await update_trade_status_in_db(trade['id'], status, current_price)
                    
                    # Check for losing streak
                    last_trades = await get_last_n_trades(3)
                    if len(last_trades) == 3 and all(t['status'] == 'loss' for t in last_trades):
                        await analyze_strategy(last_trades)
                        # This global variable will be seen by the main loop
                        global trading_paused
                        trading_paused = True


        except Exception as e:
            print(f"Error in monitoring loop: {e}")


async def main():
    await connect_to_db()
    if not db_pool:
        print("Exiting bot due to database connection failure.")
        return

    # Create and run the two main tasks concurrently
    trading_task = asyncio.create_task(run_trading_loop())
    monitoring_task = asyncio.create_task(monitor_open_trades())

    await asyncio.gather(trading_task, monitoring_task)


if __name__ == "__main__":
    print("Starting SHIB Trading Bot...")
    # Basic check for environment variables
    if not all([DATABASE_URL, GEMINI_API_KEY, KUCOIN_KEY, KUCOIN_SECRET, KUCOIN_PASSPHRASE]):
        print("CRITICAL ERROR: One or more required environment variables are missing. Exiting.")
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("Bot stopped by user.")
