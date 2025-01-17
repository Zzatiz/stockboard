import os
import requests
import logging
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import pandas as pd
import threading
import json
import random
import yfinance as yf

from calculations import (
    calculate_bollinger_signals,
    calculate_ema_signals,
    calculate_fib_signals,
    calculate_ewt_signals,
    calc_support_put_call_flow  # <--- your new method
  # <--- add this line
)

# This is the Tastytrade SDK you installed
from tastytrade_sdk import Tastytrade
from flask_socketio import SocketIO
from tastytrade_sdk.market_data.subscription import Subscription

load_dotenv()

# -------------- Global Setup & Logging -------------- #
USER_AGENT = "my-tt-client/1.0"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------- Tastytrade & Alpaca Credentials -------------- #
TASTYTRADE_USERNAME = os.getenv("TASTYTRADE_USERNAME")
TASTYTRADE_PASSWORD = os.getenv("TASTYTRADE_PASSWORD")
TASTYTRADE_API_BASE = "https://api.tastytrade.com"

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# -------------- Tastytrade Setup -------------- #\
tasty = Tastytrade().login(login=TASTYTRADE_USERNAME, password=TASTYTRADE_PASSWORD)

GLOBAL_SESSION_TOKEN = None
def handle_quote(quote_data):
    try:
        if isinstance(quote_data, dict):
            print("Received Quote:", quote_data)
        else:
            print("Unexpected Quote Format:", quote_data)
    except Exception as e:
        print(f"Error in handle_quote: {e}")

tasty = Tastytrade().login(login=TASTYTRADE_USERNAME, password=TASTYTRADE_PASSWORD)

# subscription = tasty.market_data.subscribe(
#     symbols=["AAPL"],  # Replace with a valid symbol if needed
#     on_quote=handle_quote
# )
# subscription.open()
# -------------- Flask & SocketIO Setup -------------- #
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

def refresh_session_token():
    """
    Refresh Tastytrade session token if you're calling Tastytrade's REST endpoints manually.
    This is separate from the tastytrade_sdk usage.
    """
    global GLOBAL_SESSION_TOKEN
    url = f"{TASTYTRADE_API_BASE}/sessions"
    payload = {"login": TASTYTRADE_USERNAME, "password": TASTYTRADE_PASSWORD}
    headers = {"Content-Type": "application/json"}

    resp = requests.post(url, json=payload, headers=headers)
    if resp.status_code == 201:
        GLOBAL_SESSION_TOKEN = resp.json().get("data", {}).get("session-token")
        logging.info(f"Session token refreshed: {GLOBAL_SESSION_TOKEN}")
    else:
        logging.error(f"Failed to refresh session token: {resp.text}")
        raise RuntimeError("Unable to refresh session token.")

def get_auth_headers():
    """
    For direct Tastytrade REST calls if not using tastytrade_sdk.
    """
    if not GLOBAL_SESSION_TOKEN:
        refresh_session_token()
    return {
        "Authorization": f"{GLOBAL_SESSION_TOKEN}",
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

# -------------------------------------------------------------------
# UTILITY: Build Equity Option Symbol for Tastytrade
# -------------------------------------------------------------------
def build_equity_option_symbol(ticker, option_type, strike, expiration):
    """
    Builds a Tastytrade-style option symbol, e.g. AAPL210115C00150000
    For AAPL, 2025-01-17 call 150.
    """
    t = ticker.ljust(6)[:6]
    # Convert YYYY-MM-DD => YYMMDD
    yymmdd = expiration[2:4] + expiration[5:7] + expiration[8:10]
    c_or_p = "C" if option_type.lower().startswith("c") else "P"
    strike_int = int(strike * 1000)
    strike_str = str(strike_int).rjust(8, "0")
    return f"{t}{yymmdd}{c_or_p}{strike_str}"

def place_options_trade(ticker, option_type, strike_price, expiration_date, quantity, order_type, price=None):
    """
    Example function if you want to place trades directly vs. Tastytrade's REST.
    """
    headers = get_auth_headers()
    tasty_account_num = os.getenv("TASTYTRADE_ACCOUNT_NUM")
    if not tasty_account_num:
        raise ValueError("Missing Tastytrade account number in .env (TASTYTRADE_ACCOUNT_NUM).")

    request_body = {
        "time-in-force": "Day",
        "order-type": "Limit" if price else "Market",
        "price": float(price) if price else None,
        "price-effect": "Debit" if order_type.lower() == "buy" else "Credit",
        "legs": [
            {
                "instrument-type": "Equity Option",
                "symbol": build_equity_option_symbol(ticker, option_type, strike_price, expiration_date),
                "quantity": float(quantity),
                "action": "Buy to Open" if order_type.lower() == "buy" else "Sell to Open"
            }
        ]
    }

    if not price:
        request_body.pop("price", None)  # Remove if Market order

    order_url = f"{TASTYTRADE_API_BASE}/accounts/{tasty_account_num}/orders"
    resp = requests.post(order_url, headers=headers, json=request_body)
    if not resp.ok:
        raise RuntimeError(f"Tastytrade order error: {resp.status_code} - {resp.text}")
    return resp.json()
# Apply the monkey patch
# -------------------------------------------------------------------
# YFinance Timeframe Logic
# -------------------------------------------------------------------
def get_historical_data_yf(ticker, interval="1d", period="1mo"):
    """
    Fetch historical data from yfinance based on interval and period.
    """
    # Define valid intervals and their corresponding maximum periods as per yfinance
    tf_map = {
        "1m":  ("1m",  "5d"),     # Changed from "7d" to "5d"
        "2m":  ("2m",  "5d"),
        "5m":  ("5m",  "5d"),
        "15m": ("15m", "60d"),
        "30m": ("30m", "60d"),
        "60m": ("60m", "730d"),   # Approximately 2 years
        "90m": ("90m", "730d"),
        "1h":  ("1h",  "730d"),
        "1d":  ("1d",  "max"),
        "5d":  ("1d",  "max"),    # yfinance doesn't support 5d period with 1d interval; using 'max' period
        "1wk": ("1wk", "max"),
        "1mo": ("1mo", "max"),
        "3mo": ("3mo", "max"),
    }

    # Validate and map the interval to yfinance's supported interval and period
    if interval not in tf_map:
        logging.error(f"Invalid interval '{interval}'. Falling back to '1d'.")
        yf_interval, yf_period = ("1d", "1mo")
    else:
        yf_interval, yf_period = tf_map[interval]

    # Log the fetching parameters
    logging.info(f"Fetching {ticker}: interval={yf_interval}, period={yf_period}")

    try:
        # Fetch data using yfinance
        df = yf.download(ticker, period=yf_period, interval=yf_interval, progress=False)
        
        # Check if data is returned
        if df.empty:
            logging.warning(f"No data found for {ticker} with interval '{yf_interval}' and period '{yf_period}'.")
            return []
        
        # Sort the DataFrame by date
        df = df.sort_index()
        
        # Convert DataFrame to list of candle dictionaries
        candles = []
        for idx, row in df.iterrows():
            candles.append({
                "time": int(idx.timestamp()),
                "open": float(row["Open"]) if not isinstance(row["Open"], pd.Series) else float(row["Open"].iloc[0]),
                "high": float(row["High"]) if not isinstance(row["High"], pd.Series) else float(row["High"].iloc[0]),
                "low":  float(row["Low"]) if not isinstance(row["Low"], pd.Series) else float(row["Low"].iloc[0]),
                "close": float(row["Close"]) if not isinstance(row["Close"], pd.Series) else float(row["Close"].iloc[0]),
            })
        logging.info(f"Fetched {len(candles)} candles for {ticker}.")
        return candles
    except yf.YFInvalidPeriodError as e:
        logging.error(f"YFInvalidPeriodError for ticker '{ticker}': {e}")
        return []
    except Exception as e:
        logging.error(f"Error fetching data from yfinance for ticker '{ticker}': {e}")
        return []
# -------------------------------------------------------------------
# Endpoint: /api/fetchData
# -------------------------------------------------------------------
@app.route("/api/fetchData", methods=["GET"])
def fetch_data():
    """
    Endpoint to fetch historical data from yfinance based on ticker, interval, and period.
    """
    ticker = request.args.get("ticker", "AAPL").upper()
    interval = request.args.get("interval", "1d").lower()
    period = request.args.get("period", "1mo").lower()

    # Validate interval and period based on yfinance capabilities
    valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
    valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]

    if interval not in valid_intervals:
        return jsonify({"error": f"Invalid interval '{interval}'. Valid options are: {', '.join(valid_intervals)}."}), 400

    if period not in valid_periods:
        return jsonify({"error": f"Invalid period '{period}'. Valid options are: {', '.join(valid_periods)}."}), 400

    data = get_historical_data_yf(ticker, interval=interval, period=period)  # Modify function to accept interval and period
    return jsonify(data)

# -------------------------------------------------------------------
# Endpoint: /api/getPrice
# -------------------------------------------------------------------
@app.route("/api/getPrice", methods=["GET"])
def get_price():
    """
    Quick 'live' price check via yfinance or Tastytrade, skipping if market is closed.
    """
    ticker = request.args.get("ticker", "AAPL").upper()
    # if not is_market_open():
    #     return jsonify({"ticker": ticker, "price": None, "warning": "Market is closed."})
    try:
        latest = yf.Ticker(ticker).history(period="1d").tail(1)
        if latest.empty:
            raise ValueError(f"No data found for {ticker}")
        current_price = float(latest["Close"].iloc[-1])
        return jsonify({"ticker": ticker, "price": current_price})
    except Exception as e:
        logging.error(f"Error getting live price for {ticker}: {e}")
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------------------------
# Endpoint: /api/startLiveStream and /api/stopLiveStream
# Using tastytrade_sdk for streaming
# -------------------------------------------------------------------
@app.route("/api/saveLog", methods=["POST"])
def save_log():
    """
    Append client-side logs to server-side file (client_logs.txt).
    """
    data = request.json or {}
    msg = data.get("msg", "")
    # Optional: add date/time
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    
    with open("client_logs.log", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")
    
    return jsonify({"status": "ok", "logged": msg})

# -------------------------------------------------------------------
# Endpoint: /api/placeTrade
# -------------------------------------------------------------------
@app.route("/api/placeTrade", methods=["POST"])
def place_trade():
    """
    Example Tastytrade trade usage.  If you want to do a direct Tastytrade REST call.
    """
    try:
        payload = request.json or {}
        ticker = payload.get("ticker")
        option_type = payload.get("option_type")
        strike_price = payload.get("strike_price")
        expiration_date = payload.get("expiration_date")
        quantity = payload.get("quantity", 1)
        order_type = payload.get("order_type", "market")
        price = payload.get("price", None)

        if not ticker or not strike_price or not expiration_date:
            return jsonify({"error": "Missing ticker / strike_price / expiration_date."}), 400

        result = place_options_trade(
            ticker,
            option_type,
            strike_price,
            expiration_date,
            quantity,
            order_type,
            price
        )
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error placing trade: {e}")
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------------------------
# Endpoint: /api/runQuant
# -------------------------------------------------------------------
@app.route("/api/runQuant", methods=["POST"])
def run_quant():
    payload = request.json or {}
    ticker  = payload.get("ticker", "AAPL").upper()
    theories = payload.get("theories", {})
    custom_params = payload.get("custom_params", {})
    client_candles = payload.get("client_candles", [])
    daily_start_time_utc = custom_params.get("daily_start_time_utc", "00:00")  # Default to "00:00" if not provided

    support_price    = custom_params.get("support_price", None)
    user_in_position = custom_params.get("user_in_position", None)

    calc_steps_all = []

    # Convert user_in_position to float if present
    if user_in_position is not None:
        try:
            user_in_position = float(user_in_position)
        except ValueError:
            user_in_position = None

    try:
        # 1) Use client_candles if provided, else get from yfinance 1m/1d
        if client_candles and isinstance(client_candles, list):
            data = [
                {
                    "time":  c.get("time"),
                    "open":  c.get("open"),
                    "high":  c.get("high"),
                    "low":   c.get("low"),
                    "close": c.get("close"),
                } for c in client_candles
            ]
        else:
            data = get_historical_data_yf(ticker)  # Forces 1m/1d data

        if not data:
            message = f"No data found or received for {ticker} [1m,1d]."
            logging.warning(message)
            return jsonify({"message": message, "markers": []})

        # 2) Determine initial support
        initial_support = None
        if support_price is not None:
            try:
                support_price = float(support_price)
                min_close = min(c["close"] for c in data)
                max_close = max(c["close"] for c in data)
                if min_close <= support_price <= max_close:
                    initial_support = support_price
                    calc_steps_all.append(f"User provided support price: {initial_support:.2f}")
                else:
                    logging.warning(f"Provided support_price {support_price} is outside data range.")
                    initial_support = min_close
                    calc_steps_all.append(f"Provided support_price {support_price:.2f} invalid => using {initial_support:.2f}")
            except ValueError:
                logging.warning(f"Invalid support_price input => using min close.")
                initial_support = min(c["close"] for c in data)
                calc_steps_all.append(f"Invalid support => using {initial_support:.2f}")
        else:
            initial_support = min(c["close"] for c in data)
            calc_steps_all.append(f"No user support => using min close {initial_support:.2f}")

        # 3) Theories
        signals = []
        # Support Flow
        if theories.get("support_flow"):
            sf_signals, sf_steps = calc_support_put_call_flow(
                historical_candles=data,
                support_initial=initial_support,
                daily_start_time_utc=daily_start_time_utc  # Pass the UTC start time
            )
            signals.extend(sf_signals)
            calc_steps_all.extend(sf_steps)

        # Bollinger
        if theories.get("bollinger"):
            b_signals, b_steps = calculate_bollinger_signals(
                historical_candles=data,
                daily_start_time_utc=daily_start_time_utc  # Pass the UTC start time if required
            )
            signals.extend(b_signals)
            calc_steps_all.extend(b_steps)

        # EMA
        if theories.get("ema"):
            e_signals, e_steps = calculate_ema_signals(
                historical_candles=data,
                daily_start_time_utc=daily_start_time_utc  # Pass the UTC start time
            )
            signals.extend(e_signals)
            calc_steps_all.extend(e_steps)

        # FIB
        if theories.get("fib"):
            f_signals, f_steps = calculate_fib_signals(
                historical_candles=data,
                daily_start_time_utc=daily_start_time_utc  # Pass the UTC start time
            )
            signals.extend(f_signals)
            calc_steps_all.extend(f_steps)

        # EWT
        if theories.get("ewt"):
            w_signals, w_steps = calculate_ewt_signals(
                historical_candles=data,
                daily_start_time_utc=daily_start_time_utc  # Pass the UTC start time
            )
            signals.extend(w_signals)
            calc_steps_all.extend(w_steps)

        if not signals:
            return jsonify({
                "message": "No quant signals generated.",
                "markers": [],
                "calc_log": calc_steps_all
            })

        # 4) Convert signals => markers
        markers = []
        for s in signals:
            # Ensure 'price' key exists and is a float
            if 'price' not in s or not isinstance(s['price'], (float, int)):
                logging.warning(f"Signal missing or invalid 'price': {s}. Skipping.")
                continue  # Skip this signal or handle as needed

            # Ensure 'theory' key exists
            theory = s.get("theory", "unknown").title()

            rounded_price = round(float(s["price"]), 2)
            color = "#ccc"
            shape = "circle"
            position = "aboveBar"
            text_label = theory

            if s["signal"] == "sell":
                color = "#DD0000"  # Red
                shape = "arrowDown"
                position = "belowBar"
                text_label = f"{theory} SELL @ {rounded_price:.2f}"
            elif s["signal"] == "hold":
                color = "#FFA500"  # Orange for "Hold"
                shape = "arrowRight"
                position = "aboveBar"
                text_label = f"{theory} HOLD @ {rounded_price:.2f}"
            elif s["signal"] == "buy":
                color = "#00AA00"  # Green for "Buy"
                shape = "arrowUp"
                position = "aboveBar"
                text_label = f"{theory} BUY @ {rounded_price:.2f}"
            elif s["signal"].startswith("end_"):
                # Handle end markers
                if s["signal"] == "end_hold":
                    color = "#FFA500"
                    shape = "arrowDown"
                    position = "belowBar"
                    text_label = f"End HOLD @ {rounded_price:.2f}"
                elif s["signal"] == "end_sell":
                    color = "#DD0000"
                    shape = "arrowUp"
                    position = "aboveBar"
                    text_label = f"End SELL @ {rounded_price:.2f}"
                elif s["signal"] == "end_buy":
                    color = "#00AA00"
                    shape = "arrowDown"
                    position = "belowBar"
                    text_label = f"End BUY @ {rounded_price:.2f}"
                else:
                    # Unknown end signal
                    logging.warning(f"Unknown end signal type: {s['signal']}. Skipping.")
                    continue
            else:
                # Unknown signal type
                logging.warning(f"Unknown signal type: {s['signal']}. Skipping.")
                continue

            markers.append({
                "time": s["time"],
                "position": position,
                "color": color,
                "shape": shape,
                "text": text_label
            })

        # Optional: Log signals and markers to a server-side log file
        try:
            with open("./calc_log.log", "a") as log_file:
                log_file.write("\n--- QUANT CALCULATION LOG ---\n")
                log_file.write(f"Ticker: {ticker} | Interval: 1m, Period: 1d\n")
                log_file.write(f"User IN Position: {user_in_position}\n")
                log_file.write(f"Theories Enabled: {theories}\n")
                log_file.write("Client Candle Data:\n")
                for c in data[:5]:  # Log the first 5 candles for brevity
                    log_file.write(f"  {c}\n")
                if len(data) > 5:
                    log_file.write(f"  ... ({len(data) - 5} more candles)\n")

                log_file.write("Signals Generated:\n")
                for idx, sig in enumerate(signals, start=1):
                    log_file.write(f"{idx}. time={sig['time']}, signal={sig['signal']}, "
                                   f"theory={sig['theory']}, price={round(sig['price'], 2) if 'price' in sig else 'N/A'}\n")

                log_file.write("Markers:\n")
                for m in markers:
                    log_file.write(f"time={m['time']}, text={m['text']}, color={m['color']}, "
                                   f"shape={m['shape']}, pos={m['position']}\n")
                log_file.write("--- END QUANT CALC LOG ---\n")
        except Exception as log_err:
            logging.error(f"Failed to write calc log: {log_err}")

        return jsonify({
            "message": "Quant calculations completed.",
            "markers": markers,
            "calc_log": calc_steps_all
        })

    except Exception as e:
        error_msg = f"Error during quant calculation: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return jsonify({
            "message": "An error occurred during quant calculations. Check logs.",
            "markers": []
        })

def get_bar_interval_seconds(interval):
    """
    Maps yfinance interval to seconds.
    """
    mapping = {
        "1m": 60,
        "2m": 120,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "60m": 3600,
        "90m": 5400,
        "1h": 3600,
        "1d": 86400,
        "5d": 86400,
        "1wk": 604800,
        "1mo": 2629800,  # Approximate
        "3mo": 7889400,  # Approximate
    }
    return mapping.get(interval, 3600)  # Default to 1 hour if not found

# -------------------------------------------------------------------
# Main - Start SocketIO
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Attempt Tastytrade login at startup
    try:
        tasty.login(TASTYTRADE_USERNAME, TASTYTRADE_PASSWORD)
        logging.info("Logged in to Tastytrade.")
    except Exception as e:
        logging.warning(f"Could not log in to Tastytrade initially: {e}")

    socketio.run(app, debug=False, port=5000)
