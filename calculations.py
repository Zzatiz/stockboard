import pandas as pd
import numpy as np
import logging
import datetime
###############################################################################
# Helper: Group Candles by Custom Daily Start Time
###############################################################################
def _group_candles_by_day(df, daily_start_time_utc="00:00"):
    """
    Split the DataFrame into daily segments based on a custom daily start time in UTC.
    Returns a list of sub-dataframes, each for one trading day.

    :param df: DataFrame with 'time' column as UNIX timestamp in seconds.
    :param daily_start_time_utc: String "HH:MM" indicating when a new day starts in UTC.
    """
    # Parse the daily start time
    start_hour, start_minute = map(int, daily_start_time_utc.split(":"))
    
    # Convert UNIX timestamp to UTC datetime
    df["datetime"] = pd.to_datetime(df["time"], unit='s', utc=True)
    
    # Shift datetime by the daily start time to align sessions
    df["shifted_datetime"] = df["datetime"] - pd.Timedelta(hours=start_hour, minutes=start_minute)
    
    # Extract the session date based on the shifted datetime
    df["session_date"] = df["shifted_datetime"].dt.date
    
    # Group by session_date
    grouped = []
    for date_val, sub_df in df.groupby("session_date"):
        # Sort by time within the group
        sub_df = sub_df.sort_values("time").reset_index(drop=True)
        grouped.append(sub_df)
    
    # Clean up helper columns
    df.drop(columns=["datetime", "shifted_datetime", "session_date"], inplace=True, errors="ignore")
    
    return grouped


###############################################################################
# Signal Filters
###############################################################################
def _skip_repeated_signals(signals):
    """
    Remove consecutive signals that have the same 'signal'.
    E.g., BUY => BUY => SELL, remove the second BUY.
    """
    if not signals:
        return signals
    
    filtered = [signals[0]]
    for s in signals[1:]:
        if s["signal"] == filtered[-1]["signal"]:
            # Skip repeated signal
            continue
        filtered.append(s)
    return filtered

def _enforce_profitable_trades(signals, min_profit=0.00):
    """
    Enforce a minimal profit only if you really want to skip unprofitable trades.
    By default, set min_profit=0.00 to allow frequent signals.
    """
    if len(signals) < 2:
        return signals
    
    filtered = [signals[0]]
    last_sig = filtered[0]["signal"]
    last_price = filtered[0]["price"]

    for s in signals[1:]:
        cur_sig = s["signal"]
        cur_price = s["price"]
        
        if last_sig == "buy" and cur_sig == "sell":
            # Ensure sell price is higher than buy price by min_profit
            if cur_price >= last_price + min_profit:
                filtered.append(s)
                last_sig = cur_sig
                last_price = cur_price
        
        elif last_sig == "sell" and cur_sig == "buy":
            # Ensure buy price is lower than sell price by min_profit
            if cur_price <= last_price - min_profit:
                filtered.append(s)
                last_sig = cur_sig
                last_price = cur_price
        
        # If same signal or not profitable, skip
    return filtered

def _filter_signals_by_gap(signals, min_bars_gap, bar_interval_seconds):
    """
    Ensures we don't get signals too close in time or with minimal price changes.
    If you want more signals, reduce `min_bars_gap` or remove the price-diff ratio check.
    """
    if not signals:
        return []
    
    filtered = []
    last_signal_time = None
    
    for s in signals:
        if last_signal_time is None:
            filtered.append(s)
            last_signal_time = s["time"]
            continue
        
        time_gap = s["time"] - last_signal_time
        if time_gap < (min_bars_gap * bar_interval_seconds):
            # Skip signals that are too close in time
            continue

        # If you want more signals, consider removing or lowering ratio < 0.003
        price_diff_ratio = abs(s["price"] - filtered[-1]["price"]) / max(filtered[-1]["price"], 1e-9)
        if price_diff_ratio < 0.002:  
            # ~0.2% difference - adjust or remove if you want more signals
            continue
        
        filtered.append(s)
        last_signal_time = s["time"]
    
    return filtered

###############################################################################
# Advanced Wave Detection
###############################################################################
def detect_advanced_waves(
    df,
    zigzag_pct=0.5,       # normal ZigZag threshold, e.g., 0.5%
    big_drop_pct=3.0,     # wait for first drop bigger than 3% to define initial bottom
    reset_drop_pct=10.0,  # if we see drop >=10%, forcibly reset wave detection from that candle
    bounce_factor=2.0     # if bounce is bigger than (bounce_factor × last_bounce), treat as wave pivot
):
    """
    1) Wait for a first big drop (>= big_drop_pct) from the initial open to call that a 'major bottom'.
    2) Then do a ZigZag approach with threshold=zigzag_pct.
    3) If we see drop >= reset_drop_pct => reset wave detection.
    4) If while falling, each bounce is < 1%, except we see a bounce that is > bounce_factor × last bounce => treat as wave pivot.

    Returns a list of wave segments (direction='up'/'down', start_idx, end_idx, times, prices).
    """
    if df.empty:
        return []

    prices = df["close"].values
    times = df["time"].values
    n = len(prices)

    def pct_change(old, new):
        return ((new - old) / max(abs(old), 1e-9)) * 100.0

    waves = []
    pivot_idx = 0
    pivot_price = prices[0]
    direction = None   # 'up' or 'down'
    last_bounce_size = 0.0
    found_first_bottom = False

    for i in range(1, n):
        cur_price = prices[i]
        move_pct = pct_change(pivot_price, cur_price)

        # ~ (A) Wait for first big drop => define bottom
        if not found_first_bottom:
            if move_pct <= -big_drop_pct:
                found_first_bottom = True
                direction = "up"  
                pivot_idx = i
                pivot_price = cur_price
            else:
                # Update pivot if we go higher
                if cur_price > pivot_price:
                    pivot_idx = i
                    pivot_price = cur_price
            continue

        # (B) Once we have a major bottom, do wave detection with resets
        if direction == "up":
            if move_pct <= -reset_drop_pct:
                # Forcibly end wave & reset
                waves.append({
                    "direction": "up",
                    "start_idx": pivot_idx,
                    "end_idx": i-1,
                    "start_time": float(times[pivot_idx]),
                    "end_time": float(times[i-1]),
                    "start_price": float(prices[pivot_idx]),
                    "end_price": float(prices[i-1])
                })
                pivot_idx = i
                pivot_price = cur_price
                direction = "down"
                continue

            # Normal ZigZag: if price has dropped from pivot >= zigzag_pct => pivot was top
            if move_pct <= -zigzag_pct:
                waves.append({
                    "direction": "up",
                    "start_idx": pivot_idx,
                    "end_idx": i-1,
                    "start_time": float(times[pivot_idx]),
                    "end_time": float(times[i-1]),
                    "start_price": float(prices[pivot_idx]),
                    "end_price": float(prices[i-1])
                })
                pivot_idx = i-1
                pivot_price = prices[i-1]
                direction = "down"

            else:
                if cur_price > pivot_price:
                    pivot_idx = i
                    pivot_price = cur_price

        else:  # direction == "down"
            if move_pct >= reset_drop_pct:
                # Forcibly end wave & reset
                waves.append({
                    "direction": "down",
                    "start_idx": pivot_idx,
                    "end_idx": i-1,
                    "start_time": float(times[pivot_idx]),
                    "end_time": float(times[i-1]),
                    "start_price": float(prices[pivot_idx]),
                    "end_price": float(prices[i-1])
                })
                pivot_idx = i
                pivot_price = cur_price
                direction = "up"
                continue

            # Normal ZigZag: if price has risen from pivot >= zigzag_pct => pivot was bottom
            if move_pct >= zigzag_pct:
                waves.append({
                    "direction": "down",
                    "start_idx": pivot_idx,
                    "end_idx": i-1,
                    "start_time": float(times[pivot_idx]),
                    "end_time": float(times[i-1]),
                    "start_price": float(prices[pivot_idx]),
                    "end_price": float(prices[i-1])
                })
                new_bounce_size = abs(pct_change(prices[i-1], cur_price))
                if new_bounce_size > bounce_factor * last_bounce_size:
                    pivot_idx = i-1
                    pivot_price = prices[i-1]
                else:
                    pivot_idx = i
                    pivot_price = cur_price

                direction = "up"
                last_bounce_size = new_bounce_size
            else:
                if cur_price < pivot_price:
                    pivot_idx = i
                    pivot_price = cur_price

    # Handle final wave
    waves.append({
        "direction": direction if direction else "down",
        "start_idx": pivot_idx,
        "end_idx": n-1,
        "start_time": float(times[pivot_idx]),
        "end_time": float(times[n-1]),
        "start_price": float(prices[pivot_idx]),
        "end_price": float(prices[n-1])
    })

    return waves

def advanced_waves_to_signals(waves, theory_name="adv_waves"):
    """
    If wave i is 'down' & wave i+1 is 'up' => BUY at wave i end
    If wave i is 'up' & wave i+1 is 'down' => SELL at wave i end
    """
    signals = []
    for i in range(len(waves) - 1):
        w_cur = waves[i]
        w_next = waves[i+1]
        if w_cur["direction"] == "down" and w_next["direction"] == "up":
            signals.append({
                "time": w_cur["end_time"],
                "signal": "buy",
                "price": w_cur["end_price"],
                "theory": theory_name
            })
        elif w_cur["direction"] == "up" and w_next["direction"] == "down":
            signals.append({
                "time": w_cur["end_time"],
                "signal": "sell",
                "price": w_cur["end_price"],
                "theory": theory_name
            })
    return signals

###############################################################################
# Standard Wave Detection (Fallback)
###############################################################################
def detect_waves(
    df, 
    min_wave_bars=1,         # wave must span at least X bars (1 => even single-candle waves)
    min_wave_amplitude=0.1,  # if use_percent=True => 0.1% 
    use_percent=True
):
    """
    Detect wave "turns" in price movement. 
    Lower min_wave_bars => more waves.
    Lower min_wave_amplitude => more waves.
    """
    waves = []
    if len(df) < min_wave_bars:
        return waves

    wave_start_idx = 0
    wave_direction = None
    wave_high = df.loc[0, "close"]
    wave_low = df.loc[0, "close"]

    for i in range(1, len(df)):
        cur_close = df.loc[i, "close"]
        prev_close = df.loc[i - 1, "close"]

        if cur_close > prev_close:
            cur_direction = "up"
        elif cur_close < prev_close:
            cur_direction = "down"
        else:
            cur_direction = wave_direction  # unchanged if equal

        if wave_direction is None:
            wave_direction = cur_direction

        wave_high = max(wave_high, cur_close)
        wave_low = min(wave_low, cur_close)

        # Direction changed => wave ended
        if cur_direction != wave_direction and cur_direction is not None:
            wave_end_idx = i - 1
            wave_len = wave_end_idx - wave_start_idx + 1
            if wave_len >= min_wave_bars:
                # Amplitude check
                amplitude = wave_high - wave_low
                ref_price = wave_low if wave_direction == "up" else wave_high

                if use_percent and ref_price > 0:
                    wave_pct = (amplitude / ref_price) * 100
                    is_valid = wave_pct >= min_wave_amplitude
                else:
                    # Absolute
                    is_valid = amplitude >= min_wave_amplitude

                if is_valid:
                    waves.append({
                        "wave_start_idx": wave_start_idx,
                        "wave_end_idx": wave_end_idx,
                        "direction": wave_direction,
                        "high": float(wave_high),
                        "low": float(wave_low),
                        "start_time": float(df.loc[wave_start_idx, "time"]),
                        "end_time": float(df.loc[wave_end_idx, "time"]),
                        "start_price": float(df.loc[wave_start_idx, "close"]),
                        "end_price": float(df.loc[wave_end_idx, "close"])
                    })

            # Start a new wave
            wave_start_idx = i - 1
            wave_direction = cur_direction
            wave_high = df.loc[wave_start_idx, "close"]
            wave_low = df.loc[wave_start_idx, "close"]

    # Handle final wave
    i = len(df) - 1
    wave_end_idx = i
    wave_len = wave_end_idx - wave_start_idx + 1
    if wave_len >= min_wave_bars:
        amplitude = wave_high - wave_low
        ref_price = wave_low if wave_direction == "up" else wave_high

        if use_percent and ref_price > 0:
            wave_pct = (amplitude / ref_price) * 100
            is_valid = wave_pct >= min_wave_amplitude
        else:
            is_valid = amplitude >= min_wave_amplitude

        if is_valid:
            waves.append({
                "wave_start_idx": wave_start_idx,
                "wave_end_idx": wave_end_idx,
                "direction": wave_direction,
                "high": float(wave_high),
                "low": float(wave_low),
                "start_time": float(df.loc[wave_start_idx, "time"]),
                "end_time": float(df.loc[wave_end_idx, "time"]),
                "start_price": float(df.loc[wave_start_idx, "close"]),
                "end_price": float(df.loc[wave_end_idx, "close"])
            })

    return waves

def wave_to_signals(waves, theory_name="wave"):
    """
    If wave i is 'down' & wave i+1 is 'up' => BUY at wave i end
    If wave i is 'up' & wave i+1 is 'down' => SELL at wave i end
    """
    signals = []
    for i in range(len(waves) - 1):
        w_cur = waves[i]
        w_next = waves[i+1]
        if w_cur["direction"] == "down" and w_next["direction"] == "up":
            signals.append({
                "time": w_cur["end_time"],
                "signal": "buy",
                "price": w_cur["end_price"],
                "theory": theory_name
            })
        elif w_cur["direction"] == "up" and w_next["direction"] == "down":
            signals.append({
                "time": w_cur["end_time"],
                "signal": "sell",
                "price": w_cur["end_price"],
                "theory": theory_name
            })
    return signals
###############################################################################
# Bollinger + Advanced Waves
###############################################################################
###############################################################################
# Example: Bollinger + Daily Splitting + Softer Filters
###############################################################################
def calculate_bollinger_signals(
    historical_candles,
    daily_start_time_utc,    # Now required, no default
    period=20, 
    std_multiplier=2,
    rsi_period=14,
    rsi_buy=30,
    rsi_sell=70,
    # post-processing
    min_bars_gap=0,       # allow signals on consecutive bars if needed
    bar_interval_seconds=60,
    min_profit=0.0        # minimal profit filter => basically off
):
    """
    1) Split 5D data into daily chunks based on custom start time.
    2) For each day, compute Bollinger Bands & RSI.
    3) Generate signals based on Bollinger and RSI conditions.
    4) Post-process signals to filter based on gaps and profitability.

    :param historical_candles: List of candle dictionaries with 'time', 'open', 'high', 'low', 'close'.
    :param daily_start_time_utc: String "HH:MM" indicating when a new day starts in UTC.
    :param period: Window size for Bollinger Bands.
    :param std_multiplier: Standard deviation multiplier for Bollinger Bands.
    :param rsi_period: Period for RSI calculation.
    :param rsi_buy: RSI threshold below which to buy.
    :param rsi_sell: RSI threshold above which to sell.
    :param min_bars_gap: Minimum number of bars between signals.
    :param bar_interval_seconds: Time between bars in seconds.
    :param min_profit: Minimum profit to enforce trade.
    :return: Tuple of (signals list, steps log)
    """
    steps = []
    if not historical_candles:
        return [], steps

    # Convert to DataFrame
    df = pd.DataFrame(historical_candles).sort_values("time").reset_index(drop=True)
    steps.append(f"[Bollinger] Loaded {len(df)} total candles (covering ~5 days).")

    # ~~~~~ 1) Split by day using custom start time
    day_groups = _group_candles_by_day(df, daily_start_time_utc)
    steps.append(f"[Bollinger] Found {len(day_groups)} daily segments.")

    all_signals = []

    # ~~~~~ 2) For each day, run the detection
    for day_idx, day_df in enumerate(day_groups):
        steps.append(f"[Bollinger] Day {day_idx+1}: {len(day_df)} candles.")
        if len(day_df) < period:
            # not enough bars for Bollinger, skip
            steps.append(f"  Skipping day {day_idx+1}, not enough bars for period={period}.")
            continue

        # Bollinger calculations
        day_df["typical_price"] = (day_df["high"] + day_df["low"] + day_df["close"]) / 3
        day_df["ma"] = day_df["typical_price"].rolling(window=period, min_periods=1).mean()
        day_df["std"] = day_df["typical_price"].rolling(window=period, min_periods=1).std()
        day_df["upper"] = day_df["ma"] + (std_multiplier * day_df["std"])
        day_df["lower"] = day_df["ma"] - (std_multiplier * day_df["std"])

        # RSI
        delta = day_df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        roll_up = gain.rolling(window=rsi_period, min_periods=1).mean()
        roll_down = loss.rolling(window=rsi_period, min_periods=1).mean()
        rs = roll_up / (roll_down + 1e-9)
        day_df["rsi"] = 100 - (100 / (1 + rs))

        # *** Simple condition-based signals *** (no complicated wave detection, just a direct approach)
        # For more frequent signals, we can do something like:
        # - If candle crosses below 'lower' band & RSI < 35 => BUY
        # - If candle crosses above 'upper' band & RSI > 65 => SELL
        # This often yields more signals on intraday swings.

        signals = []
        for i in range(1, len(day_df)):
            row_prev = day_df.iloc[i-1]
            row = day_df.iloc[i]

            # Condition for buy
            cross_below = (row_prev["close"] >= row_prev["lower"]) and (row["close"] < row["lower"])
            rsi_condition_buy = (row["rsi"] < rsi_buy)
            if cross_below and rsi_condition_buy:
                signals.append({
                    "time": row["time"],
                    "signal": "buy",
                    "price": row["close"],
                    "theory": "bollinger"
                })

            # Condition for sell
            cross_above = (row_prev["close"] <= row_prev["upper"]) and (row["close"] > row["upper"])
            rsi_condition_sell = (row["rsi"] > rsi_sell)
            if cross_above and rsi_condition_sell:
                signals.append({
                    "time": row["time"],
                    "signal": "sell",
                    "price": row["close"],
                    "theory": "bollinger"
                })

        steps.append(f"  Day {day_idx+1}: Generated {len(signals)} raw signals before filters.")

        # ~~~~~ 3) Post-process signals for this day
        signals = _filter_signals_by_gap(signals, min_bars_gap, bar_interval_seconds)
        signals = _skip_repeated_signals(signals)
        signals = _enforce_profitable_trades(signals, min_profit)
        steps.append(f"  Day {day_idx+1}: {len(signals)} signals after filters.")

        all_signals.extend(signals)

    # ~~~~~ 4) Merge + finalize
    steps.append(f"[Bollinger] TOTAL signals after all days => {len(all_signals)}")

    # Convert time & price to appropriate types
    for s in all_signals:
        s["time"] = int(s["time"])
        s["price"] = float(s["price"])

    return all_signals, steps
###############################################################################
# EMA + Advanced Waves with Daily Reset
###############################################################################
def calculate_ema_signals(
    historical_candles,
    daily_start_time_utc="14:30",  # Custom entry start time in UTC
    # Advanced Wave Toggle
    use_advanced=False,
    zigzag_pct=0.5,
    big_drop_pct=3.0,
    reset_drop_pct=10.0,
    bounce_factor=2.0,
    # Standard EMA Parameters
    short_period=12,
    long_period=26,
    # Post-processing Parameters
    min_bars_gap=0,           # Set to 0 for more signals
    bar_interval_seconds=60,  # 1-minute intervals
    min_profit=0.0            # Set to 0 for no profit restriction
):
    """
    Calculate EMA-based signals with optional advanced wave detection and daily resets.

    :param historical_candles: List of candle dictionaries with 'time', 'open', 'high', 'low', 'close'.
    :param daily_start_time_utc: String "HH:MM" indicating when a new day starts in UTC.
    :param use_advanced: If True, use advanced wave detection logic.
    :param zigzag_pct: ZigZag threshold percentage.
    :param big_drop_pct: Threshold for initial big drop.
    :param reset_drop_pct: Threshold for resetting wave detection.
    :param bounce_factor: Factor to determine significant bounces.
    :param short_period: Short EMA period.
    :param long_period: Long EMA period.
    :param min_bars_gap: Minimum number of bars between signals.
    :param bar_interval_seconds: Time between bars in seconds.
    :param min_profit: Minimum profit to enforce trade.
    :return: Tuple of (signals list, steps log)
    """
    steps = []
    if not historical_candles:
        return [], steps

    # Convert to DataFrame
    df = pd.DataFrame(historical_candles).sort_values("time").reset_index(drop=True)
    steps.append(f"[EMA] Loaded {len(df)} candles.")

    # Calculate EMAs
    df["ema_short"] = df["close"].ewm(span=short_period, adjust=False).mean()
    df["ema_long"] = df["close"].ewm(span=long_period, adjust=False).mean()

    # Split data by day with custom start time
    day_groups = _group_candles_by_day(df, daily_start_time_utc)
    steps.append(f"[EMA] Found {len(day_groups)} daily segments.")

    all_signals = []

    for day_idx, day_df in enumerate(day_groups):
        steps.append(f"[EMA] Day {day_idx+1}: {len(day_df)} candles.")

        if len(day_df) < max(short_period, 5):  # Ensure enough data for EMAs
            steps.append(f"  Skipping Day {day_idx+1}: Not enough candles for EMAs.")
            continue

        # Detect Waves
        if use_advanced:
            waves = detect_advanced_waves(
                day_df,
                zigzag_pct=zigzag_pct,
                big_drop_pct=big_drop_pct,
                reset_drop_pct=reset_drop_pct,
                bounce_factor=bounce_factor
            )
            raw_signals = advanced_waves_to_signals(waves, theory_name="ema_advanced")
            steps.append(f"  [EMA Adv] Detected {len(waves)} waves => {len(raw_signals)} raw signals.")
        else:
            waves = detect_waves(day_df, min_wave_bars=1, min_wave_amplitude=0.1, use_percent=True)
            raw_signals = wave_to_signals(waves, theory_name="ema")
            steps.append(f"  [EMA Basic] Detected {len(waves)} waves => {len(raw_signals)} raw signals.")

        # Confirm signals with EMA cross
        confirmed_signals = []
        for sig in raw_signals:
            t = sig["time"]
            idx_list = day_df.index[day_df["time"] == t].tolist()
            if not idx_list:
                continue
            idx = idx_list[0]
            eshort = day_df.loc[idx, "ema_short"]
            elong = day_df.loc[idx, "ema_long"]

            if sig["signal"] == "buy":
                if eshort > elong:
                    confirmed_signals.append(sig)
            elif sig["signal"] == "sell":
                if eshort < elong:
                    confirmed_signals.append(sig)

        steps.append(f"  [EMA] Day {day_idx+1}: {len(confirmed_signals)} signals after EMA confirmation.")

        # Post-process signals
        confirmed_signals = _filter_signals_by_gap(confirmed_signals, min_bars_gap, bar_interval_seconds)
        confirmed_signals = _skip_repeated_signals(confirmed_signals)
        confirmed_signals = _enforce_profitable_trades(confirmed_signals, min_profit)
        steps.append(f"  [EMA] Day {day_idx+1}: {len(confirmed_signals)} signals after post-processing.")

        all_signals.extend(confirmed_signals)

    steps.append(f"[EMA] Total signals after all days => {len(all_signals)}")

    # Finalize signals
    for s in all_signals:
        s["time"] = int(s["time"])
        s["price"] = float(s["price"])

    return all_signals, steps

###############################################################################
# Fib + Advanced Waves
###############################################################################
###############################################################################
# Fib + Advanced Waves with Daily Reset
###############################################################################
def calculate_fib_signals(
    historical_candles,
    daily_start_time_utc="14:30",  # Custom entry start time in UTC
    # Advanced Wave Toggle
    use_advanced=False,
    zigzag_pct=0.5,
    big_drop_pct=3.0,
    reset_drop_pct=10.0,
    bounce_factor=2.0,
    # Fibonacci Parameters
    fib_levels=(0.236, 0.382, 0.5, 0.618, 0.786),
    # Post-processing Parameters
    min_bars_gap=0,           # Set to 0 for more signals
    bar_interval_seconds=60,  # 1-minute intervals
    min_profit=0.0            # Set to 0 for no profit restriction
):
    """
    Calculate Fibonacci-based signals with optional advanced wave detection and daily resets.

    :param historical_candles: List of candle dictionaries with 'time', 'open', 'high', 'low', 'close'.
    :param daily_start_time_utc: String "HH:MM" indicating when a new day starts in UTC.
    :param use_advanced: If True, use advanced wave detection logic.
    :param zigzag_pct: ZigZag threshold percentage.
    :param big_drop_pct: Threshold for initial big drop.
    :param reset_drop_pct: Threshold for resetting wave detection.
    :param bounce_factor: Factor to determine significant bounces.
    :param fib_levels: Tuple of Fibonacci levels to consider.
    :param min_bars_gap: Minimum number of bars between signals.
    :param bar_interval_seconds: Time between bars in seconds.
    :param min_profit: Minimum profit to enforce trade.
    :return: Tuple of (signals list, steps log)
    """
    steps = []
    if not historical_candles:
        return [], steps

    # Convert to DataFrame
    df = pd.DataFrame(historical_candles).sort_values("time").reset_index(drop=True)
    steps.append(f"[Fib] Loaded {len(df)} candles.")

    # Calculate Fibonacci Levels for the entire period
    day_high = df["high"].max()
    day_low  = df["low"].min()
    fib_map = {}
    for fib in fib_levels:
        fib_map[fib] = day_low + (day_high - day_low) * fib

    # Split data by day with custom start time
    day_groups = _group_candles_by_day(df, daily_start_time_utc)
    steps.append(f"[Fib] Found {len(day_groups)} daily segments.")

    all_signals = []

    for day_idx, day_df in enumerate(day_groups):
        steps.append(f"[Fib] Day {day_idx+1}: {len(day_df)} candles.")

        if len(day_df) < 5:  # Ensure enough data
            steps.append(f"  Skipping Day {day_idx+1}: Not enough candles.")
            continue

        # Detect Waves
        if use_advanced:
            waves = detect_advanced_waves(
                day_df,
                zigzag_pct=zigzag_pct,
                big_drop_pct=big_drop_pct,
                reset_drop_pct=reset_drop_pct,
                bounce_factor=bounce_factor
            )
            raw_signals = advanced_waves_to_signals(waves, theory_name="fib_advanced")
            steps.append(f"  [Fib Adv] Detected {len(waves)} waves => {len(raw_signals)} raw signals.")
        else:
            waves = detect_waves(day_df, min_wave_bars=1, min_wave_amplitude=0.1, use_percent=True)
            raw_signals = wave_to_signals(waves, theory_name="fib")
            steps.append(f"  [Fib Basic] Detected {len(waves)} waves => {len(raw_signals)} raw signals.")

        # Check if wave pivot is near any Fibonacci level
        confirmed_signals = []
        for sig in raw_signals:
            t = sig["time"]
            idx_list = day_df.index[day_df["time"] == t].tolist()
            if not idx_list:
                continue
            idx = idx_list[0]
            pivot_close = day_df.loc[idx, "close"]
            # "Near" any fib level => within ~0.3%
            for fib_val in fib_map.values():
                diff_ratio = abs(pivot_close - fib_val) / max(fib_val, 1e-9)
                if diff_ratio < 0.003:
                    confirmed_signals.append(sig)
                    break

        steps.append(f"  [Fib] Day {day_idx+1}: {len(confirmed_signals)} signals after Fibonacci proximity.")

        # Post-process signals
        confirmed_signals = _filter_signals_by_gap(confirmed_signals, min_bars_gap, bar_interval_seconds)
        confirmed_signals = _skip_repeated_signals(confirmed_signals)
        confirmed_signals = _enforce_profitable_trades(confirmed_signals, min_profit)
        steps.append(f"  [Fib] Day {day_idx+1}: {len(confirmed_signals)} signals after post-processing.")

        all_signals.extend(confirmed_signals)

    steps.append(f"[Fib] Total signals after all days => {len(all_signals)}")

    # Finalize signals
    for s in all_signals:
        s["time"] = int(s["time"])
        s["price"] = float(s["price"])

    return all_signals, steps

###############################################################################
# EWT + Advanced Waves
###############################################################################
###############################################################################
# EWT + Advanced Waves with Daily Reset
###############################################################################
def calculate_ewt_signals(
    historical_candles,
    daily_start_time_utc="14:30",  # Custom entry start time in UTC
    # Advanced Wave Toggle
    use_advanced=False,
    zigzag_pct=0.5,
    big_drop_pct=3.0,
    reset_drop_pct=10.0,
    bounce_factor=2.0,
    # EWT Parameters
    exit_after_bars=10,
    # Post-processing Parameters
    min_bars_gap=0,           # Set to 0 for more signals
    bar_interval_seconds=60,  # 1-minute intervals
    min_profit=0.0            # Set to 0 for no profit restriction
):
    """
    Calculate EWT-based signals with optional advanced wave detection and daily resets.

    :param historical_candles: List of candle dictionaries with 'time', 'open', 'high', 'low', 'close'.
    :param daily_start_time_utc: String "HH:MM" indicating when a new day starts in UTC.
    :param use_advanced: If True, use advanced wave detection logic.
    :param zigzag_pct: ZigZag threshold percentage.
    :param big_drop_pct: Threshold for initial big drop.
    :param reset_drop_pct: Threshold for resetting wave detection.
    :param bounce_factor: Factor to determine significant bounces.
    :param exit_after_bars: Number of bars after which to force exit a trade.
    :param min_bars_gap: Minimum number of bars between signals.
    :param bar_interval_seconds: Time between bars in seconds.
    :param min_profit: Minimum profit to enforce trade.
    :return: Tuple of (signals list, steps log)
    """
    steps = []
    if len(historical_candles) < 5:
        return [], steps

    # Convert to DataFrame
    df = pd.DataFrame(historical_candles).sort_values("time").reset_index(drop=True)
    steps.append(f"[EWT] Loaded {len(df)} candles.")

    # Split data by day with custom start time
    day_groups = _group_candles_by_day(df, daily_start_time_utc)
    steps.append(f"[EWT] Found {len(day_groups)} daily segments.")

    all_signals = []

    for day_idx, day_df in enumerate(day_groups):
        steps.append(f"[EWT] Day {day_idx+1}: {len(day_df)} candles.")

        if len(day_df) < 5:
            # Skip if too few bars
            steps.append(f"  Skipping Day {day_idx+1}, not enough bars.")
            continue

        # Detect Waves
        if use_advanced:
            waves = detect_advanced_waves(
                day_df,
                zigzag_pct=zigzag_pct,
                big_drop_pct=big_drop_pct,
                reset_drop_pct=reset_drop_pct,
                bounce_factor=bounce_factor
            )
            raw_signals = advanced_waves_to_signals(waves, theory_name="ewt_advanced")
            steps.append(f"  [EWT Adv] Detected {len(waves)} waves => {len(raw_signals)} raw signals.")
        else:
            waves = detect_ewt_waves(day_df, min_wave_bars=1, min_wave_amplitude=0.1, use_percent=True)
            raw_signals = wave_to_ewt_signals(waves)
            steps.append(f"  [EWT Basic] Detected {len(waves)} waves => {len(raw_signals)} raw signals.")

        # Implement Time-based Exit
        signals = []
        in_trade = None
        bars_in_trade = 0

        pivot_map = { s["time"]: s for s in raw_signals }

        for i in range(len(day_df)):
            t = day_df.loc[i, "time"]
            c = day_df.loc[i, "close"]
            if t in pivot_map:
                pivot_signal = pivot_map[t]
                if in_trade is None:
                    # Open trade
                    signals.append({
                        "time": t,
                        "signal": pivot_signal["signal"],
                        "price": c,
                        "theory": pivot_signal.get("theory", "ewt")
                    })
                    in_trade = pivot_signal["signal"]
                    bars_in_trade = 0
                    steps.append(
                        f"  [EWT] Pivot {pivot_signal['signal'].upper()} at t={t} => {c:.2f}"
                    )

            if in_trade is not None:
                bars_in_trade += 1
                if bars_in_trade >= exit_after_bars:
                    # Forced exit
                    exit_signal = "sell" if in_trade == "buy" else "buy"
                    signals.append({
                        "time": t,
                        "signal": exit_signal,
                        "price": c,
                        "theory": "ewt"
                    })
                    steps.append(
                        f"  [EWT] TIME-EXIT => closed {in_trade} at t={t} => {c:.2f}"
                    )
                    in_trade = None
                    bars_in_trade = 0

        steps.append(f"  [EWT] Day {day_idx+1}: {len(signals)} signals after time-based exit.")

        # Post-process signals
        signals = _filter_signals_by_gap(signals, min_bars_gap, bar_interval_seconds)
        signals = _skip_repeated_signals(signals)
        signals = _enforce_profitable_trades(signals, min_profit)
        steps.append(f"  [EWT] Day {day_idx+1}: {len(signals)} signals after post-processing.")

        all_signals.extend(signals)

    steps.append(f"[EWT] Total signals after all days => {len(all_signals)}")

    # Finalize signals
    for s in all_signals:
        s["time"] = int(s["time"])
        s["price"] = float(s["price"])

    return all_signals, steps

###############################################################################
# Standard EWT Wave Detection (Fallback)
###############################################################################
def detect_ewt_waves(
    df,
    min_wave_bars=1,
    min_wave_amplitude=0.1,
    use_percent=True
):
    """
    Similar to detect_waves but specifically for EWT naming. 
    We'll keep it separate if you want to tweak EWT differently.
    """
    waves = []
    if len(df) < min_wave_bars:
        return waves

    wave_start_idx = 0
    wave_direction = None
    wave_high = df.loc[0, "close"]
    wave_low = df.loc[0, "close"]

    for i in range(1, len(df)):
        cur_close = df.loc[i, "close"]
        prev_close = df.loc[i - 1, "close"]

        if cur_close > prev_close:
            cur_direction = "up"
        elif cur_close < prev_close:
            cur_direction = "down"
        else:
            cur_direction = wave_direction

        if wave_direction is None:
            wave_direction = cur_direction

        wave_high = max(wave_high, cur_close)
        wave_low = min(wave_low, cur_close)

        # Direction changed => wave ended
        if cur_direction != wave_direction and cur_direction is not None:
            wave_end_idx = i - 1
            wave_len = wave_end_idx - wave_start_idx + 1
            if wave_len >= min_wave_bars:
                amplitude = wave_high - wave_low
                ref_price = wave_low if wave_direction == "up" else wave_high
                if use_percent and ref_price > 0:
                    wave_pct = (amplitude / ref_price) * 100
                    is_valid = wave_pct >= min_wave_amplitude
                else:
                    is_valid = amplitude >= min_wave_amplitude
                if is_valid:
                    waves.append({
                        "wave_start_idx": wave_start_idx,
                        "wave_end_idx": wave_end_idx,
                        "direction": wave_direction,
                        "high": float(wave_high),
                        "low": float(wave_low),
                        "start_time": float(df.loc[wave_start_idx, "time"]),
                        "end_time": float(df.loc[wave_end_idx, "time"]),
                        "start_price": float(df.loc[wave_start_idx, "close"]),
                        "end_price": float(df.loc[wave_end_idx, "close"])
                    })

            # Start a new wave
            wave_start_idx = i - 1
            wave_direction = cur_direction
            wave_high = df.loc[wave_start_idx, "close"]
            wave_low = df.loc[wave_start_idx, "close"]

    # Handle final wave
    i = len(df) - 1
    wave_end_idx = i
    wave_len = wave_end_idx - wave_start_idx + 1
    if wave_len >= min_wave_bars:
        amplitude = wave_high - wave_low
        ref_price = wave_low if wave_direction == "up" else wave_high
        if use_percent and ref_price > 0:
            wave_pct = (amplitude / ref_price) * 100
            is_valid = wave_pct >= min_wave_amplitude
        else:
            is_valid = amplitude >= min_wave_amplitude
        if is_valid:
            waves.append({
                "wave_start_idx": wave_start_idx,
                "wave_end_idx": wave_end_idx,
                "direction": wave_direction,
                "high": float(wave_high),
                "low": float(wave_low),
                "start_time": float(df.loc[wave_start_idx, "time"]),
                "end_time": float(df.loc[wave_end_idx, "time"]),
                "start_price": float(df.loc[wave_start_idx, "close"]),
                "end_price": float(df.loc[wave_end_idx, "close"])
            })

    return waves

def wave_to_ewt_signals(waves):
    """
    If wave i is 'down' & wave i+1 is 'up' => BUY at wave i end
    If wave i is 'up'   & wave i+1 is 'down' => SELL at wave i end
    """
    signals = []
    for i in range(len(waves) - 1):
        w_cur = waves[i]
        w_next = waves[i+1]
        if w_cur["direction"] == "down" and w_next["direction"] == "up":
            signals.append({
                "time": w_cur["end_time"],
                "signal": "buy",
                "price": w_cur["end_price"],
                "theory": "ewt"
            })
        elif w_cur["direction"] == "up" and w_next["direction"] == "down":
            signals.append({
                "time": w_cur["end_time"],
                "signal": "sell",
                "price": w_cur["end_price"],
                "theory": "ewt"
            })
    return signals
###############################################################################
# Support Flow 
###############################################################################
def calc_support_put_call_flow(
    historical_candles,
    daily_start_time_utc="14:30",  # Custom entry start time in UTC
    support_initial=None,
    bar_interval_seconds=60,
    use_percent=False,
    support_gap=5.0,
    min_bars_gap=0,           # Set to 0 for more signals
    min_profit=0.0            # Set to 0 for no profit restriction
):
    """
    Calculate Support Flow signals by toggling between 'sell' and 'buy' based on price crossings.
    Resets calculations each day based on the specified daily start time.

    :param historical_candles: List of candle dictionaries with 'time', 'open', 'high', 'low', 'close'.
    :param daily_start_time_utc: String "HH:MM" indicating when a new day starts in UTC.
    :param support_initial: Optional initial support level.
    :param bar_interval_seconds: Time between bars in seconds (e.g., 60 for 1-minute intervals).
    :param use_percent: If True, treat support_gap as a percentage; else as absolute value.
    :param support_gap: Gap above support to trigger 'sell' signals.
    :param min_bars_gap: Minimum number of bars between signals to prevent clustering.
    :param min_profit: Minimum price movement to consider a trade profitable.
    :return: Tuple of (signals list, steps log)
    """
    steps = []
    signals = []
    if not historical_candles:
        return signals, steps

    # Convert to DataFrame
    df = pd.DataFrame(historical_candles).sort_values("time").reset_index(drop=True)
    steps.append(f"SupportFlow: Loaded {len(df)} candles.")

    # Split data by day with custom start time
    day_groups = _group_candles_by_day(df, daily_start_time_utc)
    steps.append(f"SupportFlow: Found {len(day_groups)} daily segments.")

    for day_idx, day_df in enumerate(day_groups):
        steps.append(f"Day {day_idx+1}: {len(day_df)} candles in SupportFlow.")

        if len(day_df) < 10:
            # Skip if too few bars
            steps.append(f"  Skipping Day {day_idx+1}, not enough bars.")
            continue

        # Initialize support for the day
        if support_initial is not None:
            support = float(support_initial)
            steps.append(f"  Using user-defined support => {support:.2f}")
        else:
            # day-specific min
            support = day_df["close"].rolling(window=60, min_periods=1).min().iloc[0]
            steps.append(f"  Day {day_idx+1} initial support => {support:.2f}")

        current_state = None
        day_signals = []

        for i, row in day_df.iterrows():
            cur_time = row["time"]
            cur_close = row["close"]

            # Update rolling min within the day
            rolling_min = day_df["close"].rolling(window=60, min_periods=1).min().iloc[i]
            support = max(rolling_min, support)

            # Calculate threshold
            if use_percent:
                threshold = support * (1 + support_gap / 100.0)
            else:
                threshold = support + support_gap

            # Determine state based on current price
            if cur_close > threshold:
                state = "sell"
            elif cur_close < support:
                state = "buy"
            else:
                state = "hold"

            # Initialize state based on the first relevant signal
            if current_state is None:
                if state in ["sell", "buy"]:
                    current_state = state
                    day_signals.append({
                        "time": cur_time,
                        "signal": state,
                        "price": float(cur_close),
                        "theory": "support_flow"
                    })
                    steps.append(f"  SupportFlow: Initial {state.upper()} at t={cur_time} => {cur_close:.2f}")
                # Remain None if state is 'hold'
                continue

            # State transition
            if state != current_state:
                if state in ["sell", "buy"]:
                    # Transition to a new active state
                    day_signals.append({
                        "time": cur_time,
                        "signal": state,
                        "price": float(cur_close),
                        "theory": "support_flow"
                    })
                    steps.append(f"  SupportFlow: {state.upper()} signal at t={cur_time} => {cur_close:.2f}")
                    current_state = state
                elif state == "hold":
                    # Transition to 'hold' from 'sell' or 'buy'
                    day_signals.append({
                        "time": cur_time,
                        "signal": f"end_{current_state}",
                        "price": float(cur_close),
                        "theory": "support_flow"
                    })
                    steps.append(f"  SupportFlow: End {current_state.upper()} => t={cur_time} => {cur_close:.2f}")
                    current_state = state

        # End of day: Optionally add an 'end' signal if in 'sell' or 'buy' state
        if current_state in ["sell", "buy"]:
            last_time = day_df.iloc[-1]["time"]
            last_close = day_df.iloc[-1]["close"]
            day_signals.append({
                "time": last_time,
                "signal": f"end_{current_state}",
                "price": float(last_close),
                "theory": "support_flow"
            })
            steps.append(f"  SupportFlow: End {current_state.upper()} => t={last_time} => {last_close:.2f}")

        # Post-process daily signals
        day_signals = _filter_signals_by_gap(day_signals, min_bars_gap, bar_interval_seconds)
        day_signals = _skip_repeated_signals(day_signals)
        day_signals = _enforce_profitable_trades(day_signals, min_profit)
        steps.append(f"  Day {day_idx+1}: {len(day_signals)} signals after post-processing.")
        signals.extend(day_signals)

    steps.append(f"SupportFlow: Total signals after all days => {len(signals)}")

    # Finalize signals
    for s in signals:
        s["time"] = int(s["time"])
        s["price"] = float(s["price"])

    return signals, steps
