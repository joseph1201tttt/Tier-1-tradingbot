
import os
import requests
import pandas as pd
import numpy as np
import sqlite3
import joblib
from datetime import datetime, time
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

TOKEN = "8449970757:AAFKmZSvQ_ka3hnrgCjQzDkIphkvT34Yyu4"
TWELVE_DATA_API_KEY = "33aec99f37d24aab8428cf43d5e58f8b"
PAIRS = ["XAU/USD", "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
TIMEFRAMES = ["5min", "15min", "1h"]
MODEL_FILE = "tier1_model.pkl"
DB_FILE = "performance.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS trades(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        timeframe TEXT,
        direction TEXT,
        entry REAL,
        tp1 REAL,
        tp2 REAL,
        sl REAL,
        result INTEGER,
        timestamp TEXT
    )""")
    conn.commit()
    conn.close()

def fetch_data(symbol, interval, outputsize=800):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={TWELVE_DATA_API_KEY}"
    r = requests.get(url).json()
    if "values" not in r:
        return None
    df = pd.DataFrame(r["values"])
    df = df.astype(float)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    return df

def add_features(df):
    df["ema20"] = EMAIndicator(df["close"], 20).ema_indicator()
    df["ema50"] = EMAIndicator(df["close"], 50).ema_indicator()
    df["ema200"] = EMAIndicator(df["close"], 200).ema_indicator()
    df["rsi"] = RSIIndicator(df["close"], 14).rsi()
    df["adx"] = ADXIndicator(df["high"], df["low"], df["close"], 14).adx()
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()
    df["ema_slope"] = df["ema50"].diff()
    df["rsi_slope"] = df["rsi"].diff()
    df["body"] = abs(df["close"] - df["open"])
    df["wick"] = (df["high"] - df["low"]) - df["body"]
    df["volatility"] = df["atr"] / df["close"]
    df["return"] = df["close"].pct_change()
    df["target"] = np.where(df["return"].shift(-1) > 0, 1, 0)
    df = df.dropna()
    return df

def train_model(df):
    features = ["ema20","ema50","ema200","rsi","adx","atr","ema_slope","rsi_slope","body","wick","volatility"]
    X = df[features]
    y = df["target"]
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier())
    ])
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    return model

def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return None

def session_filter():
    now = datetime.utcnow().time()
    london = time(7,0) <= now <= time(16,0)
    newyork = time(12,0) <= now <= time(21,0)
    return london or newyork

def pip_value(symbol):
    if "JPY" in symbol:
        return 0.01
    if "XAU" in symbol:
        return 0.1
    return 0.0001

def calculate_levels(symbol, direction, entry):
    pip = pip_value(symbol)
    tp1_pips = 100 * pip
    tp2_pips = 250 * pip
    if direction == "BUY":
        return entry + tp1_pips, entry + tp2_pips
    else:
        return entry - tp1_pips, entry - tp2_pips

def backtest(df, model):
    features = ["ema20","ema50","ema200","rsi","adx","atr","ema_slope","rsi_slope","body","wick","volatility"]
    X = df[features]
    preds = model.predict(X)
    return (preds == df["target"]).mean()

def analyze(symbol):
    if not session_filter():
        return "Outside trading session"
    model = load_model()
    results = {}
    for tf in TIMEFRAMES:
        df = fetch_data(symbol, tf)
        if df is None:
            continue
        df = add_features(df)
        if model is None:
            model = train_model(df)
        accuracy = backtest(df, model)
        latest = df.iloc[-1]
        X = df[["ema20","ema50","ema200","rsi","adx","atr","ema_slope","rsi_slope","body","wick","volatility"]].iloc[-1:]
        prob = model.predict_proba(X)[0][1]
        trend_buy = latest["ema50"] > latest["ema200"] and latest["adx"] > 25
        trend_sell = latest["ema50"] < latest["ema200"] and latest["adx"] > 25
        direction = "BUY" if prob > 0.55 and trend_buy else "SELL" if prob < 0.45 and trend_sell else "NONE"
        entry = latest["close"]
        tp1, tp2 = calculate_levels(symbol, direction, entry) if direction != "NONE" else (0,0)
        sl = entry - latest["atr"] if direction=="BUY" else entry + latest["atr"] if direction=="SELL" else 0
        confidence = round(prob * 100,2)
        results[tf] = {
            "direction": direction,
            "entry": round(entry,5),
            "tp1": round(tp1,5),
            "tp2": round(tp2,5),
            "sl": round(sl,5),
            "confidence": confidence,
            "accuracy": round(accuracy*100,2)
        }
    return results

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton(p, callback_data=p)] for p in PAIRS]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Select Pair:", reply_markup=reply_markup)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    symbol = query.data
    results = analyze(symbol)
    if isinstance(results, str):
        await query.message.reply_text(results)
        return
    message = f"{symbol} Tier 1 Analysis\n"
    for tf, data in results.items():
        message += f"\n{tf}\nDirection: {data['direction']}\nEntry: {data['entry']}\nTP1: {data['tp1']}\nTP2: {data['tp2']}\nSL: {data['sl']}\nConfidence: {data['confidence']}%\nModel Accuracy: {data['accuracy']}%\n"
    keyboard = [[InlineKeyboardButton(p, callback_data=p)] for p in PAIRS]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.message.reply_text(message, reply_markup=reply_markup)

def main():
    init_db()
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.run_polling()

if __name__ == "__main__":
    main()
