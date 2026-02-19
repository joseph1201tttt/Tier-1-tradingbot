import logging
import numpy as np
import pandas as pd
import requests
import ta
import joblib
import os
import sqlite3
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler

TOKEN ="8449970757:AAFKmZSvQ_ka3hnrgCjQzDkIphkvT34Yyu4"
TWELVE_DATA_API_KEY ="33aec99f37d24aab8428cf43d5e58f8b"

PAIRS = {
    "XAUUSD": "XAU/USD",
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "USDCAD": "USD/CAD"
}

TIMEFRAMES = {
    "5m": "5min",
    "15m": "15min"
}

MODEL_FILE = "ml_model.pkl"
SCALER_FILE = "scaler.pkl"
DB_FILE = "learning.db"

logging.basicConfig(level=logging.INFO)

if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
else:
    model = SGDClassifier(loss="log_loss")
    scaler = StandardScaler()

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS trades(features TEXT,target INTEGER)")
    conn.commit()
    conn.close()

def get_data(symbol, interval):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol,"interval": interval,"outputsize": 400,"apikey": TWELVE_DATA_API_KEY}
    r = requests.get(url, params=params).json()
    if "values" not in r:
        return pd.DataFrame()
    df = pd.DataFrame(r["values"]).iloc[::-1]
    for col in ["open","high","low","close"]:
        df[col]=df[col].astype(float)
    df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close"},inplace=True)
    df.dropna(inplace=True)
    return df

def indicators(df):
    df["EMA20"]=ta.trend.ema_indicator(df["Close"],20)
    df["EMA50"]=ta.trend.ema_indicator(df["Close"],50)
    df["EMA200"]=ta.trend.ema_indicator(df["Close"],200)
    df["RSI"]=ta.momentum.rsi(df["Close"],14)
    macd=ta.trend.MACD(df["Close"])
    df["MACD"]=macd.macd()
    df["MACD_signal"]=macd.macd_signal()
    df["ATR"]=ta.volatility.average_true_range(df["High"],df["Low"],df["Close"],14)
    df["ADX"]=ta.trend.adx(df["High"],df["Low"],df["Close"],14)
    df["CCI"]=ta.trend.cci(df["High"],df["Low"],df["Close"],14)
    df["STOCH"]=ta.momentum.stoch(df["High"],df["Low"],df["Close"])
    df.dropna(inplace=True)
    return df

def market_structure(df):
    hh=df["High"].iloc[-1]>df["High"].iloc[-2]
    hl=df["Low"].iloc[-1]>df["Low"].iloc[-2]
    lh=df["High"].iloc[-1]<df["High"].iloc[-2]
    ll=df["Low"].iloc[-1]<df["Low"].iloc[-2]
    if hh and hl:return 1
    if lh and ll:return -1
    return 0

def bos(df):
    return 1 if df["Close"].iloc[-1]>df["High"].iloc[-3] else -1 if df["Close"].iloc[-1]<df["Low"].iloc[-3] else 0

def engulfing(df):
    prev=df.iloc[-2];cur=df.iloc[-1]
    if cur["Close"]>cur["Open"] and prev["Close"]<prev["Open"] and cur["Close"]>prev["Open"] and cur["Open"]<prev["Close"]:return 1
    if cur["Close"]<cur["Open"] and prev["Close"]>prev["Open"] and cur["Open"]>prev["Close"] and cur["Close"]<prev["Open"]:return -1
    return 0

def liquidity(df):
    return 1 if df["High"].iloc[-1]>df["High"].iloc[-5] else -1 if df["Low"].iloc[-1]<df["Low"].iloc[-5] else 0

def sr(df):
    return df["Low"].tail(20).min(),df["High"].tail(20).max()

def features(df):
    s=market_structure(df)
    b=bos(df)
    e=engulfing(df)
    l=liquidity(df)
    last=df.iloc[-1]
    return np.array([[last["Close"],last["EMA20"],last["EMA50"],last["EMA200"],last["RSI"],last["MACD"],last["MACD_signal"],last["ATR"],last["ADX"],last["CCI"],last["STOCH"],s,b,e,l]])

def init_model(x):
    if not hasattr(model,"classes_"):
        scaler.fit(x)
        model.partial_fit(scaler.transform(x),[1],classes=[0,1])

def pips(symbol):
    if "JPY" in symbol:return 0.01
    if "XAU" in symbol:return 0.1
    return 0.0001

def tp_levels(symbol,entry,dir):
    p=pips(symbol)
    if dir=="BUY":
        return entry+60*p,entry+120*p,entry+300*p
    else:
        return entry-60*p,entry-120*p,entry-300*p

def analyze(df,symbol):
    df=indicators(df)
    f=features(df)
    init_model(f)
    prob=model.predict_proba(scaler.transform(f))[0]
    direction="BUY" if prob[1]>0.55 else "SELL"
    entry=df["Close"].iloc[-1]
    tp1,tp2,tp3=tp_levels(symbol,entry,direction)
    sl=entry-df["ATR"].iloc[-1] if direction=="BUY" else entry+df["ATR"].iloc[-1]
    return direction,entry,sl,tp1,tp2,tp3,f

def update(df,f):
    future=df["Close"].iloc[-1];cur=df["Close"].iloc[-2]
    t=1 if future>cur else 0
    scaler.partial_fit(f)
    model.partial_fit(scaler.transform(f),[t])
    joblib.dump(model,MODEL_FILE)
    joblib.dump(scaler,SCALER_FILE)

def backtest(df):
    df=indicators(df)
    X=[];y=[]
    for i in range(5,len(df)-1):
        sub=df.iloc[:i]
        f=features(sub)
        X.append(f[0])
        y.append(1 if df["Close"].iloc[i+1]>df["Close"].iloc[i] else 0)
    if len(X)>10:
        scaler.partial_fit(X)
        model.partial_fit(scaler.transform(X),y)
        joblib.dump(model,MODEL_FILE)
        joblib.dump(scaler,SCALER_FILE)

async def start(update:Update,context:ContextTypes.DEFAULT_TYPE):
    kb=[[InlineKeyboardButton(p,callback_data=p)] for p in PAIRS]
    await update.message.reply_text("Select Pair:",reply_markup=InlineKeyboardMarkup(kb))

async def button(update:Update,context:ContextTypes.DEFAULT_TYPE):
    q=update.callback_query
    await q.answer()
    pair=q.data
    symbol=PAIRS[pair]
    tb=0;ts=0;res=None;f=None;df_last=None
    for tf in TIMEFRAMES.values():
        df=get_data(symbol,tf)
        if df.empty or len(df)<50:continue
        backtest(df)
        d,e,sl,tp1,tp2,tp3,feat=analyze(df,symbol)
        if d=="BUY":tb+=1
        else:ts+=1
        res=(d,e,sl,tp1,tp2,tp3)
        f=feat
        df_last=df
    if not res:
        await q.edit_message_text("No data")
        return
    direction=res[0]
    entry=res[1]
    sl=res[2]
    tp1,tp2,tp3=res[3],res[4],res[5]
    update(df_last,f)
    text=f"{pair}\n{direction}\nEntry:{entry:.5f}\nSL:{sl:.5f}\nTP1:{tp1:.5f}\nTP2:{tp2:.5f}\nTP3:{tp3:.5f}"
    await q.edit_message_text(text)

def main():
    init_db()
    app=ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start",start))
    app.add_handler(CallbackQueryHandler(button))
    app.run_polling()

if __name__=="__main__":
    main()
