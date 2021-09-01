"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import datetime
import pandas as pd
import yfinance as yf
import numpy as np


def get_yfinance_close_2(ticker, date):
    a = datetime.datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
    d = yf.Ticker(ticker).history(period="max")
    return d["Close"].loc[a]


def get_yfinance_close(ticker, date):
    a = datetime.datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
    b1 = datetime.datetime.strptime(date, '%Y%m%d') - datetime.timedelta(days=3)
    b2 = datetime.datetime.strptime(date, '%Y%m%d') + datetime.timedelta(days=3)
    d = yf.download(ticker, start=b1.strftime('%Y-%m-%d'), end=b2.strftime('%Y-%m-%d'))
    if ticker == "ORBIA.MX" and date == "20190930":
        return 39.10
    else:
        return d["Close"].loc[a]


def get_tickers(df):
    t = df["Ticker"].unique().tolist()
    t = [i.replace("*", "").replace("MEXCHEM", "ORBIA").replace("GFREGIOO", "RA").replace(".", "-")
         + ".MX" for i in t]
    w = dict(zip(t, df["Peso (%)"]))
    return t, w


def shares_number(capital, weight, ticker, date):
    price = get_yfinance_close(ticker, date)
    shares = capital * weight / price
    return int(shares)


def init_port(capital, weights, tickers, date, restricted, comission):
    port = pd.DataFrame(data=np.zeros((3, len(tickers))), columns=tickers, index=["Weight", "Shares", "Value"])
    for i in tickers:
        weight = weights[i]
        if i not in restricted:
            s = shares_number(capital, weight, i, date)
            v = s * get_yfinance_close(i, date)
            port[i] = [0, s, v]

    sobrante = capital - sum(port.iloc[[2]].values[0])*(1+comission)
    port["MXN.MX"] = [0, sobrante, sobrante]
    w = [i/sum(port.iloc[[2]].values[0]) for i in port.iloc[[2]].values[0]]
    final_df = port.T
    final_df["Weight"] = w
    return final_df


def reval_port(port, date, restricted):
    port_r = port
    for i in port_r.index:
        if i not in restricted:
            v = port_r.loc[i, "Shares"] * get_yfinance_close(i, date)
            port_r.loc[i, "Value"] = v
    w = [i / sum(port_r.loc[:, "Value"]) for i in port_r.loc[:, "Value"]]
    port_r.loc[:, "Weight"] = w
    return port_r


def create_df_pasiva(dates, portafolios_valor):
    df_pasiva_a = pd.DataFrame([datetime.datetime.strptime(i, '%Y%m%d') for i in dates], columns=["timestamp"])
    df_pasiva_a["capital"] = portafolios_valor
    df_pasiva_a["rend"] = df_pasiva_a["capital"].pct_change()
    df_pasiva_a["rend_acum"] = df_pasiva_a["rend"].cumsum()
    return df_pasiva_a
