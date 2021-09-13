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
from scipy.optimize import minimize


def get_yfinance_close_aggregate(ticker_list, dates_list, value):
    a = datetime.datetime.strptime(dates_list[0], '%Y%m%d').strftime('%Y-%m-%d')
    b2 = datetime.datetime.strptime(dates_list[-1], '%Y%m%d') + datetime.timedelta(days=3)
    b2 = b2.strftime('%Y-%m-%d')
    d = yf.download(ticker_list, start=a, end=b2)
    d = d[value]
    return d


def get_yfinance_close(ticker, date, aggregate):
    a = datetime.datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
    price = aggregate.loc[a, ticker]
    if ticker == "ORBIA.MX" and date == "20190930":
        return 39.10
    else:
        return price


def get_tickers(df):
    t = df["Ticker"].unique().tolist()
    t = [i.replace("*", "").replace("MEXCHEM", "ORBIA").replace("GFREGIOO", "RA").replace(".", "-")
         + ".MX" for i in t]
    w = dict(zip(t, df["Peso (%)"]))
    return t, w


def shares_number(capital, weight, ticker, date, aggregate):
    price = get_yfinance_close(ticker, date, aggregate)
    shares = capital * weight / price
    return int(shares)


def init_port(capital, weights, tickers, date, restricted, comission, aggregate):
    port = pd.DataFrame(data=np.zeros((3, len(tickers))), columns=tickers, index=["Weight", "Shares", "Value"])
    comission_headers = ["timestamp", "titulos_totales", "titulos_mod", "precio", "comisión",
                         "comision_acum"]
    comission_df = pd.DataFrame(data=np.zeros((len(comission_headers), len(tickers))),
                                columns=tickers, index=comission_headers)

    for i in tickers:
        weight = weights[i]
        if i not in restricted:
            s = shares_number(capital, weight, i, date, aggregate)
            v = s * get_yfinance_close(i, date, aggregate)
            port[i] = [0, s, v]
            comission_df[i] = [date, s, s, get_yfinance_close(i, date, aggregate), v*comission, v*comission]
        if i in restricted:
            comission_df.drop(columns=[i])

    sobrante = capital - sum(port.iloc[[2]].values[0])*(1+comission)
    port["MXN.MX"] = [0, sobrante, sobrante]
    w = [i/sum(port.iloc[[2]].values[0]) for i in port.iloc[[2]].values[0]]
    final_df = port.T
    final_df["Weight"] = w

    comission_df = comission_df.T
    comission_df = comission_df[comission_df['comisión'] != 0]
    comission_df.reset_index(level=0, inplace=True)

    return final_df, comission_df


def reval_port(port, date, restricted, aggregate):
    port_r = port.copy()
    for i in port_r.index:
        if i not in restricted:
            v = port_r.loc[i, "Shares"] * get_yfinance_close(i, date, aggregate)
            port_r.loc[i, "Value"] = v
    w = [i / sum(port_r.loc[:, "Value"]) for i in port_r.loc[:, "Value"]]
    port_r.loc[:, "Weight"] = w
    return port_r


def create_df_pasiva(dates, portafolios_valor):
    df_pasiva_a = pd.DataFrame([datetime.datetime.strptime(i, '%Y%m%d') for i in dates], columns=["timestamp"])
    df_pasiva_a["capital"] = portafolios_valor
    df_pasiva_a["rend"] = df_pasiva_a["capital"].pct_change()
    df_pasiva_a["rend_acum"] = (df_pasiva_a["capital"] / df_pasiva_a["capital"][0]) - 1
    return df_pasiva_a


def get_annual_summ(tickers, restricted, aggregate, d_slice):

    t = [x for x in tickers if x not in restricted]
    annual_ret_summ = pd.DataFrame(columns=t, index=['Media', 'Volatilidad'])
    for i in t:
        prices_sliced = aggregate.loc[d_slice[0]:d_slice[1], i]
        log_r = np.log(prices_sliced / prices_sliced.shift(1)).dropna()
        mean = log_r.mean() * 252
        vol = log_r.std() * np.sqrt(252)
        annual_ret_summ[i] = [mean, vol]

    annual_ret_summ["MXN.MX"] = [0, 0]
    return annual_ret_summ


def corr_matrix(tickers, restricted, aggregate, d_slice):
    t = [x for x in tickers if x not in restricted]
    prices_sliced = aggregate.loc[d_slice[0]:d_slice[1], :]
    pandas_df = prices_sliced[prices_sliced.columns.intersection(t)].pct_change().apply(lambda x: np.log(1+x))
    pandas_df = pandas_df.reindex(t, axis=1)
    return pandas_df.corr()


def minus_SR(w, Sigma, Eind, rf):
    sp = (w.T.dot(Sigma).dot(w))**0.5
    Ep = Eind.T.dot(w)
    SR = (Ep - rf) / sp
    return -SR


def varianza(w, Sigma):
    return w.T.dot(Sigma).dot(w)


def min_sr_port(annual_ret_summ, corr, rf):
    annual_ret_summ = annual_ret_summ.iloc[:, :-1]
    S = np.diag(annual_ret_summ.loc['Volatilidad', :].values)
    Sigma = S.dot(corr).dot(S)
    Eind = annual_ret_summ.loc['Media', :].values
    n = len(Eind)
    w0 = np.ones(n) / n
    bnds = ((0, 1),) * n
    cons = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
    EMV = minimize(fun=minus_SR, x0=w0, args=(Sigma, Eind, rf), bounds=bnds, constraints=cons)
    w_EMV = EMV.x
    E_EMV = Eind.T.dot(w_EMV)
    s_EMV = (w_EMV.T.dot(Sigma).dot(w_EMV)) ** 0.5
    SR_EMV = -minus_SR(w_EMV, Sigma, annual_ret_summ.loc["Media"], rf)
    return EMV, w_EMV, E_EMV, SR_EMV, s_EMV, Sigma


def min_var_port(annual_ret_summ, corr):
    annual_ret_summ = annual_ret_summ.iloc[:, :-1]
    S = np.diag(annual_ret_summ.loc['Volatilidad', :].values)
    Sigma = S.dot(corr).dot(S)
    Eind = annual_ret_summ.loc['Media', :].values
    n = len(Eind)
    w0 = np.ones(n) / n
    bnds = ((0, 1),) * n
    cons = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
    minvar = minimize(fun=varianza, x0=w0, args=(Sigma,), bounds=bnds, constraints=cons)
    w_minvar = minvar.x
    E_minvar = Eind.T.dot(w_minvar)
    s_minvar = (w_minvar.T.dot(Sigma).dot(w_minvar)) ** 0.5
    return minvar, w_minvar, E_minvar, s_minvar


def init_port_act(capital, weights, tickers, date, restricted, comission, aggregate):
    port = pd.DataFrame(data=np.zeros((3, len(tickers))), columns=tickers, index=["Weight", "Shares", "Value"])
    comission_headers = ["timestamp", "titulos_totales", "titulos_mod", "precio", "comisión",
                         "comision_acum"]
    comission_df = pd.DataFrame(data=np.zeros((len(comission_headers), len(tickers))),
                                columns=tickers, index=comission_headers)

    for i in tickers:
        weight = weights[i]
        if i not in restricted:
            s = shares_number(capital*(1-comission), weight, i, date, aggregate)
            v = s * get_yfinance_close(i, date, aggregate)
            port[i] = [0, s, v]
            comission_df[i] = [date, s, s, get_yfinance_close(i, date, aggregate), v*comission, v*comission]
        if i in restricted:
            comission_df.drop(columns=[i])

    sobrante = capital - sum(port.iloc[[2]].values[0])*(1+comission)
    port["MXN.MX"] = [0, sobrante, sobrante]
    w = [i/sum(port.iloc[[2]].values[0]) for i in port.iloc[[2]].values[0]]
    final_df = port.T
    final_df["Weight"] = w

    comission_df = comission_df.T
    comission_df = comission_df[comission_df['comisión'] != 0]
    comission_df.reset_index(level=0, inplace=True)
    final_df = final_df[final_df['Shares'] !=0]

    return final_df, comission_df


def rebalance_port(port_activa, aggregate, dates, comission, comission_df_activa):
    operaciones = comission_df_activa.copy()
    portact_d = {}
    portact_d[dates[0]] = port_activa.copy()
    for i in range(1, len(dates)):
        portact_d[dates[i]] = portact_d[dates[i - 1]].copy()
        ts = port_activa.index.to_list()
        del ts[-1]
        rend_list = pd.DataFrame(data=[0, 0]).T
        rend_list.columns = ts
        rend_list.index = ["rend"]

        for j in range(0, len(ts)):
            precio_h = get_yfinance_close(ts[j], dates[i], aggregate)
            precio_y = get_yfinance_close(ts[j], dates[i - 1], aggregate)
            rend_list[ts[j]] = precio_h / precio_y - 1

        Rlist = rend_list.T.sort_values(by="rend")
        for j in Rlist.index.tolist():
            if Rlist.loc[j, "rend"] <= -0.05:
                portact_d[dates[i]].loc[j, "Shares"] = int(portact_d[dates[i - 1]].loc[j, "Shares"] * 0.975)
                portact_d[dates[i]].loc["MXN.MX", "Shares"] = portact_d[dates[i]].loc["MXN.MX", "Shares"] + \
                                                              ((portact_d[dates[i - 1]].loc[j, "Shares"] -
                                                                portact_d[dates[i]].loc[j, "Shares"]) *
                                                               get_yfinance_close(j, dates[i], aggregate) * (
                                                                           1 - comission))

                portact_d[dates[i]].loc["MXN.MX", "Value"] = portact_d[dates[i]].loc["MXN.MX", "Shares"]
                portact_d[dates[i]].loc[j, "Value"] = portact_d[dates[i]].loc[j, "Shares"] * get_yfinance_close(j,
                                                                                                                dates[
                                                                                                                    i],
                                                                                                                aggregate)

                commmm = ((portact_d[dates[i - 1]].loc[j, "Shares"] -
                           portact_d[dates[i]].loc[j, "Shares"]) *
                          get_yfinance_close(j, dates[i], aggregate) * (comission))
                acum = operaciones[operaciones["index"] == j].comisión.sum()
                to_append = [j, dates[i],
                             portact_d[dates[i]].loc[j, "Shares"],
                             - portact_d[dates[i - 1]].loc[j, "Shares"] + portact_d[dates[i]].loc[j, "Shares"],
                             get_yfinance_close(j, dates[i], aggregate), abs(commmm), acum + abs(commmm)]

                to_append = pd.DataFrame([to_append], columns=operaciones.columns)
                operaciones = operaciones.append(to_append, ignore_index=True)

        Rlist = rend_list.T.sort_values(by="rend", ascending=False)
        for j in Rlist.index.tolist():
            if Rlist.loc[j, "rend"] >= 0.05:
                available = portact_d[dates[i]].loc["MXN.MX", "Value"]
                price = get_yfinance_close(j, dates[i], aggregate)
                twofive_c = int(portact_d[dates[i - 1]].loc[j, "Shares"] * 0.025) * price * (1 + comission)
                cap_c = available * (1 - comission)
                sha_tb = 0
                if twofive_c <= cap_c:
                    sha_tb = int(portact_d[dates[i - 1]].loc[j, "Shares"] * 0.025)
                if twofive_c > cap_c:
                    sha_tb = available * (1 - comission) / price

                portact_d[dates[i]].loc[j, "Shares"] = int(portact_d[dates[i - 1]].loc[j, "Shares"] + sha_tb)
                portact_d[dates[i]].loc["MXN.MX", "Shares"] = portact_d[dates[i]].loc["MXN.MX", "Shares"] - \
                                                              (sha_tb * price * (1 + comission))

                portact_d[dates[i]].loc["MXN.MX", "Value"] = portact_d[dates[i]].loc["MXN.MX", "Shares"]
                portact_d[dates[i]].loc[j, "Value"] = portact_d[dates[i]].loc[j, "Shares"] * price

                commmm = ((portact_d[dates[i - 1]].loc[j, "Shares"] -
                           portact_d[dates[i]].loc[j, "Shares"]) *
                          get_yfinance_close(j, dates[i], aggregate) * (comission))
                acum = operaciones[operaciones["index"] == j].comisión.sum()
                to_append = [j, dates[i],
                             portact_d[dates[i]].loc[j, "Shares"],
                             - portact_d[dates[i - 1]].loc[j, "Shares"] + portact_d[dates[i]].loc[j, "Shares"],
                             get_yfinance_close(j, dates[i], aggregate), abs(commmm), acum + abs(commmm)]
                to_append = pd.DataFrame([to_append], columns=operaciones.columns)
                operaciones = operaciones.append(to_append, ignore_index=True)

        for j in Rlist.index.tolist():
            if Rlist.loc[j, "rend"] < 0.05 and Rlist.loc[j, "rend"] > -0.05:

                portact_d[dates[i]].loc[j, "Shares"] = int(portact_d[dates[i - 1]].loc[j, "Shares"] * 1)
                portact_d[dates[i]].loc[j, "Value"] = portact_d[dates[i]].loc[j, "Shares"] * get_yfinance_close(j,
                                                                                                                dates[
                                                                                                                    i],
                                                                                                                aggregate)


    return portact_d, operaciones


def metrics(df_pasiva_a, df_pasiva_b, df_activa, rf):
    metrics = pd.DataFrame(data=np.zeros((3, 5)),
                           columns=["medida", "descripcion", "inv_activa", "inv_pasiva_a", "inv_pasiva_b"])
    metrics["medida"] = ["rend_m", "rend_c", "sharpe"]
    metrics["descripcion"] = ["Rendimiento Promedio Mensual", "Rendimiento mensual acumulado", "Sharpe Ratio"]

    metrics.loc[0, "inv_activa"] = df_activa.rend.mean()*30
    metrics.loc[1, "inv_activa"] = df_activa.rend_acum.tolist()[-1]
    metrics.loc[2, "inv_activa"] = (metrics.loc[0, "inv_activa"] - rf) / df_activa.rend.std() * np.sqrt(30)

    metrics.loc[0, "inv_pasiva_a"] = df_pasiva_a.rend.mean()
    metrics.loc[1, "inv_pasiva_a"] = df_pasiva_a.rend_acum.tolist()[-1]
    metrics.loc[2, "inv_pasiva_a"] = (metrics.loc[0, "inv_pasiva_a"] - rf) / df_pasiva_a.rend.std()

    metrics.loc[0, "inv_pasiva_b"] = df_pasiva_b.rend.mean()
    metrics.loc[1, "inv_pasiva_b"] = df_pasiva_b.rend_acum.tolist()[-1]
    metrics.loc[2, "inv_pasiva_b"] = (metrics.loc[0, "inv_pasiva_b"] - rf) / df_pasiva_b.rend.std()

    return metrics