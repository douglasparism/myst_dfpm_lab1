
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import data as dt
from functions import *
from visualizations import *

NAFTRAC_dicc = dt.get_naftrac_data(recalc_weights=False)
comission = 0.00125
restricted = ["KOFL.MX", "KOFUBL.MX", "USD.MXN", "BSMXB.MX", "NMKA.MX", "MXN.MX"]
dates_a = dt.dates("pre")
dates_b = dt.dates("in")
dates_full = dt.dates("all")
capital = 1000000

tickers, weights = get_tickers(NAFTRAC_dicc["20180131"])
aggregate = get_yfinance_close_aggregate(tickers, dates_full, "Adj Close")


port_a, comission_df_a = init_port(capital, weights, tickers, dates_a[0], restricted, comission, aggregate)
portafolios_valor = [sum(reval_port(port_a, i, restricted, aggregate)["Value"]) for i in dates_a]
df_pasiva_a = create_df_pasiva(dates_a, portafolios_valor).set_index("timestamp")

port_b, comission_df_b = init_port(capital, weights, tickers, dates_b[0], restricted, comission, aggregate)
portafolios_valor = [sum(reval_port(port_b, i, restricted, aggregate)["Value"]) for i in dates_b]
df_pasiva_b = create_df_pasiva(dates_b, portafolios_valor).set_index("timestamp")


d_slice = [datetime.datetime.strptime("20180131", '%Y%m%d').strftime('%Y-%m-%d'),
           datetime.datetime.strptime("20200228", '%Y%m%d').strftime('%Y-%m-%d')]

annual_ret_summ, log_r = get_annual_summ(tickers, restricted, aggregate, d_slice)
corr = corr_matrix(tickers, restricted, aggregate, d_slice)

x_points = annual_ret_summ.loc['Volatilidad'].values
y_points = annual_ret_summ.loc['Media'].values
graph_tickers(x_points, y_points, annual_ret_summ)

rf = 0.0429
EMV, w_EMV, E_EMV, SR_EMV, s_EMV, Sigma = min_sr_port(annual_ret_summ, corr, rf)
minvar, w_minvar, E_minvar, s_minvar = min_var_port(annual_ret_summ, corr)

graph_frontera(w_EMV, E_EMV, s_EMV, Sigma, w_minvar, E_minvar, s_minvar, rf)

Pesos_Max_SR = pd.DataFrame(w_EMV, index=annual_ret_summ.columns[:-1]).T
tickers = annual_ret_summ.columns[:-1]
weights = dict(zip(tickers, w_EMV))

port_activa, comission_df_activa = init_port_act(capital, weights, tickers, "20200228",
                                                 restricted, comission, aggregate)

dates_activa = aggregate[aggregate.index.to_series().between('2020-02-28', '2021-03-29')].index.strftime('%Y%m%d').tolist()
portact_d = rebalance_port(port_activa, aggregate, dates_activa, comission)
portafolios_valor = [sum(portact_d[i]["Value"]) for i in dates_activa]
df_activa = create_df_pasiva(dates_activa, portafolios_valor)

