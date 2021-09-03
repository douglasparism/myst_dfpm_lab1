
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
