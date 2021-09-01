
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

NAFTRAC_dicc = dt.get_naftrac_data()
comission = 0.00125
restricted = ["KOFL.MX", "KOFUBL.MX", "USD.MXN", "BSMXB.MX", "NMKA.MX", "MXN.MX"]
dates_a = dt.dates("pre")
dates_b = dt.dates("in")
capital = 1000000

tickers, weights = get_tickers(NAFTRAC_dicc["20180131"])

port = init_port(capital, weights, tickers, dates_a[0], restricted, comission)
# port2 = reval_port(port, dates_a[1], restricted)
portafolios_valor = [sum(reval_port(port, i, restricted)["Value"]) for i in dates_a]

#df_pasiva_a = pd.DataFrame([datetime.datetime.strptime(i, '%Y%m%d') for i in dates_a], columns=["timestamp"])
#df_pasiva_a["capital"] = portafolios_valor
#df_pasiva_a["rend"] = df_pasiva_a["capital"].pct_change()
#df_pasiva_a["rend_acum"] = df_pasiva_a["rend"].cumsum()

df_pasiva_a = create_df_pasiva(dates_a, portafolios_valor)