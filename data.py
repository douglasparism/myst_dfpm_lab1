"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: data.py : python script for data collection                                                 -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import os
import pandas as pd


def dates(period):
    abspath = os.path.abspath('files/')
    datesl = [f[-12:-4] for f in os.listdir(abspath) if os.path.isfile(os.path.join(abspath, f))]
    if period == "pre":
        return datesl[0:25]
    if period == "in":
        return datesl[25:]
    if period == "all":
        return datesl


def get_naftrac_data(recalc_weights):
    abspath = os.path.abspath('files/')
    datesl = [f[-12:-4] for f in os.listdir(abspath) if os.path.isfile(os.path.join(abspath, f))]
    archivos = ["NAFTRAC_" + i for i in sorted(datesl)]
    data_dicc = {}
    j = 0

    for i in archivos:
        p = os.path.join(abspath, i + '.csv')
        data = pd.read_csv(p, skiprows=2)
        data = data[:-1]
        cols_to_num = ["Precio", "Acciones", "Valor de mercado"]
        data[cols_to_num] = data[cols_to_num].replace({',': ''}, regex=True).astype(float, errors='raise')
        if recalc_weights:
            data["Peso (%)"] = data["Valor de mercado"] / sum(data["Valor de mercado"])
        if not recalc_weights:
            data["Peso (%)"] = data["Peso (%)"] / 100
        key = datesl[j]
        j += 1
        data_dicc[key] = data

    return data_dicc
