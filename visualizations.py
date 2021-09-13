
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: visualizations.py : python script with data visualization functions                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.subplots as sp


def graph_tickers(x_points, y_points, annual_ret_summ):
    plt.figure(figsize=(10, 4))
    plt.plot(x_points, y_points, 'ro', ms=5)
    plt.xlabel('Annual Volatility $\sigma$')
    plt.ylabel('Annual Expected return $E[r]$')
    for i in range(0, len(x_points)):
        plt.text(x_points[i], y_points[i], annual_ret_summ.columns[i])

def graph_frontera(w_EMV, E_EMV, s_EMV, Sigma, w_minvar, E_minvar, s_minvar, rf):
    w = np.linspace(0, 3, 100)
    cov = w_EMV.T.dot(Sigma).dot(w_minvar)
    portafolios = pd.DataFrame(data={'w': w,
                                     '1-w': 1 - w,
                                     'Media': w * E_EMV + (1 - w) * E_minvar,
                                     'Vol': ((w * s_EMV) ** 2 + ((1 - w) * s_minvar) ** 2 + 2 * w * (
                                                 1 - w) * cov) ** 0.5})
    portafolios['RS'] = (portafolios['Media'] - rf) / portafolios['Vol']

    plt.figure(figsize=(10, 4))
    plt.scatter(portafolios['Vol'], portafolios['Media'], c=portafolios['RS'], cmap='RdYlBu', label='Front. Min. Var.')
    plt.plot(s_minvar, E_minvar, 'og', ms=7, label='Port. Min. Var.')
    plt.plot(s_EMV, E_EMV, 'or', ms=7, label='Port. EMV')
    plt.legend(loc='best')
    plt.colorbar()
    plt.xlabel("Volatility $\sigma$")
    plt.ylabel("Expected Return $E[r]$")
    plt.grid()

def graph_1line_px(x, y, title):
    figure = px.line(x=x, y=y, markers=True)

    figure.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                       'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                          'title': title})
    figure.show()

def graph_1line_px(x, y, title):
    figure = px.line(x=x, y=y, markers=True)

    figure.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                       'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                          'title': title})
    return figure

def subplot_2(figure1, figure2, title):
    figure1_traces = []
    figure2_traces = []
    for trace in range(len(figure1["data"])):
        figure1_traces.append(figure1["data"][trace])
    for trace in range(len(figure2["data"])):
        figure2_traces.append(figure2["data"][trace])

    # Create a 1x2 subplot
    this_figure = sp.make_subplots(rows=1, cols=2)

    # Get the Express fig broken down as traces and add the traces to the proper plot within in the subplot
    for traces in figure1_traces:
        this_figure.append_trace(traces, row=1, col=1)
    for traces in figure2_traces:
        this_figure.append_trace(traces, row=1, col=2)

    # the subplot as shown in the above image
    this_figure.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                               'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                               'title': title})

    return this_figure

def subplot_3(figure1, figure2, figure3, title):
    figure1_traces = []
    figure2_traces = []
    figure3_traces = []
    for trace in range(len(figure1["data"])):
        figure1_traces.append(figure1["data"][trace])
    for trace in range(len(figure2["data"])):
        figure2_traces.append(figure2["data"][trace])
    for trace in range(len(figure3["data"])):
        figure3_traces.append(figure3["data"][trace])

    # Create a 1x2 subplot
    this_figure = sp.make_subplots(rows=1, cols=3)

    # Get the Express fig broken down as traces and add the traces to the proper plot within in the subplot
    for traces in figure1_traces:
        this_figure.append_trace(traces, row=1, col=1)
    for traces in figure2_traces:
        this_figure.append_trace(traces, row=1, col=2)
    for traces in figure3_traces:
        this_figure.append_trace(traces, row=1, col=3)

    # the subplot as shown in the above image
    this_figure.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                               'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                               'title': title})

    return this_figure