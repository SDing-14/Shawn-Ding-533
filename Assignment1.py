from dash import Dash, html, dcc, dash_table, Input, Output, State
import eikon as ek
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import os
import sklearn
from sklearn.linear_model import LinearRegression

eikon_api = '93e680b88cf440e197b190b6e2e9810e876376e9'
ek.set_app_key(eikon_api)

dt_prc_div_splt = pd.read_csv('unadjusted_price_history.csv')

app = Dash(__name__)
app.layout = html.Div([
    html.Div([
        html.Label(children = 'Benchmark', id = 'benchmark-label'),
        dcc.Input(id = 'benchmark-id', type = 'text', value="IVV", style = {'display':'block'}),
        html.Label(children = 'Asset', id = 'asset-label'),
        dcc.Input(id = 'asset-id', type = 'text', value="AAPL.O", style = {'display':'block'}),
        html.Label(children = 'Start Date', id = 'start-label'),
        dcc.Input(id = 'start-date', type = 'text', value = "2022-01-01", style = {'display':'block'}),
        html.Label(children = 'End Date', id = 'end-label'),
        dcc.Input(id = 'end-date', type = 'text', value = "2022-12-31", style = {'display':'block'})
    ]),
    html.Button('QUERY Refinitiv', id = 'run-query', n_clicks = 0),
    html.H2('Raw Data from Refinitiv'),
    dash_table.DataTable(
        id = "history-tbl",
        page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'}
    ),
    html.H2('Historical Returns'),
    dash_table.DataTable(
        id = "returns-tbl",
        page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'}
    ),
    html.H2('Alpha & Beta Scatter Plot'),
    html.Label(children='Start Date', id='start-label2'),
    dcc.Input(id='start-date2', type='text', value="2022-01-01", style={'display': 'block'}),
    html.Label(children='End Date', id='end-label2'),
    dcc.Input(id='end-date2', type='text', value="2022-12-31", style={'display': 'block'}),
    html.Button('QUERY Plot', id = 'run-query2', n_clicks = 0),
    dcc.Graph(id="ab-plot"),
    html.P(id='summary-text', children=""),
    #dcc.Markdown(id = 'alpha-container'),
    #dcc.Markdown(id = 'beta-container')
])

return_dict = {}
@app.callback(
    Output("history-tbl", "data"),
    Input("run-query", "n_clicks"),
    [State('benchmark-id', 'value'), State('asset-id', 'value'),
     State('start-date', 'value'), State('end-date', 'value')],
    prevent_initial_call=True
)

def query_refinitiv(n_clicks, benchmark_id, asset_id, start_date, end_date):
    global return_dict
    assets = [benchmark_id, asset_id]
    prices, prc_err = ek.get_data(
        instruments=assets,
        fields=[
            'TR.OPENPRICE(Adjusted=0)',
            'TR.HIGHPRICE(Adjusted=0)',
            'TR.LOWPRICE(Adjusted=0)',
            'TR.CLOSEPRICE(Adjusted=0)',
            'TR.PriceCloseDate'
        ],
        parameters={
            'SDate': start_date,
            'EDate': end_date,
            'Frq': 'D'
        }
    )

    divs, div_err = ek.get_data(
        instruments=assets,
        fields=[
            'TR.DivExDate',
            'TR.DivUnadjustedGross',
            'TR.DivType',
            'TR.DivPaymentType'
        ],
        parameters={
            'SDate': start_date,
            'EDate': end_date,
            'Frq': 'D'
        }
    )

    splits, splits_err = ek.get_data(
        instruments=assets,
        fields=['TR.CAEffectiveDate', 'TR.CAAdjustmentFactor'],
        parameters={
            "CAEventType": "SSP",
            'SDate': start_date,
            'EDate': end_date,
            'Frq': 'D'
        }
    )

    prices.rename(
        columns={
            'Open Price': 'open',
            'High Price': 'high',
            'Low Price': 'low',
            'Close Price': 'close'
        },
        inplace=True
    )
    prices.dropna(inplace=True)
    prices['Date'] = pd.to_datetime(prices['Date']).dt.date

    divs.rename(
        columns={
            'Dividend Ex Date': 'Date',
            'Gross Dividend Amount': 'div_amt',
            'Dividend Type': 'div_type',
            'Dividend Payment Type': 'pay_type'
        },
        inplace=True
    )
    divs.dropna(inplace=True)
    divs['Date'] = pd.to_datetime(divs['Date']).dt.date
    divs = divs[(divs.Date.notnull()) & (divs.div_amt > 0)]

    splits.rename(
        columns={
            'Capital Change Effective Date': 'Date',
            'Adjustment Factor': 'split_rto'
        },
        inplace=True
    )
    splits.dropna(inplace=True)
    splits['Date'] = pd.to_datetime(splits['Date']).dt.date

    unadjusted_price_history = pd.merge(
        prices, divs[['Instrument', 'Date', 'div_amt']],
        how='outer',
        on=['Date', 'Instrument']
    )
    unadjusted_price_history['div_amt'].fillna(0, inplace=True)

    unadjusted_price_history = pd.merge(
        unadjusted_price_history, splits,
        how='outer',
        on=['Date', 'Instrument']
    )
    unadjusted_price_history['split_rto'].fillna(1, inplace=True)

    if unadjusted_price_history.isnull().values.any():
        raise Exception('missing values detected!')

    return_dict = unadjusted_price_history.to_dict('records')

    return(unadjusted_price_history.to_dict('records'))




@app.callback(
    Output("returns-tbl", "data"),
    Input("history-tbl", "data"),
    prevent_initial_call = True
)
def calculate_returns(history_tbl):

    dt_prc_div_splt = pd.DataFrame(history_tbl)

    # Define what columns contain the Identifier, date, price, div, & split info
    ins_col = 'Instrument'
    dte_col = 'Date'
    prc_col = 'close'
    div_col = 'div_amt'
    spt_col = 'split_rto'

    dt_prc_div_splt[dte_col] = pd.to_datetime(dt_prc_div_splt[dte_col])
    dt_prc_div_splt = dt_prc_div_splt.sort_values([ins_col, dte_col])[
        [ins_col, dte_col, prc_col, div_col, spt_col]].groupby(ins_col)
    numerator = dt_prc_div_splt[[dte_col, ins_col, prc_col, div_col]].tail(-1)
    denominator = dt_prc_div_splt[[prc_col, spt_col]].head(-1)
    return(
        pd.DataFrame({
            'Date': numerator[dte_col].reset_index(drop=True),
            'Instrument': numerator[ins_col].reset_index(drop=True),
            'rtn': np.log(
                (numerator[prc_col] + numerator[div_col]).reset_index(drop=True) / (
                        denominator[prc_col] * denominator[spt_col]
                ).reset_index(drop=True)
            )
        }).pivot_table(
            values='rtn', index='Date', columns='Instrument'
        ).to_dict('records')
    )

'''
@app.callback(
    Output("ab-plot", "figure"),
    Input("returns-tbl", "data"),
    [State('benchmark-id', 'value'), State('asset-id', 'value')],
    prevent_initial_call = True
)
def render_ab_plot(returns, benchmark_id, asset_id):
    global return_dict
    return(
        px.scatter(returns, x=benchmark_id, y=asset_id, trendline='ols')
    )
'''
new_df = {}
@app.callback(
    [Output("ab-plot", "figure"),
    Output('summary-text', 'children')],
    Input("run-query2", "n_clicks"),
    [State('benchmark-id', 'value'), State('asset-id', 'value'),
     State('start-date2', 'value'), State('end-date2', 'value')],
    prevent_initial_call = True
)
def render_ab_plot_filtered(n_clicks, benchmark_id, asset_id, start_date2, end_date2):
    global return_dict
    global new_df
    local_returns = pd.DataFrame(return_dict)
    start_date2 = pd.to_datetime(start_date2)
    end_date2 = pd.to_datetime(end_date2)

    dt_prc_div_splt = local_returns[(local_returns['Date']>start_date2) & (local_returns['Date']<end_date2)]

    ins_col = 'Instrument'
    dte_col = 'Date'
    prc_col = 'close'
    div_col = 'div_amt'
    spt_col = 'split_rto'

    dt_prc_div_splt[dte_col] = pd.to_datetime(dt_prc_div_splt[dte_col])
    dt_prc_div_splt = dt_prc_div_splt.sort_values([ins_col, dte_col])[
        [ins_col, dte_col, prc_col, div_col, spt_col]].groupby(ins_col)
    numerator = dt_prc_div_splt[[dte_col, ins_col, prc_col, div_col]].tail(-1)
    denominator = dt_prc_div_splt[[prc_col, spt_col]].head(-1)

    new_df = pd.DataFrame({
        'Date': numerator[dte_col].reset_index(drop=True),
        'Instrument': numerator[ins_col].reset_index(drop=True),
        'rtn': np.log(
            (numerator[prc_col] + numerator[div_col]).reset_index(drop=True) / (
                    denominator[prc_col] * denominator[spt_col]
            ).reset_index(drop=True)
        )
    }).pivot_table(
        values='rtn', index='Date', columns='Instrument'
    ).to_dict('records')

    new_df = pd.DataFrame(new_df)
    df = pd.DataFrame({'x': new_df[benchmark_id], 'y': new_df[asset_id]})
    reg = LinearRegression().fit(df[['x']], df['y'])
    alpha = reg.intercept_
    beta = reg.coef_[0]
    text = 'Alpha is: ', alpha, ', and Beta is:', beta
    '''
    print(new_df)
    fit = plot1.data[0].fit
    alpha = fit.intercept_
    beta = fit.slope
    print('alpha: ', alpha, '\nbeta: ', beta)
    '''
    return(
        px.scatter(new_df, x=benchmark_id, y=asset_id, trendline='ols'), text
    )

print(new_df)
'''

@app.callback(
    Output("a-b", "data"),
    Input('history-tbl', 'data'),
    [State('benchmark-id', 'value'), State('asset-id', 'value')],
    prevent_initial_call = True
)
def calculate_a_b(history_table, benchmark_id, asset_id):
    global new_df
    df = pd.DataFrame(new_df)
    df = pd.DataFrame({'x':df[benchmark_id], 'y':df[asset_id]})
    reg = LinearRegression().fit(df[['x']], df['y'])
    alpha = reg.intercept_
    beta = reg.coef_[0]
    print(alpha, beta)
    return new_df

'''

if __name__ == '__main__':
    app.run_server(debug=True)