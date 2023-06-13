import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima_model import ARMA,ARMAResults,ARIMA,ARIMAResults
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tsa.stattools import adfuller
from PIL import Image
from flask import Flask,render_template,request
import os
import subprocess
import plotly.graph_objs as go


run=Flask('__name__')


def predict(product):
    
    while True:
        df = pd.read_csv("Product.csv",parse_dates=['Date'])
        df.isnull().sum()
        df.dropna(axis=0, inplace=True)

        df.reset_index(drop = True)
        df.isnull().sum()

        df.sort_values('Date')[10:20]
        df['Order_Demand'] = df['Order_Demand'].str.replace('(',"")
        df['Order_Demand'] = df['Order_Demand'].str.replace(')',"")
        df.sort_values('Date')[10:20]
        df['Order_Demand'] = df['Order_Demand'].astype('int64')
        df.groupby('Warehouse')['Order_Demand'].sum().sort_values(ascending=False)
        df1 = pd.DataFrame(df.groupby('Product_Category')['Order_Demand'].sum().sort_values(ascending=False))
        df1["% Contribution"] = df1['Order_Demand']/df1['Order_Demand'].sum()*100
        df2 = pd.pivot_table(df,index=["Date"],values=["Order_Demand"],columns=["Product_Category"],aggfunc=np.sum)
        df2.columns = df2.columns.droplevel(0)
        df2[product].dropna()
        y = df2.resample('M').sum() 
        y.index.freq = "M"
        df_019=0
        df_019 = pd.DataFrame(y[product].iloc[12:-1])
        span = 4
        alpha = 2/(span+1)
        df_019['EWMA4'] = df_019[product].ewm(alpha=alpha,adjust=False).mean()

        df_019['SES4']=SimpleExpSmoothing(df_019[product]).fit(smoothing_level=alpha,optimized=False).fittedvalues.shift(-1)

        df_019['DESadd4'] = ExponentialSmoothing(df_019[product], trend='add').fit().fittedvalues.shift(-1)

        len(df_019[product])

        model = SARIMAX(df_019[product],order=(3,1,3),seasonal_order=(0,1,1,12))
        results = model.fit()
        fcast = results.predict(len(df_019[product]),len(df_019[product])+4,typ='levels').rename('SARIMA(3,1,3)(0,1,1,12) Forecast')
        dt= pd.read_csv("task.csv")
        fig= go.Figure()
        dtFrame = df[df['Product_Category'].str.contains(product)]
        # print(dtFrame)

        fig.add_trace(go.Scatter(dict(x=dt.loc[:6,'Date'], y=df_019[product], mode='lines+markers', name='Valid')))
        fig.add_trace(go.Scatter(dict(x=dt.loc[6:10,"Date"], y=fcast, mode='lines+markers', name="Forecast")))

        fig.update_layout(title=go.layout.Title(
                text="Product Demand Forecasted",
                xref="paper",x=0),
            margin=dict(l=10, r=0, t=50, b=50),
            xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Year",font=dict(
                        family="Courier New, monospace",
                        size=18,
                        color="#7f7f7f"))),
            yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Sales",font=dict(
                        family="Courier New, monospace",
                        size=18,
                        color="#7f7f7f"))))

        fig.show()
        break


@run.route('/')
def main():  
    return render_template('main.html')

@run.route('/predict',methods=['post'])
def demand():
    product=request.form['Product']
    predict(product)
    return render_template('main.html')




run.run(debug=True)