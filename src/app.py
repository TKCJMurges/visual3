'''
 # @ Create Time: 2023-11-26 14:46:34.011634
'''
# Importing the necessary files
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.io as pio
from dash import Dash, dcc, html
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

app = Dash(__name__, title="MyVisual")

# Declare server for Heroku deployment. Needed for Procfile.
server = app.server

# Data loading and preprocessing
stock_data = pd.read_csv("data.csv")
stock_data['Volume'] = pd.to_numeric(stock_data['Volume'].str.replace(',', ''), errors='coerce')

# Convert 'Date' column to datetime format
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

current_time_str = stock_data['Date'].max()
print(current_time_str)
current_time = current_time_str.to_pydatetime()
print(current_time)

# Create a dropdown filter for company selection
companies = stock_data['Name'].unique().tolist()

# Calculate VWAP for each company
stock_data['VWAP'] = (stock_data['Closing_Price'] * stock_data['Volume']).cumsum() / stock_data['Volume'].cumsum()

# Calculate Simple Moving Average (SMA) for Closing_Price (taking a window of 3 days as an example)
stock_data['SMA'] = stock_data['Closing_Price'].rolling(window=3).mean()

# Calculate Bollinger Bands
window = 20  # You can adjust the window size as needed
stock_data['Middle_Band'] = stock_data['Closing_Price'].rolling(
    window=window).mean()
stock_data['Upper_Band'] = stock_data['Middle_Band'] + 2 * stock_data[
    'Closing_Price'].rolling(window=window).std()
stock_data['Lower_Band'] = stock_data['Middle_Band'] - 2 * stock_data[
    'Closing_Price'].rolling(window=window).std()

# Calculate RSI for Closing_Price (taking a window of 14 days as an example)
delta = stock_data['Closing_Price'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))

# Pivot the stock_dataset to have Date as the index and companies as columns
pivot_stock_data = stock_data.pivot(index='Date',
                                    columns='Name',
                                    values='Closing_Price')

# Calculate correlation matrix
correlation_matrix = pivot_stock_data.corr()

# Calculate quartiles for correlation coefficients
coefficients = correlation_matrix.values.flatten()
lower_quantile = np.percentile(coefficients, 25)
upper_quantile = np.percentile(coefficients, 75)

# Define contours for lower and upper quartiles
contour_lower = go.Contour(
    z=correlation_matrix.values,
    x=correlation_matrix.index,
    y=correlation_matrix.columns,
    contours=dict(start=-1, end=lower_quantile,
                  size=0.05),  # Adjust size as needed
    colorscale='Blues',  # Choose a colorscale for lower quartile contours
    showscale=False,
    name=f'Lower Quartile Contours (-1 - {lower_quantile:.2f})')

contour_upper = go.Contour(
    z=correlation_matrix.values,
    x=correlation_matrix.index,
    y=correlation_matrix.columns,
    contours=dict(start=upper_quantile, end=1,
                  size=0.05),  # Adjust size as needed
    colorscale='Reds',  # Choose a colorscale for upper quartile contours
    showscale=False,
    name=f'Upper Quartile Contours ({upper_quantile:.2f} - 1)')
# Create an empty figure
fig = go.Figure()
fig_rs = go.Figure()
fig_sm = go.Figure()
fig_cs = go.Figure()
fig_bc = go.Figure()
fig_ts = go.Figure()
fig2 = go.Figure()

# Define your layout for each visualization

# Visualization 3 layout
candlestick_layout = html.Div([
    dcc.Dropdown(
        id='company-dropdown_cs',
        options=[{
            'label': company,
            'value': company
        } for company in companies],
        value=companies[0]  # Set default value to the first company
    ),
    dcc.Graph(figure=fig_cs, id='candlestick-chart')
])

# Combine all visualizations vertically
app.layout = html.Div([
    html.H1("Interactive chart"), candlestick_layout
])

# CANDLE-STICK
# Iterate through each company to add candlestick traces to the figure
for company in companies:
  company_data = stock_data[stock_data['Name'] == company]
  fig_cs.add_trace(
      go.Candlestick(x=company_data['Date'],
                     open=company_data['Open'],
                     high=company_data['Daily_High'],
                     low=company_data['Daily_Low'],
                     close=company_data['Closing_Price'],
                     name=company,
                     visible=False))

# Set the visibility of the first company's trace to True initially
fig_cs.data[0].visible = True

# Create buttons for dropdown selection
buttons = []
for i, company in enumerate(companies):
  visibility = [i == j for j in range(len(companies))]
  button = dict(label=company,
                method="update",
                args=[{
                    "visible": visibility
                }, {
                    "title": f"Interactive Candlestick Chart - {company}"
                }])
  buttons.append(button)

# Update layout to include dropdown selection
fig_cs.update_layout(updatemenus=[{
    "buttons": buttons,
    "direction": "down",
    "showactive": True,
    "x": 0.5,
    "y": 1.5
}],
                  xaxis_title='Date',
                  yaxis_title='Price',
                  title='Interactive Candlestick Chart - Closing Prices')

# Define callback to update the candlestick chart based on dropdown selection
@app.callback(Output('candlestick-chart', 'figure'),
              [Input('company-dropdown_cs', 'value')])
def update_cs_chart(selected_company):
  # Create a copy of the figure to avoid modifying the original figure
  updated_fig_cs = fig_cs

  # Update visibility of the selected company's trace
  for trace in updated_fig_cs.data:
    trace.visible = (trace.name == selected_company)

  return updated_fig_cs

# Run the app
if __name__ == '__main__':
  app.run_server(debug=True, port=int(os.environ.get('PORT', 8050)))