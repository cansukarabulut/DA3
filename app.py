from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello, World!"

import pandas as pd

url = "https://raw.githubusercontent.com/cansukarabulut/DA3/refs/heads/main/global_electricity_production_data.csv"
df = pd.read_csv(url)

print(df)
turkey_data = df[df['country_name'] == 'Turkey']
print(turkey_data.info())
print(turkey_data.head())
print(turkey_data.isnull().sum())
turkey_data.loc[:, 'date'] = pd.to_datetime(turkey_data['date'])
turkey_data = turkey_data.sort_values(by='date')
print(turkey_data.head())
import matplotlib.pyplot as plt

total_production = turkey_data.groupby('date')['value'].sum().reset_index()

plt.figure(figsize=(10, 6))
plt.plot(total_production['date'], total_production['value'], marker='o', linestyle='-')
plt.title("Türkiye Electricity Production (by Time)", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Product Quantity (GWh)", fontsize=12)
plt.grid()
plt.show()

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np


X = np.arange(len(total_production)).reshape(-1, 1)  
y = total_production['value'].values  

linear_model = LinearRegression()
linear_model.fit(X, y)

linear_y_pred = linear_model.predict(X)

linear_mse = mean_squared_error(y, linear_y_pred)
linear_r2 = r2_score(y, linear_y_pred)

print(f"Linear Regression MSE: {linear_mse:.2f}")
print(f"Linear Regression R^2: {linear_r2:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(total_production['date'], y, label="Real Vals", marker='o')
plt.plot(total_production['date'], linear_y_pred, label="Lunear Prediction", linestyle='--', color='red')
plt.title("Electricity Generation Forecasting with Linear Regression", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Product Quantity (GWh)", fontsize=12)
plt.legend()
plt.grid()
plt.show()

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

X = np.arange(len(total_production)).reshape(-1, 1)  
y = total_production['value'].values 

degrees = [1, 2, 3, 4, 5]  
mse_values = []  
r2_values = []  

for degree in degrees:
    
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)

    
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)

   
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mse_values.append(mse)
    r2_values.append(r2)
    print(f"Degree of Polynomial: {degree}, MSE: {mse:.2f}, R^2: {r2:.4f}")


plt.figure(figsize=(14, 6))


plt.subplot(1, 2, 1)
plt.plot(degrees, mse_values, marker='o', linestyle='-', color='blue')
plt.title("MSE by Model Complexity", fontsize=14)
plt.xlabel("Degree of Polynomial", fontsize=12)
plt.ylabel("MSE", fontsize=12)
plt.grid()


plt.subplot(1, 2, 2)
plt.plot(degrees, r2_values, marker='o', linestyle='-', color='green')
plt.title("R^2 by Model Complexity", fontsize=14)
plt.xlabel("Degree of Polynomial", fontsize=12)
plt.ylabel("R^2", fontsize=12)
plt.grid()

plt.tight_layout()
plt.show()


best_degree_mse = degrees[np.argmin(mse_values)]
best_mse = min(mse_values)

best_degree_r2 = degrees[np.argmax(r2_values)]
best_r2 = max(r2_values)

print(f"Best polynomial degree for MSE: {best_degree_mse} ile MSE: {best_mse:.2f}")
print(f"Polynomial degree for best R^2: {best_degree_r2} ile R^2: {best_r2:.4f}")

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
bias_squared = np.array([0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64])
variance = np.array([0.0, 0.08, 0.1, 0.4, 0.4, 0.42, 0.42, 0.42, 0.42])
mse = np.array([0.64, 0.56, 0.54, 0.24, 0.24, 0.22, 0.22, 0.22, 0.22])

x_smooth = np.linspace(x.min(), x.max(), 300)
bias_smooth = make_interp_spline(x, bias_squared)(x_smooth)
variance_smooth = make_interp_spline(x, variance)(x_smooth)
mse_smooth = make_interp_spline(x, mse)(x_smooth)

plt.figure(figsize=(10, 5))
plt.plot(x_smooth, bias_smooth, 'b-', label='Bias^2')
plt.plot(x_smooth, variance_smooth, 'orange', label='Variance')
plt.plot(x_smooth, mse_smooth, 'g-', label='MSE')

plt.xlabel("Polynomial Degree")
plt.ylabel("Value")
plt.title("Bias-Variance Tradeoff")
plt.legend()
plt.grid(True)
plt.show()

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Bias-Variance Tradeoff"),

    dcc.Graph(id='tradeoff-graph'),

    dcc.Slider(
        id='degree-slider',
        min=x.min(),
        max=x.max(),
        step=1,
        value=5,
        marks={int(i): str(int(i)) for i in x}
    ),

    html.H2("Numerical Output"),
    html.Div(id='numerical-output'),
])

@app.callback(
    [Output('tradeoff-graph', 'figure'),
     Output('numerical-output', 'children')],
    [Input('degree-slider', 'value')]
)
def update_graph(degree):
    # Grafik verileri
    bias_val = bias_squared[degree - 1]
    variance_val = variance[degree - 1]
    mse_val = mse[degree - 1]

    fig = {
        'data': [
            {'x': x_smooth, 'y': bias_smooth, 'type': 'line', 'name': 'Bias^2', 'line': {'color': 'blue'}},
            {'x': x_smooth, 'y': variance_smooth, 'type': 'line', 'name': 'Variance', 'line': {'color': 'orange'}},
            {'x': x_smooth, 'y': mse_smooth, 'type': 'line', 'name': 'MSE', 'line': {'color': 'green'}},
        ],
        'layout': {
            'title': 'Bias-Variance Tradeoff',
            'xaxis': {'title': 'Polinom Derecesi'},
            'yaxis': {'title': 'Değer'},
            'legend': {'orientation': 'h'}
        }
    }

   
    numerical_output = html.Table([
        html.Tr([html.Th("Polynomial Degree"), html.Th("Bias^2"), html.Th("Variance"), html.Th("MSE")]),
        html.Tr([html.Td(degree), html.Td(f"{bias_val:.2f}"), html.Td(f"{variance_val:.2f}"), html.Td(f"{mse_val:.2f}")])
    ])

    return fig, numerical_output

if __name__ == '__main__':
    app.run_server(debug=True)







