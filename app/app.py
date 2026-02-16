import pandas as pd

df = pd.read_csv('datasets/full_grouped.csv')

import plotly.express as px

fig = px.scatter(df, x='Confirmed', y='Deaths', color='WHO Region')
fig.show()