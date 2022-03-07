import plotly.express as px
import pandas as pd
import numpy as np
import math

def load_and_plot(path):
    data = pd.read_csv(path)
    if 'decision_boundry' in data.columns:
        return data, plot_data_with_decsion_boundry(data, 'decision_boundry')
    else:
        fig = px.scatter(data, x='x',y='y',color='class_id', template='plotly_white').update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
    return data, fig

def plot_data_with_decsion_boundry(data, decision_boundry_col, class_col = 'class_id', title='None', line_legened='Decision Boundry', update_markers = True, marker_size=12, line_width=2):
    a_data = data[data[class_col] == 'A']
    b_data = data[data[class_col] == 'B']
    fig = px.line(data.sort_values(by='x'), x='x', y=decision_boundry_col, color=px.Constant(line_legened), template='plotly_white').update_traces(line=dict(color="DarkSlateGrey", width=3))
    fig = fig.add_scatter(x=a_data.x, y=a_data.y, mode='markers', name='Class A')
    if update_markers:
        fig = fig.update_traces(marker=dict(size=marker_size, line=dict(width=line_width, color='DarkSlateGrey')), selector=dict(mode='markers'))
    fig = fig.add_scatter(x=b_data.x, y=b_data.y, mode='markers', name='Class B')
    if update_markers:
        fig = fig.update_traces(marker=dict(size=marker_size, line=dict(width=line_width, color='DarkSlateGrey')), selector=dict(mode='markers'))
    if title:
        fig.update_layout(title_text=title, title_x=0.5)
    return fig


def plot_sigmoid_function():
    def sigmoid(x):
        a = []
        for item in x:
            a.append(1/(1+math.exp(-item)))
        return a
    
    x = np.arange(-10., 10., 0.2)
    sig = sigmoid(x)
    fig = px.line(pd.DataFrame({'x': x, 'sig': sig}), x='x', y='sig', template='plotly_white')
    fig.update_traces(line=dict(color="#00CC96", width=3))
    fig.update_layout(title_text='Sigmoid function', title_x=0.5)
    return fig