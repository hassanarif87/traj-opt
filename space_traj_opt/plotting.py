import plotly.graph_objects as go
#from plotly.graph_objects.Figure import add_trace
import numpy as np

def plot_axis(axis=None, id=1, line_type='solid', fig=None):
    if axis is None:
        axis = np.identity(3)
    if fig is None:
        fig = go.Figure()
    axis_name = ['x'+str(id), 'y'+str(id), 'z'+str(id)]
    colors= ['red', 'blue', 'green']
    for idx, name in enumerate(axis_name):
        fig.add_trace(
            go.Scatter3d(
                # Each row of the matrix represent the basis vector in the base frame
                x=[0,axis[idx,0]] , y=[0,axis[idx,1]], z=[0,axis[idx,2]], name=name, 
                line=dict(color=colors[idx], width=4, dash=line_type),
            )
        )
    return fig


def plot(x, y, xlabel=None, ylabel=None, title=None, trace_names=None):
    """
    Generate a Plotly plot.

    Args:
        x: List or array-like, x values.
        y: Array-like or list of array-like, y values.
        xlabel: (Optional) String, label for the x-axis. Default is None.
        ylabel: (Optional) String, label for the y-axis. Default is None.
        title: (Optional) String, title of the plot. Default is None.
        trace_names: (Optional) List of strings, names for each trace. Default is None.

    Returns:
        None (displays the plot).
    """

    # Convert y to a list of arrays if it's not already
    if not isinstance(y, list):
        y = [y]

    # Create traces
    traces = []
    if trace_names is None:
        trace_names = ['Trace {}'.format(i+1) for i in range(len(y))]

    for i, name in enumerate(trace_names):
        trace = go.Scatter(x=x, y=y[i], mode='lines', name=name)
        traces.append(trace)

    # Create layout
    layout = go.Layout(title=title, xaxis=dict(title=xlabel), yaxis=dict(title=ylabel))

    # Create figure
    fig = go.Figure(data=traces, layout=layout)

    # Display the figure
    fig.show()