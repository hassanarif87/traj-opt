import plotly.graph_objects as go


def plot(x, y, x2=None, y2=None, xlabel=None, ylabel=None, title=None, trace_names=None):
    """Generate a Plotly plot with support for a secondary y-axis.
    Args:
        x: List or array-like, x values.
        y: Array-like or list of array-like, y values for the primary y-axis.
        y2: (Optional) Array-like or list of array-like, y values for the secondary y-axis.
        xlabel: (Optional) String, label for the x-axis.
        ylabel: (Optional) Tuple of strings or a single string, labels for y-axes. 
                If tuple, applies to both y1 and y2; if single string, applies to y1 only.
        title: (Optional) String, title of the plot.
        trace_names: (Optional) List of strings, names for each trace. Default is None.

    Returns:
        fig: Plotly Figure (displays the plot).
    """


    # Convert y to a list of arrays if it's not already
    if not isinstance(y, list):
        y = [y]
    
     # Convert x to a list of arrays if it's not already
    if not isinstance(x, list):
        x = [x]*len(y)

    # Convert y2 to a list of arrays if it's provided
    if y2 is not None and not isinstance(y2, list):
        y2 = [y2]
    
    # Use the primary x 
    if x2 is None:
        x2=x

    # Set default trace names if none are provided
    if trace_names is None:
        trace_names = [f'Trace {i + 1}' for i in range(len(y) + (len(y2) if y2 else 0))]

    # Create traces for primary y-axis
    traces = []
    for i, (data_x, data) in enumerate(zip(x,y)):
        trace = go.Scatter(x=data_x, y=data, mode='lines', name=trace_names[i], yaxis="y1")
        traces.append(trace)

    # Create traces for secondary y-axis if y2 data is provided
    if y2:
        for i, (data_x, data) in enumerate(zip(x2,y2), start=len(y)):
            trace = go.Scatter(x=data_x, y=data, mode='lines', name=trace_names[i], yaxis="y2", line=dict(dash='dash'))
            traces.append(trace)

    # Set labels for y-axis based on ylabel parameter
    if isinstance(ylabel, tuple):
        y1_label, y2_label = ylabel
    else:
        y1_label, y2_label = ylabel, None

    # Define layout with secondary y-axis if y2 data is provided
    layout = go.Layout(
        title=title,
        xaxis={'title': xlabel},
        yaxis={'title': y1_label},
        yaxis2={'title': y2_label, 'overlaying': 'y', 'side': 'right', 'showgrid': False} if y2 else None,
        template="plotly"
    )

    # Create and display figure
    fig = go.Figure(data=traces, layout=layout)
    fig.show()
    #return fig
