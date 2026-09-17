import plotly.graph_objects as go
#from plotly.graph_objects.Figure import add_trace
import numpy as np
import matplotlib.pyplot as plt

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
        trace_names = ['Trace {}'.format(i + 1) for i in range(len(y) + (len(y2) if y2 else 0))]

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
        xaxis=dict(title=xlabel),
        yaxis=dict(title=y1_label),
        yaxis2=dict(title=y2_label, overlaying='y', side='right', showgrid=False) if y2 else None,
        template="plotly"
    )

    # Create and display figure
    fig = go.Figure(data=traces, layout=layout)
    fig.show()
    #return fig

def visualize_jac2(x0, jacobian):
    """
    Visualize the Jacobian matrix as a heatmap, highlighting non-zero entries.

    This function creates a heatmap representation of a Jacobian matrix with non-zero entries
    displayed using a colormap and zero or near-zero entries explicitly masked in white.

    Args:
        x0 : array-like
            The point at which the Jacobian is evaluated. Used to label the variables on the x-axis.
        jacobian : ndarray
            The Jacobian matrix to visualize, where each element represents the partial derivative
            of a function with respect to a variable.
    Notes
    -----
    - Near-zero values (absolute values below 1e-10) are treated as zero and explicitly set to NaN for masking.
    - The heatmap includes labels for variables (`x1`, `x2`, ...) on the x-axis and functions
      (`f1`, `f2`, ...) on the y-axis.
    """
    # Create a mask for near-zero entries
    mask = np.abs(jacobian) > 1e-10  # Adjust threshold as needed
    display_jacobian = np.where(mask, jacobian, np.nan)  # Replace zeros with NaN for masking

    # Define axis labels
    x_labels = [f'x{i+1}' for i in range(len(x0))]
    y_labels = [f'f{i+1}' for i in range(jacobian.shape[0])][::-1]  # Reverse for top-down order

    # Reverse rows of the Jacobian for proper orientation
    display_jacobian = display_jacobian[::-1]

    # Create the heatmap
    heatmap = go.Heatmap(
        z=display_jacobian,
        x=x_labels,
        y=y_labels,
        colorscale='RdBu',
        colorbar=dict(title='Derivative Value'),
        hoverongaps=False,
        zmin=-np.nanmax(np.abs(display_jacobian)),  # Symmetric colormap range
        zmax=np.nanmax(np.abs(display_jacobian)),
    )

    # Add annotations for masked (zero) values
    annotations = []
    for i in range(jacobian.shape[0]):
        for j in range(jacobian.shape[1]):
            if not mask[i, j]:  # Zero or near-zero values
                annotations.append(
                    dict(
                        x=x_labels[j],
                        y=y_labels[-(i + 1)],  # Adjust for reversed order
                        text="0",
                        showarrow=False,
                        font=dict(color="black"),
                        xref="x",
                        yref="y",
                    )
                )

    # Create the figure
    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        title='Jacobian Heatmap (Non-Zero Entries)',
        xaxis=dict(title='Variables'),
        yaxis=dict(title='Functions'),
        annotations=annotations,
    )

    # Show the plot
    fig.show()
    
def visualize_jac(x0, jacobian):
    # Create a mask for non-zero entries
    mask = np.abs(jacobian) > 1e-10  # Adjust threshold as needed for small values

    # Visualization using a heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(np.where(mask, jacobian, np.nan), cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Derivative value')
    plt.title('Jacobian Heatmap (Non-Zero Entries)')
    plt.xlabel('Variables')
    plt.ylabel('Functions')
    plt.xticks(range(len(x0)), [f'x{i+1}' for i in range(len(x0))])
    plt.yticks(range(jacobian.shape[0]), [f'f{i+1}' for i in range(jacobian.shape[0])])

    # Mask zero entries explicitly by overlaying white patches
    for i in range(jacobian.shape[0]):
        for j in range(jacobian.shape[1]):
            if not mask[i, j]:  # Zero or near-zero values
                plt.gca().add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='white', ec=None))

    plt.show()