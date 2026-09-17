import plotly.graph_objects as go
import plotly.colors


PHASE_LINE_STYLES = [
    "solid",
    "dot"]

#     "longdash",
#     "dash",
#     "longdashdot",
#     "dashdot",
# ]


def plot(
    x,
    y,
    x2=None,
    y2=None,
    xlabel=None,
    ylabel=None,
    title=None,
    trace_names=None,
    phases=None,
):
    """Generate a Plotly plot with phase-based line styles."""

    fig = go.Figure()

    if trace_names is None:
        trace_names = [None] * len(y)

    colors = plotly.colors.qualitative.Plotly

    for i, (xi, yi, name) in enumerate(zip(x, y, trace_names)):

        # One color for the entire logical trace
        trace_color = colors[i % len(colors)]

        if phases is None:
            fig.add_trace(
                go.Scatter(
                    x=xi,
                    y=yi,
                    mode="lines",
                    name=name,
                    line=dict(color=trace_color),
                )
            )
            continue

        phase = phases[i]
        unique_phases = sorted(set(phase))

        for j, phase_id in enumerate(unique_phases):
            indices = (phase == phase_id).nonzero()[0]

            if len(indices) == 0:
                continue

            # Keep the boundary point so the line remains continuous
            if indices[0] > 0:
                indices = [indices[0] - 1, *indices]

            fig.add_trace(
                go.Scatter(
                    x=xi[indices],
                    y=yi[indices],
                    mode="lines",
                    name=name,
                    legendgroup=name,
                    showlegend=(j == 0),
                    line={
                        "color": trace_color,
                        "dash": PHASE_LINE_STYLES[
                            j % len(PHASE_LINE_STYLES)
                        ],
                    },
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
    )

    if ylabel is not None:
        fig.update_yaxes(title_text=ylabel)

    return fig
