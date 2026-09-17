from pathlib import Path

import yaml
from space_traj_opt.postprocessing.data_client import CSVClient

from space_traj_opt.postprocessing.plotting import plot
from space_traj_opt import TEMPLATES_DIR

class TemplatePlotter:
    """Generate Plotly figures from a YAML plotting template."""

    def __init__(self, template: str, clients: list[CSVClient]):
        """
        Initialize the plotter.

        Parameters
        ----------
        template : Path or str
            Path to the YAML plotting template.
        clients : list[CSVClient]
            CSV clients containing the channel data to plot.
            Each client must have a `name` attribute.
        """
        self.template = TEMPLATES_DIR / (template + ".yaml")
        self.clients = clients

        with self.template.open("r") as f:
            self.config = yaml.safe_load(f)

    def plot(self, output: Path | str | None = None):
        """
        Generate all figures defined in the template.

        Parameters
        ----------
        output : Path or str, optional
            If provided, save the figures to an HTML file.
            Otherwise, display the figures in the notebook.

        Returns
        -------
        list
            List of generated Plotly figures.
        """
        figures = []

        for config in self.config["figures"]:
            fig = self._create_figure(config)
            figures.append(fig)

        if output is not None:
            self._save_html(figures, Path(output))
        else:
            for fig in figures:
                fig.show()

    def _create_figure(self, config):
        """Create a single figure from a template entry."""
        x_channel = config["xdata"]
        y_channels = config["ydata"]

        x_data = []
        y_data = []
        trace_names = []
        phases = []

        for client in self.clients:
            x = self._get_channel(client, x_channel)

            for channel in y_channels:
                y = self._get_channel(client, channel)

                x_data.append(x)
                y_data.append(y)
                trace_names.append(f"{channel}.{client.name}")
                phase = self._get_channel(client, "phase") if "phase" in client.df.columns else None
                phases.append(phase)

        return plot(
            x=x_data,
            y=y_data,
            title=config["name"],
            phases=phases,
            trace_names=trace_names,
        )

    @staticmethod
    def _get_channel(client, channel: str):
        """Get a channel from a CSV client."""
        if channel == "time":
            return client.get_time()

        return client.get_channels(channel)

    @staticmethod
    def save_html(figures, output: Path):
        """Save all generated figures to an HTML file."""
        import plotly.io as pio

        with output.open("w") as f:
            for i, fig in enumerate(figures):
                f.write(f"<h2>{fig.layout.title.text or f'Figure {i}'}</h2>\n")
                f.write(
                    pio.to_html(
                        fig,
                        full_html=False,
                        include_plotlyjs="cdn",
                    )
                )