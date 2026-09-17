from space_traj_opt import OUT_DIR
from space_traj_opt.postprocessing.data_client import CSVClient
from space_traj_opt.postprocessing.template_plotter import TemplatePlotter


def generate_report(sim_name: str | list[str], output: str | None = None) -> None:

    """Generate a report for the given simulation(s).
    Args:
        sim_name: Name(s) of the simulation(s) to generate the report for.
        output : Optional output filename for the report. If not provided, defaults to "<sim_name>_report.html".
    """

    if isinstance(sim_name, str):
        sim_name = [sim_name]
    clients = [CSVClient(sim_output=sim) for sim in sim_name]

    if output is None:
        output = "report"
    TemplatePlotter("plot_report", clients).plot(OUT_DIR /  (output + ".html"))
