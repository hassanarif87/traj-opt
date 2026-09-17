import argparse
from pathlib import Path

from space_traj_opt.reports.generate_report import generate_report


def main():
    parser = argparse.ArgumentParser(
        prog="topt",
        description="Generate a simulation report.",
    )

    parser.add_argument(
        "--sim",
        "-s",
        type=str,
        required=True,
        help="Simulation name.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output HTML file.",
    )

    args = parser.parse_args()

    generate_report(
        sim_name=args.sim,
        output=args.output,
    )
