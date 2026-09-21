import argparse
from pathlib import Path

from space_traj_opt.reports.generate_report import generate_report
from space_traj_opt.sims.scenario import run_scenario_file


def handle_run(args):
    print(f"Running simulation for: {args.scenario}")
    result = run_scenario_file(args.scenario)
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")

def handle_report(args):
    print(f"Generating report for: {args.scenario}")
    generate_report(
        sim_name=args.scenario,
        output=args.output,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="topt",
        description="Generate a simulation report.",
    )

    subparsers = parser.add_subparsers(
        dest="sub_commands", 
        required=True, 
        help="Subcommands")

    parser_run = subparsers.add_parser("run", help="Run a simulation")
    parser_run.add_argument(
        "--scenario",
        "-s",
        type=str,
        required=True,
        help="Simulation name.",
    )
    parser_run.set_defaults(func=handle_run)

    parser_report = subparsers.add_parser("report", help="Generate a simulation report")
    parser_report.add_argument(
        "--scenario",
        "-s",
        type=str,
        required=True,
        help="Simulation name.",
    )
    parser_report.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output HTML file.",
    )
    parser_report.set_defaults(func=handle_report)
    args = parser.parse_args()
    args.func(args)
