import json
import warnings
from pathlib import Path

import pandas as pd

from space_traj_opt import OUT_DIR
from space_traj_opt.postprocessing.extract_string import extract_parameterized_string


class CSVClient:
    """Read simulation channels from a CSV file."""

    def __init__(self, sim_output: str):
        """Initialize the client and load the CSV file.

        The client name is derived from the CSV filename. For example,
        `my_outputs/traj.csv` has client name `traj`.

        Args:
            sim_output (str): Name of the simulation output, used to construct the CSV file path.
        """
        self.out: Path = OUT_DIR / (sim_output + ".csv")
        self.name: str = self.out.stem
        self.metadata_path: Path = self.out.with_suffix(".json")
        self.metadata = None

        if self.metadata_path.exists():
            try:
                with self.metadata_path.open("r", encoding="utf-8") as metadata_file:
                    self.metadata = json.load(metadata_file)
            except (json.JSONDecodeError, OSError):
                warnings.warn(
                    f"Could not load metadata file: {self.metadata_path}",
                    UserWarning                )
                self.metadata = None
        else:
            warnings.warn(
                f"Metadata file not found for '{sim_output}': {self.metadata_path}",
                UserWarning            )

        self.df = pd.read_csv(self.out)

    def get_channels(self, channels: str | list[str]):
        """
        Return the requested channels as a NumPy array.

        Parameters
        ----------
        channels : str or list[str]
            Channel name or list of channel names. Each name must
            exactly match a column in the CSV file.
        """
        channels : list[str] = extract_parameterized_string(channels)

        for channel in channels:
            if channel not in self.df.columns:
                raise ValueError(f"Channel '{channel}' not found in CSV file.")

        return self.df[channels].values.squeeze()

    def get_time(self):
        """Return the time channel as a NumPy array."""
        if "time" not in self.df.columns:
            raise ValueError("Time column not found in CSV file.")

        return self.df["time"].values.squeeze()


