from pathlib import Path

import pandas as pd

from space_traj_opt.postprocessing.extract_string import extract_parameterized_string


class CSVClient:
    """Read simulation channels from a CSV file."""

    def __init__(self, out: Path): 
        """ Initialize the client and load the CSV file. 
        The client name is derived from the CSV filename. 
        For example, `my_outputs/traj.csv` has client name `traj`. 
        """ 
        self.out = Path(out) 
        self.name = self.out.stem 
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
        channels = extract_parameterized_string(channels)

        for channel in channels:
            if channel not in self.df.columns:
                raise ValueError(f"Channel '{channel}' not found in CSV file.")

        return self.df[channels].values

    def get_time(self):
        """Return the time channel as a NumPy array."""
        if "time" not in self.df.columns:
            raise ValueError("Time column not found in CSV file.")

        return self.df["time"].values

    def get_vector(self, vector: str):
        """
        Return a 3-element vector from indexed CSV channels.

        Parameters
        ----------
        vector : str
            Base channel name. Channels are expected to be named
            '<vector>_0', '<vector>_1', and '<vector>_2'.
        """
        channels = extract_parameterized_string(vector + "_<0-2>")
        return self.get_channels(channels)
