from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from PIL import Image

EARTH_RADIUS = 6_378_137.0  # [m]


class OrbitPlot:
    """
    3D Earth/orbit visualization in the ECI frame.

    Parameters
    ----------
    earth_texture : str | Path
        Path to an equirectangular Earth image:
            x-axis -> longitude [-180, 180]
            y-axis -> latitude  [90, -90]

    epoch_rotation : float
        Rotation of Earth about the ECI +Z axis [rad].

        This can be the Greenwich sidereal angle / Earth rotation angle
        corresponding to the desired epoch.

    earth_radius : float
        Earth radius [m].

    resolution : int
        Number of latitude/longitude samples used for the Earth sphere.
    """

    def __init__(
        self,
        earth_texture: str | Path = Path(__file__).parent / "earth_texture.jpg",
        epoch_rotation: float = 0.0,
        earth_radius: float = EARTH_RADIUS,
        resolution: int = 180,
    ):
        self.earth_radius = earth_radius
        self.epoch_rotation = epoch_rotation

        self.fig = go.Figure()

        self._add_earth(
            Path(earth_texture),
            resolution,
        )

        self._configure_layout()

    # ------------------------------------------------------------------
    # Earth
    # ------------------------------------------------------------------

    def _add_earth(
        self,
        texture_path: Path,
        resolution: int,
    ) -> None:

        image = Image.open(texture_path).convert("RGB")

        # Downsample texture to something reasonable for Plotly.
        image = image.resize(
            (2 * resolution, resolution),
            Image.Resampling.LANCZOS,
        )

        texture = np.asarray(image)

        n_lat, n_lon, _ = texture.shape

        lat = np.linspace(
            np.pi / 2,
            -np.pi / 2,
            n_lat,
        )

        lon = np.linspace(
            -np.pi,
            np.pi,
            n_lon,
        )

        lon_grid, lat_grid = np.meshgrid(lon, lat)

        # Rotate Earth from Earth-fixed longitude into ECI.
        lon_eci = lon_grid + self.epoch_rotation

        R = self.earth_radius

        x = R * np.cos(lat_grid) * np.cos(lon_eci)
        y = R * np.cos(lat_grid) * np.sin(lon_eci)
        z = R * np.sin(lat_grid)

        # Plotly Surface does not directly support RGB texture maps.
        # Convert RGB -> scalar index and supply a colorscale.
        surfacecolor, colorscale = self._texture_to_surface(texture)

        self.fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=surfacecolor,
                colorscale=colorscale,
                cmin=0,
                cmax=255,
                showscale=False,
                lighting={
                    "ambient": 0.8,
                    "diffuse": 0.8,
                    "specular": 0.15,
                    "roughness": 0.8,
                    "fresnel": 0.1,
                },
                lightposition={
                    "x": 100_000,
                    "y": 100_000,
                    "z": 100_000,
                },
                name="Earth",
                hoverinfo="skip",
            )
        )

    @staticmethod
    def _texture_to_surface(
        texture: npt.NDArray[np.uint8],
    ) -> tuple[npt.NDArray[np.float64], list]:

        # Quantize RGB image into 256 colors.
        image = Image.fromarray(texture)

        indexed = image.quantize(
            colors=256,
            method=Image.Quantize.MEDIANCUT,
        )

        surfacecolor = np.asarray(indexed, dtype=float)

        palette = np.asarray(
            indexed.getpalette(),
            dtype=np.uint8,
        ).reshape(-1, 3)

        colorscale = []

        n = len(palette)

        for i, rgb in enumerate(palette):
            r, g, b = rgb

            colorscale.append(
                [
                    i / max(n - 1, 1),
                    f"rgb({r},{g},{b})",
                ]
            )

        return surfacecolor, colorscale

    # ------------------------------------------------------------------
    # Orbit traces
    # ------------------------------------------------------------------

    def add_trace(
        self,
        position: npt.ArrayLike,
        name: str = "Orbit",
        *,
        width: float = 4,
        mode: str = "lines",
        **kwargs,
    ) -> OrbitPlot:
        """
        Add an ECI trajectory.

        Parameters
        ----------
        position : array_like, shape (N, 3)
            ECI position vectors [m].

        name : str
            Trace name.

        width : float
            Orbit line width.

        mode : str
            Plotly Scatter3d mode:
                "lines"
                "markers"
                "lines+markers"

        kwargs :
            Additional arguments passed to go.Scatter3d.

        Returns
        -------
        self
            Allows chained calls.
        """

        r = np.asarray(position)

        if r.ndim != 2 or r.shape[1] != 3:
            raise ValueError("position must have shape (N, 3)")

        line = kwargs.pop(
            "line",
            {"width": width},
        )

        self.fig.add_trace(
            go.Scatter3d(
                x=r[:, 0],
                y=r[:, 1],
                z=r[:, 2],
                mode=mode,
                name=name,
                line=line,
                **kwargs,
            )
        )

        return self

    def add_point(
        self,
        position: npt.ArrayLike,
        name: str = "Vehicle",
        size: float = 5,
        **kwargs,
    ) -> OrbitPlot:

        r = np.asarray(position)

        if r.shape != (3,):
            raise ValueError("position must have shape (3,)")

        marker = kwargs.pop(
            "marker",
            {"size": size},
        )

        self.fig.add_trace(
            go.Scatter3d(
                x=[r[0]],
                y=[r[1]],
                z=[r[2]],
                mode="markers",
                marker=marker,
                name=name,
                **kwargs,
            )
        )

        return self

    # ------------------------------------------------------------------
    # Plot configuration
    # ------------------------------------------------------------------

    def _configure_layout(self) -> None:

        R = self.earth_radius

        self.fig.update_layout(
            scene=dict(
                xaxis={
                    "title": "ECI X [m]",
                    "showbackground": False,
                    "showgrid": False,
                    "zeroline": False,
                },
                yaxis=dict(
                    title="ECI Y [m]",
                    showbackground=False,
                    showgrid=False,
                    zeroline=False,
                ),
                zaxis=dict(
                    title="ECI Z [m]",
                    showbackground=False,
                    showgrid=False,
                    zeroline=False,
                ),
                aspectmode="data",
                camera=dict(
                    eye=dict(
                        x=1.5,
                        y=1.5,
                        z=1.0,
                    )
                ),
            ),
            margin=dict(
                l=0,
                r=0,
                t=0,
                b=0,
            ),
            legend=dict(
                x=0.02,
                y=0.98,
            ),
        )

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def show(self) -> None:
        self.fig.show()

    def write_html(
        self,
        filename: str | Path,
        **kwargs,
    ) -> None:
        self.fig.write_html(
            filename,
            **kwargs,
        )
