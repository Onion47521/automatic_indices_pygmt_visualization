import pygmt
import xarray as xr
import numpy as np
import math as m

def render_map_pygmt(fig: pygmt.Figure, array: np.ndarray, reference_ds, index_name: str, cmap_name: str = "gmt/gray"):
    """
    Converts the array to xarray and generates a formatted map in PyGMT.
    Supports dynamic names + base GMT color scales. 
    """
    print(f"[{index_name}] Generating formatted cartography in PyGMT (Palette: {cmap_name})...")

    # We get a tuple with transformation parameters (imaging info) e.g. image corners and spatial res.
    gt = reference_ds.GetGeoTransform()
    # We get the number of columns and rows
    cols = reference_ds.RasterXSize
    rows = reference_ds.RasterYSize

    # We generate a coordinate grid
    # - np.arange(cols): generates a list of indices [0, 1, 2, ..., cols-1].
    # - +0.5 shifts the anchor point from the pixel edge to the center
    # - *gt[1] multiplies the index by the pixel width (e.g., 10 meters).
    # - gt[0] + ...: adds the X coordinate of the top-left corner, setting the correct position.
    # similarly for y but a different place in the tuple holds the info
    x = gt[0] + (np.arange(cols) + 0.5) * gt[1]
    y = gt[3] + (np.arange(rows) + 0.5) * gt[5]


    # conversion of data to xarray to be able to display in pygmt without saving the raster first
    da = xr.DataArray(
        array, # raw numpy data
        coords=[("y", y), ("x", x)],
        dims=["y", "x"],
        name=index_name,
    )

    # calculating the frame area so the raster is not on the frame
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    def_space = max(xmax - xmin, ymax - ymin) * 0.10
    # def_space = 0

    region = [
        xmin - def_space,
        xmax + def_space,
        ymin - def_space,
        ymax + def_space,
    ]

    with pygmt.config(
        FONT_TITLE="30p,Helvetica-Bold,black", 
        FONT_ANNOT_PRIMARY="12p,Helvetica,black",
        PS_IMAGE_COMPRESS="DEFLATE", # additional compression
        PS_CHAR_ENCODING="ISO-8859-2", # Supports Polish characters, used in original project 
        COLOR_BACKGROUND="white", # white background because it gives black by default
        COLOR_NAN="white", # same for nan 
        # Enlarging and bolding the scale (and rose by accident)
        MAP_SCALE_HEIGHT="8p",
        MAP_TICK_PEN_PRIMARY="2p,black"
    ):

        pygmt.makecpt(cmap=cmap_name, series=[-1, 1],background=False) # TODO: hardcoded normalized range

        # Drawing the raster
        fig.grdimage(
            grid=da,
            region=region,
            projection="X15c",
            cmap=True
        )

        fig.basemap(
            frame=[
                f"WSne+tIndex {index_name}", 
                "afg+u m",], 
            rose="JRM+w2.0c+f1+lW,E,S,N+o0.5c/4.5c",
            map_scale="jBL+w10000+lkm+o0.5c/0.7c" # TODO: mapscale hardcoded, should scale
        )

        fig.colorbar(
            position="JRM+o1.8c/-1.5c+w8c/0.5c",
            frame=["af", f"x+lIndex value {index_name}"], 
        )