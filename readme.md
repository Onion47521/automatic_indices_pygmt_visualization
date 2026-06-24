# l-2 Data Processing and Visualization:
This project is a set of Python scripts for automated loading, analysis, and visualization of satellite data from the Sentinel-2 mission (L2A level).
The tool calculates spectral indices based on dynamically loaded formulas and generates high-quality maps using the PyGMT library.

## Main Feature:
- **Automatic resolution alignment:** The script automatically equalizes spatial resolution (e.g., from 20m to 10m) using bilinear interpolation, fetching only the required bands from subdatasets.
- **Index calculation (JSON):** Spectral indices (e.g., NDVI, LCI) are calculated using the `NumPy` library based on the configuration loaded from the `indices.json` file.
- **Vector clipping:** Optional clipping of rasters to an area of interest (e.g., a `.gpkg` file).
- **PyGMT visualization:** Rendering the final raster into an aesthetic map in vector/raster format, ready for preview or printing, using the optimized `show_band.py` module.
    

## Requirements
The project requires Python 3.x and the following libraries (preferably installed via the `conda` package manager due to system dependencies like GDAL):
- `gdal` (osgeo)
- `numpy`
- `xarray`
- `pygmt`
    

## File Structure
- `sentinel_processor.py` – The main script managing the core logic, GDAL formats handling, optimization (downsampling), memory management, and formula loading.
- `show_band.py` – An external module containing the `render_map_pygmt` function, which uses Xarray and PyGMT packages to display a formatted map (scale, legend, coordinate system).   
- `indices.json` – The indices database, which must contain information about the required bands, the mathematical formula, and (optionally) a CPT color palette.
    

## How to use
Currently, basic configuration is done by editing variables at the end of the main `sentinel_processor.py` file in the `if __name__ == "__main__":` block. You need to specify there:
1. The path to the Sentinel-2 product (e.g., a `.zip` file).
2. The path to the vector file (or leave it empty to skip clipping).
3. The name of the index to calculate (e.g., `["NDVI"]`), which must match a key from the `indices.json` file.
    

Then, simply run the script:

```
python sentinel_processor.py
```

## Future Plans 
The current way of running the script (editing the code directly) is meant for development purposes. **In the future, a form of interaction with the program will be added, allowing work without the need to edit the source code.** Planned feature includes interactive GUI or CLI approach (like in GDAL)