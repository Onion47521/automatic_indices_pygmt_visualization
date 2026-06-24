import json
from osgeo import gdal
from show_band import render_map_pygmt
import numpy as np
import os
import gc

gdal.UseExceptions()

# Dictionary storing band positions inside Sentinel-2 L2A subdatasets.
# subdataset_idx: 0 -> 10m, 1 -> 20m, 2 -> 60m
# band_idx: position in the subdataset, numbered from 1 due to GetRasterBand.
#   10m (sub 0): B4=1, B3=2, B2=3, B8=4,
#   20m (sub 1): B5=1, B6=2, B7=3, B8A=4, B11=5, B12=6,
#   60m (sub 2): B1=1, B9=2
BANDS_INFO = {
    "B01": {"resolution": 60, "subdataset_idx": 2, "band_idx": 1},
    "B02": {"resolution": 10, "subdataset_idx": 0, "band_idx": 3},
    "B03": {"resolution": 10, "subdataset_idx": 0, "band_idx": 2},
    "B04": {"resolution": 10, "subdataset_idx": 0, "band_idx": 1},
    "B05": {"resolution": 20, "subdataset_idx": 1, "band_idx": 1},
    "B06": {"resolution": 20, "subdataset_idx": 1, "band_idx": 2},
    "B07": {"resolution": 20, "subdataset_idx": 1, "band_idx": 3},
    "B08": {"resolution": 10, "subdataset_idx": 0, "band_idx": 4},
    "B8A": {"resolution": 20, "subdataset_idx": 1, "band_idx": 4},
    "B09": {"resolution": 60, "subdataset_idx": 2, "band_idx": 2},
    "B11": {"resolution": 20, "subdataset_idx": 1, "band_idx": 5},
    "B12": {"resolution": 20, "subdataset_idx": 1, "band_idx": 6},
}


def load_required_bands(product_path: str, index_name: str, json_path: str = "indices.json"):
    """
    Reads JSON, determines which bands are needed and loads only them.
    Returns:
        loaded_bands (dict):        {band_name: gdal.Band}, ready for calculations
        opened_subdatasets (dict):  {subdataset_idx: gdal.Dataset}, opened JP2/VRT files
        root_dataset (gdal.Dataset): product root dataset — keep reference,
                                    so GDAL doesn't free subdatasets via GC
    """
    with open(json_path, "r", encoding="utf-8") as f:
        indices = json.load(f)

    if index_name not in indices:
        raise ValueError(f"Index '{index_name}' does not exist in {json_path}")

    required_bands = indices[index_name]["required_bands"]
    print(f"\n[MENU] Selected index: '{index_name}'. Required bands: {required_bands}")

    # check before opening anything
    unknown = [b for b in required_bands if b not in BANDS_INFO]
    if unknown:
        raise ValueError(f"Unknown bands in the definition of index '{index_name}': {unknown}")

    root_dataset = gdal.Open(product_path)
    if root_dataset is None:
        raise RuntimeError(f"GDAL could not open product: {product_path}")

    subdatasets = root_dataset.GetSubDatasets()

    opened_subdatasets = {} # cache for opened datasets
    loaded_bands = {} # bands ready for calculations

    for band_name in required_bands:
        meta = BANDS_INFO[band_name]
        sub_idx = meta["subdataset_idx"]
        band_idx = meta["band_idx"] 

        if sub_idx not in opened_subdatasets:
            sub_path = subdatasets[sub_idx][0]
            print(f"- Opening subdataset [{sub_idx}] ({meta['resolution']}m): {sub_path}")
            ds = gdal.Open(sub_path)
            if ds is None:
                raise RuntimeError(f"GDAL could not open subdataset: {sub_path}")
            opened_subdatasets[sub_idx] = ds

        ds = opened_subdatasets[sub_idx]
        loaded_bands[band_name] = ds.GetRasterBand(band_idx)
        print(f"- Successfully loaded band")

    return loaded_bands, opened_subdatasets, root_dataset


def print_band_metadata(band_name: str, band: gdal.Band):
    """Displays RasterBand metadata"""
    print(f"\n--- Band {band_name} metadata ---")
    print(f"  Size:         {band.XSize} x {band.YSize} px")
    print(f"  Data type:      {gdal.GetDataTypeName(band.DataType)}")
    print(f"  NoData:         {band.GetNoDataValue()}")
    print(f"  Scale/Offset:   scale={band.GetScale()}, offset={band.GetOffset()}")

    metadata = band.GetMetadata()
    if metadata:
        print("  GDAL Metadata:")
        for k, v in metadata.items():
            print(f"    {k}: {v}")
    else:
        print("  GDAL Metadata: none")


def prepare_arrays_for_calc(required_bands: list[str], opened_subdatasets: dict[int, gdal.Dataset]):
    """
    Gets required bands from opened subdatasets, aligns their resolution 
    to the highest one (e.g. from 20m to 10m) and returns a dictionary with ready numpy arrays.
    """
    # finding the highest spatial resolution
    target_resolution = min(BANDS_INFO[b]["resolution"] for b in required_bands)
    
    ref_sub_idx = next(BANDS_INFO[b]["subdataset_idx"] for b in required_bands if BANDS_INFO[b]["resolution"] == target_resolution)
    ref_ds = opened_subdatasets[ref_sub_idx]
    
    # Get target dimensions from the reference dataset
    target_x_size = ref_ds.RasterXSize
    target_y_size = ref_ds.RasterYSize
    
    print(f"\n[RESAMPLING] Target resolution: {target_resolution}m ({target_x_size}x{target_y_size} px)")
    
    arrays = {}
    
    for band_name in required_bands:
        meta = BANDS_INFO[band_name]
        sub_idx = meta["subdataset_idx"]
        band_idx = meta["band_idx"]
        
        # If band resolution matches, read it directly
        if meta["resolution"] == target_resolution:
            print(f"- {band_name} ({meta['resolution']}m): Resampling skipped, loading to numpy...")
            band = opened_subdatasets[sub_idx].GetRasterBand(band_idx)
            arrays[band_name] = band.ReadAsArray()
        else:
            # If resolution is different, resample on the fly to memory
            print(f"- {band_name} ({meta['resolution']}m): Resampling to {target_resolution}m...")
            
            warped_ds = gdal.Warp(
                '', # Empty string means in-memory file
                opened_subdatasets[sub_idx],
                format='MEM',
                width=target_x_size,
                height=target_y_size,
                srcBands=[band_idx],
                resampleAlg=gdal.GRA_Bilinear # conversion method
            )
            
            if warped_ds is None:
                raise RuntimeError(f"Error during resampling of band {band_name}")
            
            # dataset now has only 1 band, load it to numpy
            arrays[band_name] = warped_ds.GetRasterBand(1).ReadAsArray()
            
            # memory cleanup
            warped_ds = None 
            
    return arrays

def clip_dataset_to_vector(dataset: gdal.Dataset, vector_path: str):
    """
    Clips input gdal.Dataset object to vector boundaries
    """
    print(f"- Clipping dataset to vector: {vector_path}")
    
    raster_crs = dataset.GetProjection()

    clipped_ds = gdal.Warp(
        '', 
        dataset,
        format='MEM',
        cutlineDSName=vector_path,
        cropToCutline=True,
        dstSRS=raster_crs,
        srcNodata=0,
    )
    
    if clipped_ds is None:
        raise RuntimeError(f"GDAL Error: Failed to clip raster using vector {vector_path}. Check if the vector file is valid.")
        
    return clipped_ds

def calculate_index_numpy(calculation_string: str, arrays: dict[str, np.ndarray]):
    """
    Calculates spectral index based on the formula from JSON.
    Includes correction for Sentinel-2 radiometric offset +1000
    """
    print(f"\n[CALCULATION] Calculating index from formula: {calculation_string}")
    
    allowed_namespace = {"__builtins__": {}, "np": np} 
    float_arrays = {}
    
    for name, array in arrays.items():
        array_float = array.astype(np.float32)
        # Edge isolation (Sentinel's NoData is 0 in raw files)
        array_float[array == 0] = np.nan

        array_float = array_float - 1000.0
        
        float_arrays[name] = array_float
        
    allowed_namespace.update(float_arrays)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        result_array = eval(calculation_string, allowed_namespace) 
        result_array[np.isinf(result_array)] = np.nan
        
    return result_array

def compress_raster_for_display(array: np.ndarray, reference_ds: gdal.Dataset, level: int = 4):
    """
    Reduces the spatial resolution of the array via NN.
    Significantly speeds up PyGMT operations and optimizes memory.
    Level determines the reduction factor, e.g., level=4 means 4 times fewer pixels per axis.
    """
    if level <= 1: # no compression
        return array, reference_ds
        
    print(f"\n[OPTIMIZATION] Array downsampling (NearestNeighbour, reduction: {level}x)...")
    
    rows, cols = array.shape
    driver = gdal.GetDriverByName('MEM') # loads from memory
    
    # Create temporary dataset in memory with original data
    temp_ds = driver.Create('', cols, rows, 1, gdal.GDT_Float32)
    temp_ds.SetGeoTransform(reference_ds.GetGeoTransform())
    temp_ds.SetProjection(reference_ds.GetProjection())
    
    band = temp_ds.GetRasterBand(1)
    band.WriteArray(array)
    band.SetNoDataValue(np.nan)
    
    # Calculate new dimensions (minimum 1x1 pixel)
    new_cols = max(1, cols // level)
    new_rows = max(1, rows // level)
    
    # compression 
    compressed_ds = gdal.Warp(
        '', temp_ds, 
        format='MEM', 
        width=new_cols, 
        height=new_rows, 
        resampleAlg=gdal.GRA_NearestNeighbour,
        outputType=gdal.GDT_Float32
    )
    
    new_array = compressed_ds.GetRasterBand(1).ReadAsArray()
    
    # memory cleanup
    temp_ds = None
    
    return new_array, compressed_ds



if __name__ == "__main__":

    import pygmt
    
    product = "S2A_MSIL2A_20220803T100041_N0510_R122_T33UXS_20240719T061631.zip" # random fragment I tested on
    #it's not included on Github bcs it's large, but project should work on any Sentinel-2 L2A product
    
    json_path = "indices.json"
    
    vector_path = "boundary.gpkg" # random boundary used in project, also not included
    # vector_path = "" # empty str = no clipping

    # selected_indices = ["NDVI", "NDBI", "NDMI", "LCI"]
    selected_indices = ["NDVI", "LCI"]

    print("PROGRAM START")

    with open(json_path, "r", encoding="utf-8") as f:
        indices_database = json.load(f)

    # List to store ready maps to display at the end
    ready_maps = []

    for selected_index in selected_indices:
        print("\n" + "="*40)
        print(f"STARTING PROCESSING: {selected_index}")
        
        if selected_index not in indices_database:
            print(f"[ERROR] Index {selected_index} does not exist in JSON. Skipping.")
            continue

        required_bands = indices_database[selected_index]["required_bands"]
        json_formula = indices_database[selected_index]["calculation"]

        # loading and clipping
        bands, subdatasets, root_ds = load_required_bands(product, selected_index, json_path)
        if vector_path:
            for sub_idx, ds in subdatasets.items():
                subdatasets[sub_idx] = clip_dataset_to_vector(ds, vector_path)

        # resolution alignment and numpy calculations
        arrays = prepare_arrays_for_calc(required_bands, subdatasets)
        result_array = calculate_index_numpy(json_formula, arrays)

        # removing unused elements from ram
        arrays.clear()
        bands.clear()
        gc.collect()

        # finding the highest spatial resolution so as not to ruin compression
        target_resolution = min(BANDS_INFO[b]["resolution"] for b in required_bands)
        ref_sub_idx = next(BANDS_INFO[b]["subdataset_idx"] for b in required_bands if BANDS_INFO[b]["resolution"] == target_resolution)
        
        # nearestneighbour compression
        comp_array, comp_ds = compress_raster_for_display(result_array, subdatasets[ref_sub_idx], level=4)

        # next memory cleanup
        del result_array
        subdatasets.clear()
        root_ds = None
        gc.collect()

        
        
        # pygmt generation (if there is no color in json, gives base gmt/gray)
        selected_color = indices_database[selected_index].get("color", "gmt/gray")
        fig = pygmt.Figure()
        render_map_pygmt(fig, comp_array, comp_ds, selected_index, selected_color)
        ready_maps.append((selected_index, fig))

        # last cleanup in processing one index, only the map stays in memory
        comp_array = None
        comp_ds = None
        gc.collect()
        
        print(f"[{selected_index}] Processing finished.")

    # displaying created maps
    print("\n[FINISHED] Launching system viewer with PyGMT maps...")
    for name, map_fig in ready_maps:
        print(f"- Displaying map: {name}")
        map_fig.show()