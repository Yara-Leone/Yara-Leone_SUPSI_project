# "Yara-Leone_SUPSI_project" 
# Remote Sensing Workflows

This repository contains Python scripts developed for the Project of internship for the supplementary geomatics certificate,
focusing on remote sensing analysis of:

- Land and Lake Surface Temperature (LST)
- Chlorophyll-a (Chl-a)

using **Sentinel-3** and **Landsat 8–9** satellite data.

## Data sources

- Copernicus Data Space Ecosystem (Sentinel-3 OLCI / SLSTR)
- USGS - Google Earth Engine STAC (Landsat Collection 2)
- SNAP / C2RCC processor

## Python files
File  :   [inputs]    [outputs]
- 1_1_landsat_GEEforCIPAIS.py : [SIMILE_laghi_UTM32.shp] [landsat8-9_L2SP, landsat_lugano_CIPAIS_inventory_2024.xlsx]
- 1_2_landsat_comparisonsmaps.py : [temp_reference_images_2024, landsat8-9_L2SP] [landsat8-9_L2SP]
- 2_1_download_OLCI1EFR.py : [lac_lugano.shp] [sentinel3b_l1efr, sentinel3_olci_luganoforCIPAIS_inventory.xlsx]
- 2_2_processing_OLCI1EFR.py : [SIMILE_laghi_UTM32.shp, sentinel3b_l1efr, temp_observed_estimated_2024.xlsx] [sentinel3b_l1efr]
- 2_3_OL1EFR_comparisonmaps.py : [sentinel3b_l1efr, chla_reference_images_2024 folder] [sentinel3b_l1efr]
  
- 3_1_heatislands_landsat_GEEdownload.py : [Tessin_AOI.shp] [landsat_lst_ticino, landsat_lst_usgs_inventory.xlsx, landsat_lst_GEE_downloaded.xlsx]
- 3_2_heatislands_sen3_download.py : [Tessin_AOI.shp] [sentinel3_lst_ticino, sentinel3_lst_copernicus_inventory.xlsx]

All required input datasets are provided as a compressed archive (`inputs.zip`) located in the `inputs/` directory.
Before running the scripts, the archive must be extracted so that the original directory structure is preserved.

## Structure 
### (CIPAIS project)
- `1_1_landsat_GEEforCIPAIS.py/`
Landsat thermal products are retrieved via STAC and processed to derive land surface temperature using the Barsi atmospheric correction

- `2_1_download_OLCI1EFR.py/` & `2_2_processing_OLCI1EFR.py`
Sentinel-3 OLCI Level-1 EFR products are retrieved via STAC and processed for chlorophyll-a using the C2RCC algorithm following the SIMILE protocol

- `1_2_landsat_comparisonsmaps.py/` & `2_3_OL1EFR_comparisonmaps.py`
The processed images are compared with reference images from 2024 in order to validate the processing workflow.

### (heat islands project)
- `3_1_heatislands_landsat_GEEdownload.py`
STAC-based inventory and data download

- `3_2_heatislands_sen3_download.py`
STAC-based inventory and data download 

## Requirements

### Python environment
- Python 3.9

### Python libraries
The following Python libraries are required:

- numpy
- pandas
- geopandas
- shapely
- rasterio
- pyproj
- fiona
- requests
- matplotlib
- openpyxl
- earthengine-api
- geemap

### External software
Some scripts require additional external software:

- ESA SNAP (tested with SNAP 9)
- SNAP Python API (snappy for Sentinel-3 OLCI processing)
- Java JDK 1.8.0

### Accounts & access
- Google Earth Engine account (requires a Google account)
- Copernicus Data Space Ecosystem account (see report for Yara Leone's access or create one)
- USGS / Landsat STAC access (public)


## Author

Yara Leone dos Santos
SUPSI — Project of internship for the supplementary geomatics certificate 
12.2025
