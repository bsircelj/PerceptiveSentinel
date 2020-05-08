from eolearn.io import S1IWWCSInput

from eolearn.core import LinearWorkflow, FeatureType, OverwritePermission, \
    SaveTask, EOExecutor
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import os
from sentinelhub import BBoxSplitter, CRS


def generate_slo_shapefile(path):
    DATA_FOLDER = os.path.join('data')

    area = gpd.read_file(os.path.join(DATA_FOLDER, 'svn_buffered.geojson'))
    # area = gpd.read_file(os.path.join(DATA_FOLDER, 'austria.geojson'))

    # Convert CRS to UTM_33N
    country_crs = CRS.UTM_33N
    area = area.to_crs(crs={'init': CRS.ogc_string(country_crs)})

    # Get the country's shape in polygon format
    country_shape = area.geometry.values.tolist()[-1]

    # Plot country
    plt.axis('off');

    # Create the splitter to obtain a list of bboxes
    bbox_splitter = BBoxSplitter([country_shape], country_crs, (25 * 2, 17 * 2))

    bbox_list = np.array(bbox_splitter.get_bbox_list())
    info_list = np.array(bbox_splitter.get_info_list())

    path_out = path + '/shapefiles'
    if not os.path.isdir(path_out):
        os.makedirs(path_out)

    geometry = [Polygon(bbox.get_polygon()) for bbox in bbox_list]
    idxs_x = [info['index_x'] for info in info_list]
    idxs_y = [info['index_y'] for info in info_list]

    gdf = gpd.GeoDataFrame({'index_x': idxs_x, 'index_y': idxs_y},
                           crs={'init': CRS.ogc_string(country_crs)},
                           geometry=geometry)

    shapefile_name = path_out + '/slovenia.shp'
    gdf.to_file(shapefile_name)

    return gdf, bbox_list


def download_patches(path, bbox_list, indexes):
    save = SaveTask(path, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

    add_iw = S1IWWCSInput(layer='BANDS-S1-IW',
                          feature=(FeatureType.DATA, 'IW'),
                          resx=10,
                          resy=10,
                          orbit='both',
                          instance_id='f9ba0e92-42b3-4483-a4b4-a08687e5232e')

    time_interval = ['2017-01-01', '2017-12-31']

    execution_args = []
    for idx, bbox in zip(indexes, bbox_list[indexes]):
        execution_args.append({
            add_iw: {'bbox': bbox, 'time_interval': time_interval},
            save: {'eopatch_folder': 'eopatch_{}'.format(idx)}
        })

    workflow = LinearWorkflow(
        add_iw,
        save)

    workflow.execute(execution_args)

    executor = EOExecutor(workflow, execution_args, save_logs=True, logs_folder='ExecutionLogs')
    executor.run(workers=2, multiprocess=False)


if __name__ == '__main__':
    path = '/home/beno/Documents/test'

    patch_location = path + '/Slovenia/'

    gdf, bbox_list = generate_slo_shapefile(path)

    download_patches(path + '/Slovenia_S1/', gdf, bbox_list)
