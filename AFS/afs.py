from typing import Tuple, Union

import numpy as np
import pandas as pd
from alphashape import alphashape
from shapely.geometry import Polygon, MultiPolygon, LineString, Point


def _calc_alpha_shape_upper_boundary(
        wvl, intensity, alpha_ball_radius: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate the upper boundary of the alpha-shape of the spectrum."""
    alpha_shape = alphashape(np.array(list(zip(wvl, intensity))), 1 / alpha_ball_radius ** 2)

    # find the vertices of the alpha-shape
    if isinstance(alpha_shape, Polygon):
        boundary = list(alpha_shape.exterior.coords)
    elif isinstance(alpha_shape, MultiPolygon):
        boundary = [coord for polygon in alpha_shape.geoms
                    for coord in polygon.exterior.coords]
    else:
        raise ValueError('Alpha shape is empty or of an unsupported geometry type.')
    boundary_df = pd.DataFrame(boundary, columns=['wvl', 'scaled_intensity'])

    # find the upper boundary of the alpha-shape
    boundary_polygon = Polygon(boundary)
    min_x, min_y, max_x, max_y = boundary_polygon.bounds
    upper_boundary = []
    for x in boundary_df['wvl'].unique():
        print(x)
        intersections = boundary_polygon.intersection(LineString([(x, min_y - 1), (x, max_y + 1)]))

        if intersections.is_empty:
            continue
        elif isinstance(intersections, Point):
            upper_boundary.append((x, intersections.y))
        elif isinstance(intersections, LineString):
            upper_boundary.append((x, np.max([intersections.coords[0][1], intersections.coords[1][1]])))
        elif isinstance(intersections, Polygon):
            upper_boundary.append((x, np.max([coord[1] for coord in intersections.exterior.coords])))
        elif isinstance(intersections, MultiPolygon):
            upper_boundary.append((x, np.max([polygon.exterior.coords[1] for polygon in intersections.geoms])))
        else:
            raise ValueError('Intersection is of an unsupported geometry type.')

    upper_boundary_df = pd.DataFrame(upper_boundary, columns=['wvl', 'scaled_intensity']
                                     ).sort_values(by='wvl').reset_index()

    return boundary_df, upper_boundary_df


def afs(
        wvl, intensity, alpha: float = None, q=0.95, d=.25,
        debug: Union[bool, str] = False,
):
    spec_df = pd.DataFrame(
        {'wvl': wvl, 'intensity': intensity}
    ).sort_values(by='wvl').dropna().reset_index(drop=True)
    if spec_df.empty:
        raise ValueError('Input data is either empty or full of NaN values, aborting.')

    wvl_range = spec_df['wvl'].max() - spec_df['wvl'].min()

    # number of data points (pixels)
    n = len(spec_df)
    # scaling factor for the intensity vector
    u = wvl_range / 10 / spec_df['intensity'].max()
    spec_df['scaled_intensity'] = spec_df['intensity'] * u
    # radius of the alpha-ball (alpha shape)
    alpha = alpha or wvl_range / 6

    # calculate the upper boundary of the alpha-shape
    boundary_df, upper_boundary_df = _calc_alpha_shape_upper_boundary(
        spec_df['wvl'], spec_df['scaled_intensity'], alpha)

    # [AFS - step 3] local polynomial regression on tilde(AS_alpha)

    # visualise and save intermediate results
    if debug:
        from AFS._plot import plot_alpha_shape
        if isinstance(debug, str):
            plot_alpha_shape(spec_df, boundary_df, upper_boundary_df, debug)
        else:
            plot_alpha_shape(spec_df, boundary_df, upper_boundary_df)
