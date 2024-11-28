from typing import Union

import pandas as pd
from alphashape import alphashape
from shapely.geometry import Point, LineString, Polygon, MultiPolygon

from pyafs.graphics.alpha_shape import plot_alpha_shape


def calc_alpha_shape_upper_boundary(
        spec_df: pd.DataFrame,
        alpha_radius: float,
        debug: Union[bool, str] = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate the upper boundary of the alpha shape (tilde_AS_alpha)."""
    spec_df = spec_df.copy()
    cleanup_spec_df = spec_df[~spec_df['is_outlier']].copy()

    if cleanup_spec_df.empty:
        raise ValueError('No non-outlier data points to calculate alpha shape, aborting...')

    # calculate the alpha shape
    alpha_shape = alphashape(
        points=cleanup_spec_df[['wvl', 'scaled_intensity']].values,
        alpha=1 / alpha_radius
    )

    # convert the alpha shape to a dataframe
    if isinstance(alpha_shape, Polygon):
        alpha_shape_points = list(alpha_shape.exterior.coords)
    elif isinstance(alpha_shape, MultiPolygon):
        alpha_shape_points = [
            coord for polygon in alpha_shape.geoms for coord in polygon.exterior.coords
        ]
    else:
        raise ValueError(f'Empty alpha-shape or unexpected type: {type(alpha_shape)}')
    alpha_shape_df = pd.DataFrame(alpha_shape_points, columns=['wvl', 'scaled_intensity'])

    # construct the alpha-shape Polygon
    alpha_shape_polygon = Polygon(alpha_shape_points)
    _, min_y, _, max_y = alpha_shape_polygon.bounds

    # find tilde_AS_alpha, the upper boundary of the alpha-shape
    upper_boundary = []
    for idx, wvl in enumerate(spec_df['wvl']):
        # vertical line at wvl
        intersection_poly = alpha_shape_polygon.intersection(
            LineString([(wvl, min_y), (wvl, max_y)])
        )

        # if no intersection, append its intensity
        if intersection_poly.is_empty:
            upper_boundary.append(spec_df['scaled_intensity'].iloc[idx])
        # if intersection is a point, append its y-coordinate
        elif isinstance(intersection_poly, Point):
            upper_boundary.append(intersection_poly.y)
        # if intersection is a line, append the maximum y-coordinate
        elif isinstance(intersection_poly, LineString):
            upper_boundary.append(max(intersection_poly.xy[1]))
        else:
            raise ValueError(f'Unexpected intersection type when '
                             f'calculating tilde_AS_alpha: {type(intersection_poly)}')

    # update with tilde_AS_alpha
    spec_df['tilde_AS_alpha'] = upper_boundary
    # mark pixels whose intensity is intersecting with tilde_AS_alpha
    spec_df['is_intersect_with_tilde_AS_alpha'] = (
            spec_df['scaled_intensity'] == spec_df['tilde_AS_alpha']
    )

    if debug:
        plot_alpha_shape(spec_df, alpha_shape_df, debug=debug)

    return spec_df, alpha_shape_df
