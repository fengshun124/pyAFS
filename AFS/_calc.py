import os
from typing import Tuple, Union

import numpy as np
import pandas as pd
from alphashape import alphashape
from shapely import Polygon, MultiPolygon, LineString, Point
from statsmodels.nonparametric.smoothers_lowess import lowess


def calc_tilde_AS_alpha(
        spec_df: pd.DataFrame, alpha_ball_radius: float, debug: Union[bool, str] = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate the upper boundary of the alpha-shape of the spectrum."""
    spec_df = spec_df.copy()
    alpha_shape = alphashape(spec_df[['wvl', 'scaled_intensity']].values, 1 / alpha_ball_radius ** 2)

    # find the vertices of the alpha-shape
    if isinstance(alpha_shape, Polygon):
        alpha_shape_points = list(alpha_shape.exterior.coords)
    elif isinstance(alpha_shape, MultiPolygon):
        alpha_shape_points = [coord for polygon in alpha_shape.geoms
                              for coord in polygon.exterior.coords]
    else:
        raise ValueError('Alpha shape is empty or of an unsupported geometry type.')
    alpha_shape_df = pd.DataFrame(alpha_shape_points, columns=['wvl', 'scaled_intensity'])

    alpha_shape_polygon = Polygon(alpha_shape_points)
    # find the boundary of the alpha-shape to construct LineString
    min_x, min_y, max_x, max_y = alpha_shape_polygon.bounds

    # find tilde(AS_alpha), the upper boundary of the alpha-shape at each spectral pixel
    upper_boundary = []
    for x in spec_df['wvl']:
        intersections = alpha_shape_polygon.intersection(LineString([(x, min_y - 1), (x, max_y + 1)]))

        if intersections.is_empty:
            continue
        elif isinstance(intersections, Point):
            upper_boundary.append(intersections.y)
        elif isinstance(intersections, LineString):
            upper_boundary.append(np.max([intersections.coords[0][1], intersections.coords[1][1]]))
        elif isinstance(intersections, Polygon):
            upper_boundary.append(np.max([coord[1] for coord in intersections.exterior.coords]))
        elif isinstance(intersections, MultiPolygon):
            upper_boundary.append(np.max([polygon.exterior.coords[1] for polygon in intersections.geoms]))
        else:
            raise ValueError('Intersection is of an unsupported geometry type.')

    spec_df['tilde_AS_alpha'] = upper_boundary
    # mark the intersection of the spectrum with the upper boundary of the alpha-shape
    spec_df['is_intersect_with_alpha_shape'] = spec_df['scaled_intensity'] == spec_df['tilde_AS_alpha']

    if debug:
        from AFS._plot import plot_spectrum

        plot_data_dict = {
            'plot_obj_dicts': [
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'scaled_intensity',
                 'label': 'spectrum',
                 'ls': '-', 'lw': 1, 'c': 'grey', 'alpha': .8, 'zorder': 1},
                {'data': alpha_shape_df, 'x_key': 'wvl', 'y_key': 'scaled_intensity',
                 'label': r'$\alpha$-shape',
                 'marker': 'x', 'ms': 8, 'mew': 1, 'ls': ':', 'lw': 1, 'c': 'tab:red', 'zorder': 3},
                {'data': spec_df[spec_df['is_intersect_with_alpha_shape']],
                 'x_key': 'wvl', 'y_key': 'scaled_intensity', 'label': 'tilde AS alpha $\cap$ spectrum',
                 'marker': '+', 'ms': 8, 'mew': 1, 'ls': '', 'c': 'tab:green', 'zorder': 3},
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'tilde_AS_alpha',
                 'label': 'tilde AS alpha',
                 'marker': 'o', 'ms': 4, 'mew': 1, 'ls': '', 'lw': 1,
                 'mec': 'tab:blue', 'mfc': 'None', 'alpha': .6, 'zorder': 2},
            ],
            'fig_title': '$\\alpha$-shape of the spectrum'
        }
        if isinstance(debug, str):
            print(f'saving alpha-shape csv and plot to {debug}')
            os.makedirs(debug, exist_ok=True)
            plot_spectrum(**plot_data_dict, exp_filename=os.path.join(debug, 'alpha_shape.png'))
            spec_df.to_csv(os.path.join(debug, 'spec_df.csv'), index=False)
        else:
            plot_spectrum(**plot_data_dict)

    return spec_df, alpha_shape_df


def apply_local_smoothing(
        spec_df: pd.DataFrame, intensity_key: str, smoothed_intensity_key: str, frac: float,
        debug: Union[bool, str] = False
) -> pd.DataFrame:
    """Apply local polynomial regression to the spectrum."""
    loess_fit = lowess(endog=spec_df[intensity_key], exog=spec_df['wvl'], frac=frac, it=0, return_sorted=False)

    spec_df = spec_df.copy()
    spec_df[smoothed_intensity_key] = loess_fit

    if debug:
        from AFS._plot import plot_spectrum

        # plot the residual between the source and smoothed intensities for comparison
        tmp_df = spec_df.copy()
        tmp_df['residual'] = tmp_df[intensity_key] - tmp_df[smoothed_intensity_key]
        plot_data_dict = {
            'plot_obj_dicts': [
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'scaled_intensity',
                 'label': 'spectrum', 'ls': '-', 'lw': 1, 'c': 'grey', 'alpha': .8, 'zorder': 1},
                {'data': spec_df, 'x_key': 'wvl', 'y_key': intensity_key,
                 'label': intensity_key.replace('_', ' '),
                 'ls': '-.', 'lw': 1.2, 'c': 'tab:red', 'zorder': 1},
                {'data': spec_df, 'x_key': 'wvl', 'y_key': smoothed_intensity_key,
                 'label': smoothed_intensity_key.replace('_', ' '),
                 'ls': '--', 'lw': 1.2, 'c': 'tab:blue', 'zorder': 2},
                {'data': tmp_df, 'x_key': 'wvl', 'y_key': 'residual',
                 'label': 'src. - smoothed',
                 'ls': '-', 'lw': 1.2, 'c': 'tab:purple', 'zorder': 3},
            ],
            'aux_plot_obj_dicts': [
                {'orientation': 'h', 'y': 0, 'ls': ':', 'lw': 1, 'c': 'k', 'alpha': .8, 'zorder': -1}
            ],
            'fig_title': 'Local polynomial regression'
        }
        if isinstance(debug, str):
            print(f'saving local smoothing csv and plot to {debug}')
            os.makedirs(debug, exist_ok=True)
            plot_spectrum(**plot_data_dict, exp_filename=os.path.join(debug, 'local_smoothing.png'))
            spec_df.to_csv(os.path.join(debug, 'spec_df.csv'), index=False)
        else:
            plot_spectrum(**plot_data_dict)

    return spec_df


def calc_norm_intensity(
        spec_df: pd.DataFrame, intensity_key: str, blaze_key: str, norm_intensity_key: str,
        debug: Union[bool, str] = False
) -> pd.DataFrame:
    """Calculate the normalised intensity of the spectrum."""
    spec_df = spec_df.copy()
    spec_df[norm_intensity_key] = spec_df[intensity_key] / spec_df[blaze_key]

    if debug:
        from AFS._plot import plot_norm_spectrum
        plot_dict = {
            'plot_obj_dicts': [
                {'data': spec_df, 'x_key': 'wvl', 'y_key': norm_intensity_key,
                 'label': norm_intensity_key.replace('_', ' '),
                 'marker': '.', 'ms': 2, 'ls': '', 'c': 'k', 'zorder': 1},
            ],
            'fig_title': 'Normalised spectrum'
        }
        if isinstance(debug, str):
            print(f'saving normalised spectrum plot to {debug}')
            os.makedirs(debug, exist_ok=True)
            plot_norm_spectrum(**plot_dict, exp_filename=os.path.join(debug, 'norm_spectrum.png'))
        else:
            plot_norm_spectrum(**plot_dict)

    return spec_df
