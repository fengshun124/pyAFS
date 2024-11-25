import os
from typing import Tuple, Union

import numpy as np
import pandas as pd
from alphashape import alphashape
from loess.loess_1d import loess_1d
from scipy.interpolate import interp1d, UnivariateSpline
from shapely import Polygon, MultiPolygon, LineString, Point

LOCAL_SMOOTHING_METHODS = ['loess', 'spline']


def _extract_smoothing_args(
        smooth_method: LOCAL_SMOOTHING_METHODS,
        **kwargs
) -> dict:
    if smooth_method == 'loess':
        return {
            'frac': kwargs.get('frac', .25),
            'degree': kwargs.get('deg', 2),
        }
    elif smooth_method == 'spline':
        return {
            's': kwargs.get('s', 1e-5),
            'k': kwargs.get('k', 2)
        }
    else:
        raise ValueError(f'Invalid processing mode: {smooth_method}.')


def apply_local_smoothing(
        spec_df: pd.DataFrame,
        intensity_key: str,
        smoothed_intensity_key: str,
        smoothing_method: LOCAL_SMOOTHING_METHODS = 'loess',
        debug: Union[bool, str] = False,
        **kwargs
) -> pd.DataFrame:
    """Apply local polynomial regression to the spectrum."""
    if smoothing_method == 'loess':
        # apply local polynomial regression to the spectrum
        smoothed_intensity = loess_1d(
            x=spec_df['wvl'].to_numpy(), y=spec_df[intensity_key].to_numpy(),
            **_extract_smoothing_args(smoothing_method, **kwargs)
        )[1]
    elif smoothing_method == 'spline':
        # apply spline to the spectrum
        smoothed_intensity = UnivariateSpline(
            spec_df['wvl'], spec_df[intensity_key], ext='extrapolate',
            **_extract_smoothing_args(smoothing_method, **kwargs)
        )(spec_df['wvl'])
    else:
        raise ValueError(f'Invalid processing mode: {smoothing_method}. '
                         f'Please choose from "loess" or "spline".')

    spec_df = spec_df.copy()
    spec_df[smoothed_intensity_key] = smoothed_intensity

    if debug:
        from pyafs._plot import plot_spectrum

        # plot the residual between the source and smoothed intensities for comparison
        tmp_df = spec_df.copy()
        tmp_df['residual'] = tmp_df[intensity_key] - tmp_df[smoothed_intensity_key]
        plot_data_dict = {
            'plot_obj_dicts': [
                # spectrum
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'scaled_intensity', 'style': 'src_spec'},
                # source intensity
                {'data': spec_df, 'x_key': 'wvl', 'y_key': intensity_key,
                 'style': 'line', 'label': intensity_key, 'c': 'tab:red'},
                # smoothed intensity
                {'data': spec_df, 'x_key': 'wvl', 'y_key': smoothed_intensity_key,
                 'style': 'line', 'label': smoothed_intensity_key, 'c': 'tab:blue'},
                # residual
                {'data': tmp_df, 'x_key': 'wvl', 'y_key': 'residual',
                 'style': 'line', 'label': 'residual', 'c': 'tab:purple'},
            ],
            'aux_plot_obj_dicts': [
                {'orientation': 'h', 'y': 0, 'ls': ':', 'lw': 1, 'c': 'k', 'alpha': .8, 'zorder': -1}
            ],
            'fig_title': f'Local Smoothing ({smoothing_method})'
        }
        if isinstance(debug, str):
            print(f'saving local smoothing plot to {debug}')
            os.makedirs(debug, exist_ok=True)
            plot_spectrum(**plot_data_dict, exp_filename=os.path.join(debug, 'local_smoothing.png'))
        else:
            plot_spectrum(**plot_data_dict)

    return spec_df


def calc_tilde_AS_alpha(
        spec_df: pd.DataFrame,
        alpha_ball_radius: float,
        debug: Union[bool, str] = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate the upper boundary of the alpha-shape of the spectrum."""
    filtered_spec_df = spec_df[spec_df['is_outlier'] == False].copy()
    alpha_shape = alphashape(
        filtered_spec_df[['wvl', 'scaled_intensity']].values, 1 / alpha_ball_radius ** 2,
    )

    # find the vertices of the alpha-shape
    if isinstance(alpha_shape, Polygon):
        alpha_shape_points = list(alpha_shape.exterior.coords)
    elif isinstance(alpha_shape, MultiPolygon):
        alpha_shape_points = [
            coord for polygon in alpha_shape.geoms
            for coord in polygon.exterior.coords
        ]
    else:
        raise ValueError('Alpha shape is empty or of an unsupported geometry type.')
    alpha_shape_df = pd.DataFrame(alpha_shape_points, columns=['wvl', 'scaled_intensity'])

    alpha_shape_polygon = Polygon(alpha_shape_points)
    # find the boundary of the alpha-shape to construct LineString
    min_x, min_y, max_x, max_y = alpha_shape_polygon.bounds

    # find tilde(AS_alpha), the upper boundary of the alpha-shape at each spectral pixel
    upper_boundary = []
    for x in spec_df['wvl']:
        intersections = alpha_shape_polygon.intersection(
            LineString([(x, min_y - 1), (x, max_y + 1)]))

        if intersections.is_empty:
            continue
        elif isinstance(intersections, Point):
            upper_boundary.append(intersections.y)
        elif isinstance(intersections, LineString):
            upper_boundary.append(np.max([intersections.coords[0][1], intersections.coords[1][1]]))
        else:
            raise ValueError('Intersection is of an unsupported geometry type.')

    spec_df['tilde_AS_alpha'] = upper_boundary
    # mark the intersection of the spectrum with the upper boundary of the alpha-shape
    spec_df['is_intersect_with_alpha_shape'] = (
            spec_df['scaled_intensity'] == spec_df['tilde_AS_alpha'])

    if debug:
        from pyafs._plot import plot_spectrum

        # check if outlier are present
        if spec_df['is_outlier'].any():
            spec_plot_obj_dicts = [
                # source spectrum
                {'data': spec_df,
                 'x_key': 'wvl', 'y_key': 'scaled_intensity', 'style': 'src_spec'},
                # outliers
                {'data': spec_df[spec_df['is_outlier']],
                 'x_key': 'wvl', 'y_key': 'scaled_intensity', 'style': 'outlier'},
                # cleanup spectrum
                {'data': filtered_spec_df,
                 'x_key': 'wvl', 'y_key': 'scaled_intensity', 'style': 'clean_spec'},
            ]
        else:
            spec_plot_obj_dicts = [
                # source spectrum
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'scaled_intensity',
                 'style': 'clean_spec', 'label': 'spec.'},
            ]

        plot_data_dict = {
            'plot_obj_dicts': [
                *spec_plot_obj_dicts,
                # alpha-shape
                {'data': alpha_shape_df,
                 'x_key': 'wvl', 'y_key': 'scaled_intensity', 'style': 'alpha_shape'},
                # tilde_AS_alpha
                {'data': spec_df,
                 'x_key': 'wvl', 'y_key': 'tilde_AS_alpha', 'style': 'tilde_AS_alpha'},
                # intersection points
                {'data': spec_df[spec_df['is_intersect_with_alpha_shape']],
                 'x_key': 'wvl', 'y_key': 'tilde_AS_alpha', 'style': 'tilde_AS_alpha_intersect'},
            ],
            'fig_title': '$\\alpha$-shape of the spectrum'
        }
        if isinstance(debug, str):
            print(f'saving alpha-shape plot to {debug}')
            os.makedirs(debug, exist_ok=True)
            plot_spectrum(**plot_data_dict, exp_filename=os.path.join(debug, 'alpha_shape.png'))
        else:
            plot_spectrum(**plot_data_dict)

    return spec_df, alpha_shape_df


def calc_primitive_norm_intensity(
        spec_df: pd.DataFrame,
        smoothing_method: LOCAL_SMOOTHING_METHODS = 'loess',
        debug: Union[bool, str] = False,
        **kwargs
) -> pd.DataFrame:
    """Calculate the normalised intensity of the spectrum."""
    filtered_spec_df = spec_df[spec_df['is_outlier'] == False].copy()

    # apply local polynomial regression to the spectrum
    filtered_spec_df = apply_local_smoothing(
        filtered_spec_df,
        intensity_key='tilde_AS_alpha',
        smoothing_method=smoothing_method,
        smoothed_intensity_key='primitive_blaze',
        debug=debug, **kwargs
    )
    # normalise the spectrum by the primitive blaze function
    primitive_blaze_func = interp1d(
        filtered_spec_df['wvl'], filtered_spec_df['primitive_blaze'],
        kind='cubic', fill_value='extrapolate'
    )
    spec_df['primitive_blaze'] = primitive_blaze_func(spec_df['wvl'])
    spec_df['primitive_norm_intensity'] = spec_df['scaled_intensity'] / spec_df['primitive_blaze']

    if debug:
        from pyafs._plot import plot_norm_spectrum

        # check if outlier are present
        if spec_df['is_outlier'].any():
            spec_plot_obj_dicts = [
                # primitive normalised spectrum
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'primitive_norm_intensity',
                 'style': 'src_spec', 'label': 'primitive norm. spec.'},
                # primitive normalised spectrum with outliers removed
                {'data': spec_df[spec_df['is_outlier'] == False],
                 'x_key': 'wvl', 'y_key': 'primitive_norm_intensity',
                 'style': 'clean_spec', 'label': 'cleaned primitive norm. spec.'},
            ]
        else:
            spec_plot_obj_dicts = [
                # primitive normalised spectrum
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'primitive_norm_intensity',
                 'style': 'clean_spec', 'label': 'primitive norm. spec.'},
            ]

        plot_dict = {
            'plot_obj_dicts': [*spec_plot_obj_dicts],
            'fig_title': 'Primitive Normalised Spectrum'
        }
        if isinstance(debug, str):
            print(f'saving normalised spectrum plot to {debug}')
            os.makedirs(debug, exist_ok=True)
            plot_norm_spectrum(
                **plot_dict, exp_filename=os.path.join(debug, 'primitive_norm_intensity.png'))
        else:
            plot_norm_spectrum(**plot_dict)

    return spec_df


def mark_outliers(
        spec_df: pd.DataFrame,
        rolling_window: int = 60,
        rolling_window_sigma: float = 1,
        intensity_delta_quantile: float = .9,
        debug: Union[bool, str] = False
) -> pd.DataFrame:
    """Mark outliers in the spectrum."""
    spec_df['is_outlier'] = False

    # calculate the differences between scaled_intensity
    spec_df['scaled_intensity_diff'] = np.max(
        [spec_df['scaled_intensity'].diff().abs().fillna(0),
         spec_df['scaled_intensity'].diff(-1).abs().fillna(0)],
    )

    # calculate rolling median
    spec_df['scaled_intensity_rolling_median'] = (
        spec_df['scaled_intensity'].rolling(rolling_window, center=True).median())
    spec_df['scaled_intensity_residual'] = (
            spec_df['scaled_intensity'] - spec_df['scaled_intensity_rolling_median'])

    # mark outliers
    spec_df['is_outlier'] = (
            (spec_df['scaled_intensity_diff'] >=
             spec_df['scaled_intensity_diff'].quantile(intensity_delta_quantile)) &
            (spec_df['scaled_intensity_residual'] >=
             rolling_window_sigma * spec_df['scaled_intensity_residual'].std())
    )

    if debug:
        from pyafs._plot import plot_spectrum

        plot_data_dict = {
            'plot_obj_dicts': [
                # spectrum
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'scaled_intensity', 'style': 'src_spec'},
                # rolling median
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'scaled_intensity_rolling_median',
                 'style': 'line', 'label': 'rolling median', 'c': 'tab:red'},
                # residual
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'scaled_intensity_residual',
                 'style': 'line', 'label': 'scaled_intensity - rolling median', 'c': 'tab:blue'},
                # clean spectrum
                {'data': spec_df[~spec_df['is_outlier']],
                 'x_key': 'wvl', 'y_key': 'scaled_intensity', 'style': 'clean_spec',
                 'label': 'cleaned spectrum'},
            ],
            'aux_plot_obj_dicts': [
                {'orientation': 'h', 'y': 0, 'ls': ':', 'lw': 1, 'c': 'k', 'alpha': .8, 'zorder': -1}
            ],
            'fig_title': 'Outlier Detection'
        }
        if isinstance(debug, str):
            print(f'saving outlier detection plot to {debug}')
            os.makedirs(debug, exist_ok=True)
            plot_spectrum(**plot_data_dict, exp_filename=os.path.join(debug, 'outlier_detection.png'))
        else:
            plot_spectrum(**plot_data_dict)

    return spec_df
