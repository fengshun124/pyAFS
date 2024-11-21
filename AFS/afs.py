import os
from typing import Union, Literal

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.interpolate import interp1d

from AFS._calc import calc_tilde_AS_alpha, apply_local_smoothing, calc_primitive_norm_intensity


def filter_pixels_above_quantiles(
        spec_df: pd.DataFrame, quantile: float, debug: Union[bool, str] = False
) -> pd.DataFrame:
    """Filter the spectrum pixels based on the quantile of the normalised intensity."""
    intersecting_points_df = spec_df[spec_df['is_intersect_with_alpha_shape']].copy()
    if len(intersecting_points_df) < 2:
        raise ValueError(f'Expect at least 2 intersection points, got {len(intersecting_points_df)}.')

    spec_df['is_selected_pixel'] = False
    quantile_data = []
    for i in range(len(intersecting_points_df) - 1):
        start_point = intersecting_points_df.iloc[i]
        end_point = intersecting_points_df.iloc[i + 1]

        window_spec_df = spec_df[
            (spec_df['wvl'] >= start_point['wvl']) &
            (spec_df['wvl'] <= end_point['wvl']) &
            (spec_df['is_outlier'] == False)
            ]
        window_spec_quantile = window_spec_df['primitive_norm_intensity'].quantile(quantile)

        spec_df.loc[window_spec_df.index, 'is_selected_pixel'] = (
                window_spec_df['primitive_norm_intensity'] >= window_spec_quantile)
        quantile_data.append({
            'x': [start_point['wvl'], end_point['wvl']],
            'y': [window_spec_quantile] * 2
        })

    if debug:
        from AFS._plot import plot_norm_spectrum

        # check if outlier are present
        if spec_df['is_outlier'].any():
            spec_plot_data_dicts = [
                # primitive normalised spectrum
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'primitive_norm_intensity',
                 'style': 'src_spec', 'label': 'primitive norm. spec.'},
                # primitive normalised spectrum with outliers removed
                {'data': spec_df[~spec_df['is_outlier']],
                 'x_key': 'wvl', 'y_key': 'primitive_norm_intensity',
                 'style': 'clean_spec', 'label': 'primitive norm. spec. (cleaned)'},
            ]
        else:
            spec_plot_data_dicts = [
                # primitive normalised spectrum
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'primitive_norm_intensity',
                 'style': 'clean_spec', 'label': 'primitive norm. spec.'},
            ]

        plot_data_dict = {
            'plot_obj_dicts': [
                *spec_plot_data_dicts,
                # selected pixels
                {'data': spec_df[spec_df['is_selected_pixel']],
                 'x_key': 'wvl', 'y_key': 'primitive_norm_intensity',
                 'style': 'selected_pixel', 'label': 'selected pixels'},
            ],
            'fig_title': f'Selected Pixels based on Quantiles (q={quantile})'
        }
        for data in quantile_data:
            quantile_df = pd.DataFrame(data)
            plot_data_dict['plot_obj_dicts'].append({
                'data': quantile_df, 'x_key': 'x', 'y_key': 'y',
                'style': 'line', 'label': f'{quantile:.0%} quantile', 'c': 'tab:red'
            })

        if isinstance(debug, str):
            plot_norm_spectrum(**plot_data_dict, exp_filename=debug)
        else:
            plot_norm_spectrum(**plot_data_dict)

    return spec_df


def calc_afs_final_norm_intensity(
        spec_df: pd.DataFrame, frac: float, debug: Union[bool, str] = False,
        smoothing_method: str = 'loess'
) -> pd.DataFrame:
    filtered_spec_df = spec_df[spec_df['is_selected_pixel'] & ~spec_df['is_outlier']].copy()

    # apply local polynomial regression to the filtered spectrum
    filtered_spec_df = apply_local_smoothing(
        filtered_spec_df,
        intensity_key='scaled_intensity',
        smoothing_method=smoothing_method,
        smoothed_intensity_key='final_blaze',
        frac=frac, debug=False
    )

    # calculate refined blaze function hat(B_2) by interpolating the smoothed spectrum
    final_blaze_func = interp1d(
        filtered_spec_df['wvl'], filtered_spec_df['final_blaze'],
        kind='cubic', fill_value='extrapolate'
    )
    spec_df['final_blaze'] = final_blaze_func(spec_df['wvl'])
    spec_df['final_norm_intensity'] = spec_df['scaled_intensity'] / spec_df['final_blaze']

    if debug:
        from AFS._plot import plot_spectrum, plot_norm_spectrum

        # check if outlier are present
        if spec_df['is_outlier'].any():
            spec_plot_data_dicts = [
                # spectrum
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'scaled_intensity',
                 'style': 'src_spec', 'label': 'source intensity'},
                # spectrum with outliers removed
                {'data': spec_df[~spec_df['is_outlier']],
                 'x_key': 'wvl', 'y_key': 'scaled_intensity',
                 'style': 'clean_spec', 'label': 'cleaned spectrum'},
            ]
        else:
            spec_plot_data_dicts = [
                # spectrum
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'scaled_intensity',
                 'style': 'clean_spec', 'label': 'source intensity'},
            ]

        tmp_df = filtered_spec_df.copy()
        tmp_df['residual'] = tmp_df['final_blaze'] - tmp_df['primitive_blaze']
        plot_data_dict = {
            'plot_obj_dicts': [
                *spec_plot_data_dicts,
                # selected pixels
                {'data': spec_df[spec_df['is_selected_pixel']],
                 'x_key': 'wvl', 'y_key': 'scaled_intensity',
                 'style': 'selected_pixel', 'label': 'selected pixels'},
                # primitive blaze function
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'primitive_blaze',
                 'style': 'line', 'label': 'primitive blaze func.', 'c': 'tab:red'},
                # final blaze function
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'final_blaze',
                 'style': 'line', 'label': 'final blaze func.', 'c': 'tab:blue'},
                # residual
                {'data': tmp_df, 'x_key': 'wvl', 'y_key': 'residual',
                 'style': 'line', 'label': 'final - primitive blaze func.', 'c': 'tab:purple'},
            ],
            'aux_plot_obj_dicts': [
                {'orientation': 'h', 'y': 0, 'ls': ':', 'lw': 1, 'c': 'k', 'alpha': .8, 'zorder': -1}
            ],
            'fig_title': f'Final Blaze Function ({smoothing_method})'
        }

        tmp_df = spec_df.copy()
        tmp_df['residual'] = tmp_df['final_norm_intensity'] - tmp_df['primitive_norm_intensity']
        norm_plot_data_dict = {
            'plot_obj_dicts': [
                # primitive normalised spectrum
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'primitive_norm_intensity',
                 'style': 'line', 'label': 'primitive norm. spec.', 'c': 'tab:red'},
                # final normalised spectrum
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'final_norm_intensity',
                 'style': 'line', 'label': 'final norm. spec.', 'c': 'tab:blue'},
                # residual
                {'data': tmp_df, 'x_key': 'wvl', 'y_key': 'residual',
                 'style': 'line', 'label': 'final - primitive norm. spec.', 'c': 'tab:purple'},
            ],
            'aux_plot_obj_dicts': [
                {'orientation': 'h', 'y': 0, 'ls': ':', 'lw': 1, 'c': 'k', 'alpha': .8, 'zorder': -1}
            ],
            'fig_title': 'Final Normalised Spectrum'
        }
        if isinstance(debug, str):
            print(f'saving final normalised spectrum plot to {debug}')
            os.makedirs(debug, exist_ok=True)
            plot_spectrum(
                **plot_data_dict, exp_filename=os.path.join(debug, 'final_blaze_function.png'))
            plot_norm_spectrum(
                **norm_plot_data_dict, exp_filename=os.path.join(debug, 'final_norm_intensity.png'))
        else:
            plot_spectrum(**plot_data_dict)
            plot_norm_spectrum(**norm_plot_data_dict)

    return spec_df


def mark_outliers(
        spec_df: pd.DataFrame,
        rolling_window: int = 60, sigma: float = 1,
        debug: Union[bool, str] = False
) -> pd.DataFrame:
    """Mark outliers in the spectrum."""
    spec_df['is_outlier'] = False

    # calculate rolling median
    spec_df['intensity_rolling_median'] = spec_df['intensity'].rolling(rolling_window, center=True).median()
    spec_df['intensity_residual'] = spec_df['intensity'] - spec_df['intensity_rolling_median']

    # mark outliers
    spec_df['is_outlier'] = (
            spec_df['intensity_residual'] > sigma * spec_df['intensity_residual'].std())

    if debug:
        from AFS._plot import plot_spectrum

        plot_data_dict = {
            'plot_obj_dicts': [
                # spectrum
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'intensity', 'style': 'src_spec'},
                # rolling median
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'intensity_rolling_median',
                 'style': 'line', 'label': 'rolling median', 'c': 'tab:red'},
                # residual
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'intensity_residual',
                 'style': 'line', 'label': 'intensity - rolling median', 'c': 'tab:blue'},
                # clean spectrum
                {'data': spec_df[~spec_df['is_outlier']],
                 'x_key': 'wvl', 'y_key': 'intensity', 'style': 'clean_spec',
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


def afs(
        wvl, intensity,
        alpha_radius: float = None,
        continuum_filter_quantile: float = .9,
        smoothing_frac: float = .25,
        debug: Union[bool, str] = False,
        is_remove_outliers: bool = True,
        outlier_rolling_window: int = 60,
        outlier_sigma: float = .8,
        primitive_blaze_smoothing: Literal['lowess', 'spline'] = 'lowess',
        final_blaze_smoothing: Literal['lowess', 'spline'] = 'lowess'
) -> tuple[np.array, DataFrame]:
    spec_df = pd.DataFrame(
        {'wvl': wvl, 'intensity': intensity}
    ).sort_values(by='wvl').dropna().reset_index(drop=True)
    if spec_df.empty:
        raise ValueError('Input data is either empty or full of NaN values, aborting.')

    wvl_range = spec_df['wvl'].max() - spec_df['wvl'].min()
    # radius of the alpha-ball (alpha shape)
    alpha_radius = alpha_radius or wvl_range / 10

    # step 1: scale the range of intensity and wavelength to be 1:10
    u = wvl_range / 10 / spec_df['intensity'].max()
    spec_df['scaled_intensity'] = spec_df['intensity'] * u

    # step 1.5: remove spectral outliers resulting from cosmic rays or other noise
    if is_remove_outliers:
        spec_df = mark_outliers(
            spec_df,
            rolling_window=outlier_rolling_window,
            sigma=outlier_sigma,
            debug=debug
        )

    # step 2: find AS_alpha and calculate tilde(AS_alpha)
    spec_df, alpha_shape_df = calc_tilde_AS_alpha(spec_df, alpha_radius, debug)

    # Step 3: smooth tilde(AS_alpha) using local polynomial regression (LOESS)
    # to estimate the blaze function hat(B_1).
    # then calculate primitive normalised intensity hat(y^1) by y / hat(B_1).
    spec_df = calc_primitive_norm_intensity(
        spec_df,
        smoothing_method=primitive_blaze_smoothing,
        frac=smoothing_frac,
        debug=debug
    )

    # step 4: filter spectrum pixels for next iteration
    spec_df = filter_pixels_above_quantiles(
        spec_df, quantile=continuum_filter_quantile, debug=debug)

    # step 5: smooth filtered pixels using local polynomial regression (LOESS)
    # to estimate the refined blaze function hat(B_2)
    # then calculate final normalised intensity hat(y^2) by y / hat(B_2)
    spec_df = calc_afs_final_norm_intensity(
        spec_df,
        smoothing_method=final_blaze_smoothing,
        frac=smoothing_frac,
        debug=debug
    )

    return np.array(spec_df['final_norm_intensity']), spec_df
