import os
from typing import Union

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

        window_spec_df = spec_df[(spec_df['wvl'] >= start_point['wvl']) & (spec_df['wvl'] <= end_point['wvl'])]
        window_spec_quantile = window_spec_df['primitive_norm_intensity'].quantile(quantile)

        spec_df.loc[window_spec_df.index, 'is_selected_pixel'] = (
                window_spec_df['primitive_norm_intensity'] >= window_spec_quantile)
        quantile_data.append({
            'x': [start_point['wvl'], end_point['wvl']],
            'y': [window_spec_quantile] * 2
        })

    if debug:
        from AFS._plot import plot_norm_spectrum
        plot_data_dict = {
            'plot_obj_dicts': [
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'primitive_norm_intensity',
                 'label': 'norm. spec.',
                 'ls': '-', 'lw': 1, 'c': 'grey', 'alpha': .8, 'zorder': 1},
                {'data': spec_df[spec_df['is_intersect_with_alpha_shape']],
                 'x_key': 'wvl', 'y_key': 'primitive_norm_intensity',
                 'label': 'tilde AS alpha $\cap$ spec.',
                 'marker': 'x', 'ms': 8, 'mew': 1, 'ls': '', 'c': 'tab:red', 'zorder': 3},
                {'data': spec_df[spec_df['is_selected_pixel']],
                 'x_key': 'wvl', 'y_key': 'primitive_norm_intensity',
                 'label': 'selected pixels',
                 'marker': '+', 'ms': 8, 'mew': 1, 'ls': '', 'c': 'tab:green', 'zorder': 3},
            ],
            'fig_title': f'Selected Pixels based on Quantiles (q={quantile})'
        }
        for data in quantile_data:
            quantile_df = pd.DataFrame(data)
            plot_data_dict['plot_obj_dicts'].append({
                'data': quantile_df, 'x_key': 'x', 'y_key': 'y',
                'label': f'q={quantile}', 'ls': '-', 'lw': 1, 'c': 'tab:blue', 'zorder': 2
            })

        if isinstance(debug, str):
            plot_norm_spectrum(**plot_data_dict, exp_filename=debug)
        else:
            plot_norm_spectrum(**plot_data_dict)

    return spec_df


def calc_afs_final_norm_intensity(
        spec_df: pd.DataFrame, frac: float, debug: Union[bool, str] = False
) -> pd.DataFrame:
    filtered_spec_df = spec_df[spec_df['is_selected_pixel']].copy()

    # apply local polynomial regression to the filtered spectrum
    filtered_spec_df = apply_local_smoothing(
        filtered_spec_df,
        intensity_key='scaled_intensity',
        smoothed_intensity_key='final_blaze',
        frac=frac, debug=False
    )

    final_blaze_func = interp1d(
        filtered_spec_df['wvl'], filtered_spec_df['final_blaze'],
        kind='cubic', fill_value='extrapolate'
    )

    # calculate refined blaze function hat(B_2) by interpolating the smoothed spectrum
    spec_df['final_blaze'] = final_blaze_func(spec_df['wvl'])
    spec_df['final_norm_intensity'] = spec_df['scaled_intensity'] / spec_df['final_blaze']

    if debug:
        from AFS._plot import plot_spectrum, plot_norm_spectrum

        tmp_df = filtered_spec_df.copy()
        tmp_df['residual'] = tmp_df['final_blaze'] - tmp_df['primitive_blaze']
        plot_data_dict = {
            'plot_obj_dicts': [
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'scaled_intensity',
                 'label': 'spectrum', 'ls': '-', 'lw': 1, 'c': 'grey', 'alpha': .8, 'zorder': 1},
                {'data': filtered_spec_df, 'x_key': 'wvl', 'y_key': 'scaled_intensity',
                 'label': 'selected pixels',
                 'marker': 'x', 'ms': 8, 'mew': 1, 'ls': '', 'c': 'tab:green', 'zorder': 3},
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'primitive_blaze',
                 'label': 'primitive blaze',
                 'ls': '-.', 'lw': 1.2, 'c': 'tab:red', 'zorder': 2},
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'final_blaze',
                 'label': 'final blaze',
                 'ls': '--', 'lw': 1.2, 'c': 'tab:blue', 'zorder': 3},
                {'data': tmp_df, 'x_key': 'wvl', 'y_key': 'residual',
                 'label': 'final - primitive blaze',
                 'ls': '-', 'lw': 1.2, 'c': 'tab:purple', 'zorder': 4},
            ],
            'aux_plot_obj_dicts': [
                {'orientation': 'h', 'y': 0, 'ls': ':', 'lw': 1, 'c': 'k', 'alpha': .8, 'zorder': -1}
            ],
            'fig_title': 'Final Blaze Function'
        }

        tmp_df = spec_df.copy()
        tmp_df['residual'] = tmp_df['final_norm_intensity'] - tmp_df['primitive_norm_intensity']
        norm_plot_data_dict = {
            'plot_obj_dicts': [
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'primitive_norm_intensity',
                 'label': 'primitive norm. spec.',
                 'marker': '.', 'ms': 2, 'ls': '', 'c': 'grey', 'zorder': 1},
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'final_norm_intensity',
                 'label': 'final norm. spec.',
                 'marker': '.', 'ms': 2, 'ls': '', 'c': 'tab:blue', 'zorder': 1},
                {'data': tmp_df, 'x_key': 'wvl', 'y_key': 'residual',
                 'label': 'final - primitive spec.',
                 'ls': '-', 'lw': 1.2, 'c': 'tab:purple', 'zorder': 3},
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


def afs(
        wvl, intensity, alpha_radius: float = None, q=0.95, d=.25,
        debug: Union[bool, str] = False,
) -> tuple[np.array, DataFrame]:
    spec_df = pd.DataFrame(
        {'wvl': wvl, 'intensity': intensity}
    ).sort_values(by='wvl').dropna().reset_index(drop=True)
    if spec_df.empty:
        raise ValueError('Input data is either empty or full of NaN values, aborting.')

    wvl_range = spec_df['wvl'].max() - spec_df['wvl'].min()

    # scaling factor for the intensity vector
    u = wvl_range / 10 / spec_df['intensity'].max()
    spec_df['scaled_intensity'] = spec_df['intensity'] * u
    # radius of the alpha-ball (alpha shape)
    alpha_radius = alpha_radius or wvl_range / 6

    # step 2: find AS_alpha and calculate tilde(AS_alpha)
    spec_df, alpha_shape_df = calc_tilde_AS_alpha(spec_df, alpha_radius, debug)

    # Step 3: smooth tilde(AS_alpha) using local polynomial regression (LOESS)
    # to estimate the blaze function hat(B_1).
    # then calculate primitive normalised intensity hat(y^1) by y / hat(B_1).
    spec_df = calc_primitive_norm_intensity(spec_df, frac=d, debug=debug)

    # step 4: filter spectrum pixels for next iteration
    spec_df = filter_pixels_above_quantiles(spec_df, q, debug)

    # step 5: smooth filtered pixels using local polynomial regression (LOESS)
    # to estimate the refined blaze function hat(B_2)
    # then calculate final normalised intensity hat(y^2) by y / hat(B_2)
    spec_df = calc_afs_final_norm_intensity(spec_df, frac=d, debug=debug)

    return np.array(spec_df['final_norm_intensity']), spec_df
