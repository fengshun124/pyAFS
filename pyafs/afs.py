import os
from typing import Union

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.interpolate import interp1d

from pyafs._calc import (
    LOCAL_SMOOTHING_METHODS,
    apply_local_smoothing,
    mark_outliers,
    calc_tilde_AS_alpha,
    calc_primitive_norm_intensity
)


def filter_pixels_above_quantiles(
        spec_df: pd.DataFrame,
        filter_quantile: float = .95,
        debug: Union[bool, str] = False
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
        window_spec_quantile = window_spec_df['primitive_norm_intensity'].quantile(filter_quantile)

        spec_df.loc[window_spec_df.index, 'is_selected_pixel'] = (
                window_spec_df['primitive_norm_intensity'] >= window_spec_quantile)
        quantile_data.append({
            'x': [start_point['wvl'], end_point['wvl']],
            'y': [window_spec_quantile] * 2
        })

    if debug:
        from pyafs._plot import plot_norm_spectrum

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
            'fig_title': f'Selected Pixels based on Quantiles (q={filter_quantile})'
        }
        for data in quantile_data:
            quantile_df = pd.DataFrame(data)
            plot_data_dict['plot_obj_dicts'].append({
                'data': quantile_df, 'x_key': 'x', 'y_key': 'y',
                'style': 'line', 'label': f'{filter_quantile:.0%} quantile', 'c': 'tab:red'
            })

        if isinstance(debug, str):
            plot_norm_spectrum(**plot_data_dict, exp_filename=debug)
        else:
            plot_norm_spectrum(**plot_data_dict)

    return spec_df


def calc_afs_final_norm_intensity(
        spec_df: pd.DataFrame,
        smoothing_method: LOCAL_SMOOTHING_METHODS = 'loess',
        debug: Union[bool, str] = False,
        **kwargs
) -> pd.DataFrame:
    filtered_spec_df = spec_df[
        ~spec_df['is_outlier'] &
        spec_df['is_intersect_with_alpha_shape'] &
        spec_df['is_selected_pixel']
        ].copy()

    # apply local polynomial regression to the filtered spectrum
    filtered_spec_df = apply_local_smoothing(
        filtered_spec_df,
        intensity_key='scaled_intensity',
        smoothing_method=smoothing_method,
        smoothed_intensity_key='final_blaze',
        debug=False, **kwargs
    )

    # calculate refined blaze function hat(B_2) by interpolating the smoothed spectrum
    final_blaze_func = interp1d(
        filtered_spec_df['wvl'], filtered_spec_df['final_blaze'],
        kind='cubic', fill_value='extrapolate'
    )
    spec_df['final_blaze'] = final_blaze_func(spec_df['wvl'])
    spec_df['final_norm_intensity'] = spec_df['scaled_intensity'] / spec_df['final_blaze']

    if debug:
        from pyafs._plot import plot_spectrum, plot_norm_spectrum

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
                # intersection points
                {'data': spec_df[spec_df['is_intersect_with_alpha_shape']],
                 'x_key': 'wvl', 'y_key': 'scaled_intensity',
                 'style': 'tilde_AS_alpha_intersect', 'marker': 'x', 'mec': 'tab:orange'},
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


def afs(
        wvl, intensity,
        alpha_radius: float = None,
        continuum_filter_quantile: float = .95,
        is_remove_outliers: bool = True,
        outlier_rolling_window: int = 120,
        outlier_sigma: float = 1,
        outlier_delta_quantile: float = .9,
        primitive_blaze_smoothing: LOCAL_SMOOTHING_METHODS = 'loess',
        final_blaze_smoothing: LOCAL_SMOOTHING_METHODS = 'loess',
        debug: Union[bool, str] = False,
        **kwargs
) -> tuple[np.array, DataFrame]:
    """
    Adaptive Filtering Spectra (AFS) algorithm to normalise the intensity of a spectrum.
    Please read https://iopscience.iop.org/article/10.3847/1538-3881/ab1b47 for more details about this algorithm.
    
    Based on the original AFS algorithm implemented in R, this method offers greater flexibility in the choice of smoothing methods.
    In addition to the standard `loess` (`loess.loess_1d.loess1d`) smoothing,
    this method also supports smoothing using `scipy.interpolate.UnivariateSpline`.
    The smoothing method can be specified using the `primitive_blaze_smoothing` and `final_blaze_smoothing` parameters.
    The arguments for the smoothing methods can be provided in the format of `(stage)_smooth_(arg)`,
    where `(stage)` can be `primitive` or `final`, and `(arg)` can be one of the following:

    - `frac`: Fraction of pixels to consider in the local approximation (for `loess`). Example: `primitive_smooth_frac=0.1`.

    - `degree`: Degree of the local polynomial approximation (for `loess`). Example: `final_smooth_degree=2`.

    - `s`: Positive smoothing factor used to choose the number of knots (for `spline`). Example: `primitive_smooth_s=1e-5`.

    - `k`: Degree of the smoothing spline (for `spline`). Example: `final_smooth_k=3`.

    :param outlier_delta_quantile:
    :param wvl: Spectral wavelength in Angstrom
    :param intensity: Spectral intensity (flux)
    :param alpha_radius: Radius of the alpha-ball used for alpha shape calculation. Default is 1/10 of the wavelength range.
    :param continuum_filter_quantile: Quantile threshold for selecting pixels based on the normalised intensity to refine the blaze function. Default is 0.95.
    :param is_remove_outliers: Whether to remove outliers from the spectrum. Default is True.
    :param outlier_rolling_window: Rolling window size (in pixels) for outlier detection. Default is 120.
    :param outlier_sigma: Sigma threshold for outlier detection. Default is 1.
    :param outlier_delta_quantile: Quantile threshold for outlier detection in the difference between adjacent pixels. Default is 0.9.
    :param primitive_blaze_smoothing: Method for smoothing the primitive blaze function. Options are 'loess' and 'spline'. Default is 'loess'.
    :param final_blaze_smoothing: Method for smoothing the final blaze function. Options are 'loess' and 'spline'. Default is 'loess'.
    :param debug: Enable debugging. If a string path is provided, saves plots to the path.
    :param kwargs: Additional parameters for smoothing algorithms, provided in the format of `(stage)_smooth_(arg)`.
    :return: normalised intensity, DataFrame containing all the intermediate results
    """

    spec_df = pd.DataFrame(
        {'wvl': wvl, 'intensity': intensity}
    ).sort_values(by='wvl').dropna().reset_index(drop=True)
    if spec_df.empty:
        raise ValueError('Input data is either empty or full of NaN values, aborting.')

    wvl_range = spec_df['wvl'].max() - spec_df['wvl'].min()
    # radius of the alpha-ball (alpha shape)
    alpha_radius = alpha_radius or wvl_range / 10

    # step 1: scale the range of intensity and wavelength to be approximately 1:10
    u = wvl_range / 10 / spec_df['intensity'].max()
    spec_df['scaled_intensity'] = spec_df['intensity'] * u

    # step 1.5: remove spectral outliers resulting from cosmic rays or other noise
    # (not part of the original AFS algorithm)
    if is_remove_outliers:
        spec_df = mark_outliers(
            spec_df,
            rolling_window=outlier_rolling_window,
            rolling_window_sigma=outlier_sigma,
            intensity_delta_quantile=outlier_delta_quantile,
            debug=debug
        )
    else:
        spec_df['is_outlier'] = False

    # step 2: find AS_alpha and calculate tilde(AS_alpha)
    spec_df, alpha_shape_df = calc_tilde_AS_alpha(spec_df, alpha_radius, debug)

    # Step 3: smooth tilde(AS_alpha) to estimate the blaze function hat(B_1)
    # (the original work uses local polynomial regression (LOESS) for this step)
    # after smoothing, calculate the primitive normalised intensity y^2 by y / hat(B_1)
    spec_df = calc_primitive_norm_intensity(
        spec_df,
        smoothing_method=primitive_blaze_smoothing,
        debug=debug,
        **{new_key: v for k, v in kwargs.items() if k.startswith('primitive_smooth_')
           for new_key in [k.replace('primitive_smooth_', '')]}
    )

    # step 4: filter spectrum pixels for final refinement of the blaze function
    spec_df = filter_pixels_above_quantiles(
        spec_df, filter_quantile=continuum_filter_quantile, debug=debug)

    # step 5: smooth filtered pixels to estimate the refined blaze function hat(B_2)
    # (the original work also uses local polynomial regression (LOESS) for this step)
    # after smoothing, calculate the final normalised intensity y^3 by y^2 / hat(B_2)
    spec_df = calc_afs_final_norm_intensity(
        spec_df,
        smoothing_method=final_blaze_smoothing,
        debug=debug,
        **{new_key: v for k, v in kwargs.items() if k.startswith('final_smooth_')
           for new_key in [k.replace('final_smooth_', '')]}
    )

    # return the final normalised intensity
    return np.array(spec_df['final_norm_intensity']), spec_df
