from typing import Union

import pandas as pd

from AFS._calc import calc_tilde_AS_alpha, apply_local_smoothing, calc_norm_intensity


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
        window_spec_quantile = window_spec_df['norm_intensity1'].quantile(quantile)

        spec_df.loc[window_spec_df.index, 'is_selected_pixel'] = (
                window_spec_df['norm_intensity1'] >= window_spec_quantile)
        quantile_data.append({
            'x': [start_point['wvl'], end_point['wvl']],
            'y': [window_spec_quantile] * 2
        })

    if debug:
        from AFS._plot import plot_norm_spectrum
        plot_data_dict = {
            'plot_obj_dicts': [
                {'data': spec_df, 'x_key': 'wvl', 'y_key': 'norm_intensity1',
                 'label': 'norm. spec.',
                 'ls': '-', 'lw': 1, 'c': 'grey', 'alpha': .8, 'zorder': 1},
                {'data': spec_df[spec_df['is_intersect_with_alpha_shape']],
                 'x_key': 'wvl', 'y_key': 'norm_intensity1',
                 'label': 'tilde AS alpha $\cap$ spec.',
                 'marker': 'x', 'ms': 8, 'mew': 1, 'ls': '', 'c': 'tab:red', 'zorder': 3},
                {'data': spec_df[spec_df['is_selected_pixel']],
                 'x_key': 'wvl', 'y_key': 'norm_intensity1',
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


def afs(
        wvl, intensity, alpha_radius: float = None, q=0.95, d=.25,
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
    alpha_radius = alpha_radius or wvl_range / 6

    # step 2: find AS_alpha and calculate tilde(AS_alpha)
    spec_df, alpha_shape_df = calc_tilde_AS_alpha(spec_df, alpha_radius, debug)

    # step 3: smooth tilde(AS_alpha) using local polynomial regression (LOESS)
    # to estimate the blaze function hat(B_1)
    spec_df = apply_local_smoothing(
        spec_df,
        intensity_key='tilde_AS_alpha',
        smoothed_intensity_key='blaze1',
        frac=d, debug=debug
    )
    # calculate normalised intensity hat(y^1) by y / hat(B_1)
    spec_df = calc_norm_intensity(
        spec_df,
        intensity_key='scaled_intensity',
        blaze_key='blaze1',
        norm_intensity_key='norm_intensity1',
        debug=debug
    )

    # step 4: filter spectrum pixels for next iteration
    spec_df = filter_pixels_above_quantiles(spec_df, q, debug)
