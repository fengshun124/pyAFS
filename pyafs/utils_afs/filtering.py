from typing import Union

import numpy as np
import pandas as pd

from pyafs.graphics.filtering import plot_filtered_pixels


def filter_pixels_above_quantile(
        spec_df: pd.DataFrame,
        filter_quantile: float,
        debug: Union[bool, str] = False
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Filter out pixels above the specified quantile."""
    spec_df = spec_df.copy()

    if np.sum(spec_df['is_intersect_with_tilde_AS_alpha']) < 2:
        raise ValueError('Expected at least 2 pixels to intersect with the alpha shape, '
                         f'found {np.sum(spec_df["is_intersect_with_alpha_shape"])}')

    spec_df['is_above_quantile'] = False
    intersect_wvl = spec_df.loc[spec_df['is_intersect_with_tilde_AS_alpha'], 'wvl']
    quantile_dfs = []
    # mark pixels above the quantile of windows formed by the intersecting points
    for start_wvl, end_wvl in zip(intersect_wvl[:-1], intersect_wvl[1:]):
        window_mask = (
                (spec_df['wvl'] >= start_wvl) &
                (spec_df['wvl'] <= end_wvl) &
                ~spec_df['is_outlier']
        )

        # skip if there are no pixels in the window
        if not window_mask.any():
            continue
        # calculate the quantile of the window
        intensity_q_threshold = (
            spec_df.loc[window_mask, 'primitive_norm_intensity'].quantile(filter_quantile)
        )
        quantile_dfs.append(pd.DataFrame({
            'wvl': [start_wvl, end_wvl],
            'primitive_norm_intensity': [intensity_q_threshold] * 2
        }))

        # update the selection mask
        spec_df.loc[window_mask, 'is_above_quantile'] = (
                spec_df.loc[window_mask, 'primitive_norm_intensity'] > intensity_q_threshold
        )

    if debug:
        plot_filtered_pixels(spec_df, filter_quantile, quantile_dfs, debug)

    return spec_df, quantile_dfs
