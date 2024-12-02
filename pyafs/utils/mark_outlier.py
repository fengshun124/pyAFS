from typing import Union

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from pyafs.graphics import plot_outliers
from pyafs.utils.interpolate import interpolate_intensity
from pyafs.utils.smooth import smooth_intensity


def mark_outlier(
        spec_df: pd.DataFrame,
        rolling_window: int,
        rolling_baseline_quantile: float,
        rolling_mad_scale: float,
        max_iterations: int,
        debug: Union[bool, str] = False
) -> pd.DataFrame:
    """Mark the outliers in the spectrum."""
    spec_df = spec_df.copy()

    spec_df['is_outlier'] = False
    outlier_mask_iter = np.zeros_like(spec_df['is_outlier'], dtype=bool)
    for iteration in range(max_iterations):
        non_outlier_mask_iter = ~spec_df['is_outlier']
        filtered_spec_df = spec_df[non_outlier_mask_iter].copy()

        filtered_spec_df['scaled_intensity_rolling_baseline'] = filtered_spec_df['scaled_intensity'].rolling(
            window=rolling_window, center=True, min_periods=1).quantile(rolling_baseline_quantile)
        filtered_spec_df['scaled_intensity_rolling_mad'] = filtered_spec_df['scaled_intensity'].rolling(
            window=rolling_window, center=True, min_periods=1).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True)

        for col_key in ['scaled_intensity_rolling_baseline', 'scaled_intensity_rolling_mad']:
            spec_df = smooth_intensity(
                spec_df=interpolate_intensity(
                    spec_df=spec_df, filtered_spec_df=filtered_spec_df, intensity_key=col_key
                ),
                intensity_key=col_key,
                smoothed_intensity_key=col_key,
                smoothing_method='loess',
            )

        mad_outlier_mask_iter = (spec_df['scaled_intensity'] > (
                spec_df['scaled_intensity_rolling_baseline'] +
                rolling_mad_scale * spec_df['scaled_intensity_rolling_mad']
        ))
        spec_df.loc[mad_outlier_mask_iter, 'is_outlier'] = True

        # find peaks in remaining non-outliers
        peak_indices, _ = find_peaks(
            spec_df.loc[non_outlier_mask_iter, 'scaled_intensity'],
        )
        spec_df.loc[spec_df.index[non_outlier_mask_iter][peak_indices], 'is_outlier'] = True
        outlier_mask_iter |= spec_df['is_outlier']

    if debug:
        plot_outliers(spec_df, rolling_mad_scale, debug)

    return spec_df
