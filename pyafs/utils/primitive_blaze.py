from typing import Union

import pandas as pd

from pyafs.graphics.norm_blaze import plot_primitive_norm_spec
from pyafs.utils.interpolate import interpolate_intensity
from pyafs.utils.smooth import SMOOTHING_METHODS, smooth_intensity


def calc_primitive_norm_intensity(
        spec_df: pd.DataFrame,
        smoothing_method: SMOOTHING_METHODS = 'loess',
        debug: Union[bool, str] = False,
        **kwargs
) -> pd.DataFrame:
    """Calculate the primitive normalised intensity of the spectrum."""
    spec_df = spec_df.copy()
    filtered_spec_df = spec_df[~spec_df['is_outlier']].copy()

    # smooth the intensity values
    filtered_spec_df = smooth_intensity(
        spec_df=filtered_spec_df,
        intensity_key='tilde_AS_alpha',
        smoothed_intensity_key='primitive_blaze',
        smoothing_method=smoothing_method,
        **kwargs,
    )

    # fill in the outliers with the smoothed values
    spec_df = interpolate_intensity(
        spec_df=spec_df,
        filtered_spec_df=filtered_spec_df,
        intensity_key='primitive_blaze',
        **kwargs,
    )

    # normalise the intensity values
    spec_df['primitive_norm_intensity'] = spec_df['scaled_intensity'] / spec_df['primitive_blaze']

    if debug:
        plot_primitive_norm_spec(spec_df, debug)

    return spec_df
