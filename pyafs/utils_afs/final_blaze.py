from typing import Union

import pandas as pd

from pyafs.graphics.norm_blaze import plot_afs_final_norm_spec
from pyafs.graphics.spec import plot_spec
from pyafs.utils import SMOOTHING_METHODS, smooth_intensity
from pyafs.utils.interpolate import interpolate_intensity


def calc_final_norm_intensity(
        spec_df: pd.DataFrame,
        smoothing_method: SMOOTHING_METHODS = 'loess',
        debug: Union[bool, str] = False,
        **kwargs
) -> pd.DataFrame:
    """Calculate the final normalised intensity of the spectrum."""
    spec_df = spec_df.copy()
    filtered_spec_df = spec_df[spec_df['is_above_quantile']].copy()

    # smooth the intensity values
    filtered_spec_df = smooth_intensity(
        spec_df=filtered_spec_df,
        intensity_key='scaled_intensity',
        smoothed_intensity_key='final_blaze',
        smoothing_method=smoothing_method,
        **kwargs,
    )

    # fill in the outliers with the smoothed values
    spec_df = interpolate_intensity(
        spec_df=spec_df,
        filtered_spec_df=filtered_spec_df,
        intensity_key='final_blaze',
        **kwargs,
    )

    # normalise the intensity values
    spec_df['final_norm_intensity'] = spec_df['scaled_intensity'] / spec_df['final_blaze']

    if debug:
        plot_afs_final_norm_spec(spec_df, debug)
        plot_spec(spec_df, debug)

    return spec_df
