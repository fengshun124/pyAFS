from typing import Literal

import pandas as pd
from loess.loess_1d import loess_1d
from scipy.interpolate import UnivariateSpline

SMOOTHING_METHODS = Literal['loess', 'spline']
_SMOOTHING_METHODS = ('loess', 'spline')


def extract_smoothing_args(
        method: SMOOTHING_METHODS, **kwargs
) -> dict:
    """Extract the smoothing arguments for the specified method."""
    if method == 'loess':
        return {
            'frac': kwargs.get('frac', .25),
            'degree': kwargs.get('degree', 2),
        }
    elif method == 'spline':
        return {
            'k': kwargs.get('k', 2),
            's': kwargs.get('s', 0),
        }
    else:
        raise ValueError(f'Unknown smoothing method: {method}, '
                         f'expected one of {_SMOOTHING_METHODS}')


def smooth_intensity(
        spec_df: pd.DataFrame,
        intensity_key: str,
        smoothed_intensity_key: str,
        smoothing_method: SMOOTHING_METHODS = 'loess',
        **kwargs
) -> pd.DataFrame:
    """Smooth the intensity values of the spectrum with the specified method."""
    spec_df = spec_df.copy()

    # extract the smoothing arguments
    smoothing_args = extract_smoothing_args(smoothing_method, **kwargs)

    if smoothing_method == 'loess':
        spec_df[smoothed_intensity_key] = loess_1d(
            spec_df['wvl'].to_numpy(), spec_df[intensity_key].to_numpy(),
            **smoothing_args
        )[1]
    elif smoothing_method == 'spline':
        spec_df[smoothed_intensity_key] = UnivariateSpline(
            spec_df['wvl'], spec_df[intensity_key],
            **smoothing_args
        )(spec_df['wvl'])
    else:
        raise ValueError(f'Unknown smoothing method: {smoothing_method}, '
                         f'expected one of {_SMOOTHING_METHODS}')

    return spec_df
