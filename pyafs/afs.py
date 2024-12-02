from typing import Union, Any

import numpy as np
import pandas as pd

from pyafs.utils import SMOOTHING_METHODS
from pyafs.utils import (
    scale_intensity,
    mark_outlier,
    calc_alpha_shape_upper_boundary,
    calc_primitive_norm_intensity,
    filter_pixels_above_quantile,
    calc_final_norm_intensity
)


def afs(
        wvl, intensity,
        alpha_radius: float = None,
        continuum_filter_quantile: float = .95,
        primitive_blaze_smoothing: SMOOTHING_METHODS = 'loess',
        final_blaze_smoothing: SMOOTHING_METHODS = 'loess',
        is_include_intersections: bool = False,
        is_remove_outliers: bool = True,
        outlier_rolling_window: int = 80,
        outlier_rolling_baseline_quantile: float = .8,
        outlier_rolling_mad_scale: float = 1.4,
        outlier_max_iterations: int = 2,
        debug: Union[bool, str] = False,
        **kwargs
) -> tuple[np.ndarray[Any, np.dtype[Any]], pd.DataFrame] | np.ndarray[Any, np.dtype[Any]]:
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

    :param wvl: Spectral wavelength in Angstrom.
    :param intensity: Spectral intensity.
    :param alpha_radius: Radius of the alpha shape (default: 1/10 of the wavelength range).
    :param continuum_filter_quantile: Quantile for filtering pixels near the primary blaze function (default: 0.95).
    :param primitive_blaze_smoothing: Smoothing method for the primitive blaze function (default: 'loess').
    :param final_blaze_smoothing: Smoothing method for the final blaze function (default: 'loess').
    :param is_include_intersections: Whether to include the intersections with the alpha shape in the final normalised intensity (default: False).
    :param is_remove_outliers: Whether to remove outliers before normalisation (default: True).
    :param outlier_rolling_window: Window size for the rolling median filter (default: 80).
    :param outlier_rolling_baseline_quantile: Quantile for constructing the rolling baseline (default: 0.8).
    :param outlier_rolling_mad_scale: Scale factor for the rolling median filter (default: 1).
    :param outlier_max_iterations: Maximum number of iterations for outlier removal (default: 2).
    :param debug: Whether to return the intermediate results for debugging. If a string is provided, the intermediate results will be saved to the directory specified by the string (default: False).
    :param kwargs: Additional parameters for smoothing algorithms, provided in the format of `(stage)_smooth_(arg)`.
    :return: Normalised intensity of the spectrum. If `debug` is True, also return the intermediate results in a DataFrame.
    """
    spec_df = pd.DataFrame({'wvl': wvl, 'intensity': intensity})

    # step 1: scale the range of intensity and wavelength to be approximately 1:10
    spec_df = scale_intensity(spec_df)

    # step 1.5: remove spectral outliers resulting from cosmic rays or other noise
    # (not part of the original AFS algorithm)
    if is_remove_outliers:
        spec_df = mark_outlier(
            spec_df,
            rolling_window=outlier_rolling_window,
            rolling_baseline_quantile=outlier_rolling_baseline_quantile,
            rolling_mad_scale=outlier_rolling_mad_scale,
            max_iterations=outlier_max_iterations,
            debug=debug
        )
    else:
        spec_df['is_outlier'] = False

    # step 2: find AS_alpha and calculate tilde(AS_alpha)
    alpha_radius = alpha_radius or (spec_df['wvl'].max() - spec_df['wvl'].min()) / 10
    spec_df, alpha_shape_df = calc_alpha_shape_upper_boundary(
        spec_df=spec_df,
        alpha_radius=alpha_radius,
        debug=debug
    )

    # step 3: smooth tilde(AS_alpha) to estimate the blaze function hat(B_1)
    # (the original work uses local polynomial regression (LOESS) for this step)
    # after smoothing, calculate the primitive normalised intensity y^2 by y / hat(B_1)
    spec_df = calc_primitive_norm_intensity(
        spec_df=spec_df,
        smoothing_method=primitive_blaze_smoothing,
        debug=debug,
        **{new_key: v for k, v in kwargs.items() if k.startswith('primitive_smooth_')
           for new_key in [k.replace('primitive_smooth_', '')]}
    )

    # step 4: filter pixels above the given quantile for refining the blaze function
    spec_df, quantile_dfs = filter_pixels_above_quantile(
        spec_df=spec_df,
        filter_quantile=continuum_filter_quantile,
        debug=debug
    )

    # step 5: smooth filtered pixels in step 4 to estimate the refined blaze function hat(B_2)
    # (the original work also uses local polynomial regression (LOESS) for this step)
    # the flag `is_include_intersections` determines whether to
    # include intersections of tilde(AS_alpha) with the spectrum when smoothing the final blaze function,
    # potentially improving continuum recovery at the spectrum's edges.
    # after smoothing, calculate the final normalised intensity y^3 by y^2 / hat(B_2)
    spec_df = calc_final_norm_intensity(
        spec_df=spec_df,
        smoothing_method=final_blaze_smoothing,
        is_include_intersections=is_include_intersections,
        debug=debug,
        **{new_key: v for k, v in kwargs.items() if k.startswith('final_smooth_')
           for new_key in [k.replace('final_smooth_', '')]}
    )

    final_norm_intensity = spec_df['final_norm_intensity']

    if debug:
        return np.array(final_norm_intensity), spec_df
    else:
        return np.array(final_norm_intensity)
