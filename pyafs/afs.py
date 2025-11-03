from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from numpy import dtype, ndarray

from pyafs.core import (
    SMOOTHING_METHODS,
    calc_alpha_shape_upper_boundary,
    calc_final_norm_intensity,
    calc_primitive_norm_intensity,
    filter_pixels_above_quantile,
    mark_outlier,
    scale_intensity,
)


def afs(
    wvl: Union[ndarray[Any, dtype[Any]], List[float]],
    intensity: Union[ndarray[Any, dtype[Any]], List[float]],
    intensity_err: Optional[Union[ndarray[Any, dtype[Any]], List[float]]] = None,
    alpha_radius: Optional[float] = None,
    continuum_filter_quantile: float = 0.95,
    primitive_blaze_smoothing: SMOOTHING_METHODS = "loess",
    final_blaze_smoothing: SMOOTHING_METHODS = "loess",
    is_include_intersections: bool = False,
    is_remove_outliers: bool = True,
    outlier_rolling_window: int = 80,
    outlier_rolling_baseline_quantile: float = 0.8,
    outlier_rolling_mad_scale: float = 1.4,
    outlier_max_iterations: int = 2,
    plot: Union[bool, str] = False,
    **kwargs,
) -> Dict[str, Union[ndarray[Any, dtype[Any]], pd.DataFrame]]:
    """
    Normalize spectral intensity using the Alpha-shape Fitting to Spectrum (AFS) algorithm.

    This implementation extends the original AFS algorithm
    (Xu et al. 2019, https://iopscience.iop.org/article/10.3847/1538-3881/ab1b47)
    with flexible smoothing methods.
    In addition to LOESS smoothing, this method supports scipy's UnivariateSpline interpolation.

    Smoothing parameters are specified using the format `(stage)_smooth_(arg)`, where:
    - `(stage)` is either `primitive` or `final`
    - `(arg)` depends on the smoothing method:
      - LOESS: `frac` (local window fraction), `degree` (polynomial degree)
      - Spline: `s` (smoothing factor), `k` (spline degree)

    Examples:
        - `primitive_smooth_frac=0.1` sets the LOESS window fraction for primitive blaze
        - `final_smooth_s=1e-5` sets the spline smoothing factor for final blaze

    :param wvl: Wavelength array or list in Angstroms.
    :param intensity: Spectral intensity array or list.
    :param intensity_err: Spectral intensity uncertainty array or list (optional).
    :param alpha_radius: Alpha shape radius. Defaults to 1/10 of the wavelength range.
    :param continuum_filter_quantile: Quantile threshold for continuum pixel selection (default: 0.95).
    :param primitive_blaze_smoothing: Smoothing method for the initial blaze function (default: 'loess').
    :param final_blaze_smoothing: Smoothing method for the refined blaze function (default: 'loess').
    :param is_include_intersections: Include alpha shape-spectrum intersections in final blaze estimation (default: False).
    :param is_remove_outliers: Apply outlier removal before normalization (default: True).
    :param outlier_rolling_window: Rolling window size for outlier detection (default: 80).
    :param outlier_rolling_baseline_quantile: Baseline quantile for outlier detection (default: 0.8).
    :param outlier_rolling_mad_scale: MAD scaling factor for outlier thresholding (default: 1.4).
    :param outlier_max_iterations: Maximum outlier removal iterations (default: 2).
    :param plot: Enable diagnostic plotting. If string, specifies output directory. If True, displays plots (default: False).
    :param kwargs: Additional smoothing parameters in `(stage)_smooth_(arg)` format.

    :return: Dictionary with keys:
        - 'norm_intensity': Normalized spectral intensity (numpy array)
        - 'blaze': Estimated continuum/blaze function in original intensity scale (numpy array)
        - 'norm_intensity_err': Normalized intensity uncertainty (numpy array, if intensity_err provided)
        - 'debug': DataFrame with intermediate results
    """

    spec_df = pd.DataFrame(
        {
            "wvl": np.asarray(wvl),
            "intensity": np.asarray(intensity),
            "intensity_err": np.asarray(intensity_err)
            if intensity_err is not None
            else np.zeros_like(np.asarray(intensity)),
        }
    )

    # step 1: scale the range of intensity and wavelength to be approximately 1:10
    spec_df, scaling_factor = scale_intensity(spec_df)

    # step 1.5: remove spectral outliers resulting from cosmic rays or other noise
    # (not part of the original AFS algorithm)
    if is_remove_outliers:
        spec_df = mark_outlier(
            spec_df,
            rolling_window=outlier_rolling_window,
            rolling_baseline_quantile=outlier_rolling_baseline_quantile,
            rolling_mad_scale=outlier_rolling_mad_scale,
            max_iterations=outlier_max_iterations,
            plot=plot,
        )
    else:
        spec_df["is_outlier"] = False

    # step 2: find AS_alpha and calculate tilde(AS_alpha)
    wvl_range = spec_df["wvl"].max() - spec_df["wvl"].min()
    alpha_radius = alpha_radius or wvl_range / 10
    spec_df, alpha_shape_df = calc_alpha_shape_upper_boundary(
        spec_df=spec_df,
        alpha_radius=alpha_radius,
        plot=plot,
    )

    # step 3: smooth tilde(AS_alpha) to estimate the blaze function hat(B_1)
    # (the original work uses local polynomial regression (LOESS) for this step)
    # after smoothing, calculate the primitive normalised intensity y^2 by y / hat(B_1)
    primitive_smooth_params = {
        key.replace("primitive_smooth_", ""): value
        for key, value in kwargs.items()
        if key.startswith("primitive_smooth_")
    }
    spec_df = calc_primitive_norm_intensity(
        spec_df=spec_df,
        smoothing_method=primitive_blaze_smoothing,
        plot=plot,
        **primitive_smooth_params,
    )

    # step 4: filter pixels above the given quantile for refining the blaze function
    spec_df, quantile_dfs = filter_pixels_above_quantile(
        spec_df=spec_df,
        filter_quantile=continuum_filter_quantile,
        plot=plot,
    )

    # step 5: smooth filtered pixels in step 4 to estimate the refined blaze function hat(B_2)
    # (the original work also uses local polynomial regression (LOESS) for this step)
    # the flag `is_include_intersections` determines whether to
    # include intersections of tilde(AS_alpha) with the spectrum when smoothing the final blaze function,
    # potentially improving continuum recovery at the edges of the spectrum.
    # after smoothing, calculate the final normalised intensity y^3 by y^2 / hat(B_2)
    final_smooth_params = {
        key.replace("final_smooth_", ""): value
        for key, value in kwargs.items()
        if key.startswith("final_smooth_")
    }

    spec_df = calc_final_norm_intensity(
        spec_df=spec_df,
        smoothing_method=final_blaze_smoothing,
        is_include_intersections=is_include_intersections,
        debug=plot,
        **final_smooth_params,
    )

    # calculate the continuum in original intensity scale
    spec_df["continuum"] = spec_df["final_blaze"] / scaling_factor

    result: Dict[str, Union[ndarray[Any, dtype[Any]], pd.DataFrame]] = {
        "norm_intensity": np.array(spec_df["final_norm_intensity"]),
        "blaze": np.array(spec_df["continuum"]),
        "debug": spec_df,
    }

    # calculate the final normalised intensity error, if provided
    if intensity_err is not None:
        result["norm_intensity_err"] = np.array(
            spec_df["scaled_intensity_err"] / spec_df["final_blaze"]
        )

    return result
