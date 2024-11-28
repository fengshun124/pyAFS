import pandas as pd
from scipy.interpolate import interp1d


def extract_interpolate_args(
        **kwargs
) -> dict:
    """Extract the interpolation arguments."""
    return {
        'kind': kwargs.get('interp_kind', 'quadratic'),
        'fill_value': kwargs.get('fill_value', 'extrapolate'),
    }


def interpolate_intensity(
        spec_df: pd.DataFrame,
        filtered_spec_df: pd.DataFrame,
        intensity_key: str,
        **kwargs
):
    """Interpolate the intensity values of the spectrum."""
    spec_df = spec_df.copy()
    filtered_spec_df = filtered_spec_df.copy()

    # extract the interpolation arguments
    interp_args = extract_interpolate_args(**kwargs)

    # interpolate the intensity values
    spec_df[intensity_key] = interp1d(
        filtered_spec_df['wvl'], filtered_spec_df[intensity_key], **interp_args
    )(spec_df['wvl'])

    return spec_df
