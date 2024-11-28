import pandas as pd


def scale_intensity(spec_df: pd.DataFrame) -> pd.DataFrame:
    """Scale the intensity of the spectrum."""
    spec_df['scaled_intensity'] = spec_df['intensity'] * (
            (spec_df['wvl'].max() - spec_df['wvl'].min()) /
            (10 * spec_df['intensity'].max())
    )

    return spec_df
