from typing import Tuple

import pandas as pd


def scale_intensity(spec_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """Scale the intensity of the spectrum."""
    scaling_factor = (spec_df["wvl"].max() - spec_df["wvl"].min()) / (
        10 * spec_df["intensity"].max()
    )

    spec_df["scaled_intensity"] = spec_df["intensity"] * scaling_factor

    # scale the intensity error
    if "intensity_err" in spec_df:
        spec_df["scaled_intensity_err"] = spec_df["intensity_err"] * scaling_factor

    return spec_df, scaling_factor
