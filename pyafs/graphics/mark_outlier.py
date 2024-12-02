import os.path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd

from pyafs.graphics.utils import set_axis_ticks, export_figure


def plot_outliers(
        spec_df: pd.DataFrame, rolling_mad_scale: float,
        debug: Union[bool, str] = False
):
    """Plot the outliers in the spectrum."""
    fig, axis = plt.subplots(figsize=(10, 4), dpi=300)

    outlier_df = spec_df[spec_df['is_outlier']]
    axis.plot(outlier_df['wvl'], outlier_df['scaled_intensity'],
              'x', ms=6, mew=1, c='tab:red', alpha=.8, label='outliers')
    cleanup_spec_df = spec_df[~spec_df['is_outlier']]
    axis.plot(cleanup_spec_df['wvl'], cleanup_spec_df['scaled_intensity'],
              '.', ms=6, mew=0, mfc='k', alpha=.6, label='cleaned spec.')

    axis.plot(spec_df['wvl'], spec_df['scaled_intensity_rolling_baseline'],
              color='tab:green', ls='--', lw=1, label='rolling baseline')
    y_upper_lim = axis.get_ylim()[1]
    axis.fill_between(
        spec_df['wvl'],
        spec_df['scaled_intensity_rolling_baseline'] + rolling_mad_scale * spec_df['scaled_intensity_rolling_mad'],
        y_upper_lim, hatch='////', edgecolor='tab:orange', facecolor='none',
        zorder=-1, alpha=.2,
        label='outlier threshold'
    )
    axis.set_ylim(None, y_upper_lim)


    axis.legend(loc='upper right', fontsize='medium')

    set_axis_ticks(axis)
    axis.tick_params(labelsize='large')
    axis.set_xlabel('wavelength', fontsize='x-large')
    axis.set_ylabel('scaled intensity', fontsize='x-large')

    if isinstance(debug, str):
        export_figure(fig, filename=os.path.join(debug, 'outliers.png'))
    elif debug:
        export_figure(fig)
