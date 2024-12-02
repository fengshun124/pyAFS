import os.path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd

from pyafs.graphics.utils import set_axis_ticks, export_figure


def plot_alpha_shape(
        spec_df: pd.DataFrame,
        alpha_shape_df: pd.DataFrame,
        debug: Union[bool, str] = False
) -> None:
    """Plot the alpha shape of the spectrum."""
    fig, axis = plt.subplots(figsize=(10, 4), dpi=300)

    # fill the alpha-shape region
    axis.fill_between(
        alpha_shape_df['wvl'], alpha_shape_df['scaled_intensity'],
        color='tab:blue', alpha=.05, lw=0
    )
    axis.plot(
        alpha_shape_df['wvl'], alpha_shape_df['scaled_intensity'],
        'x--', ms=6, mew=1, lw=1, c='tab:blue', alpha=.8, label='alpha shape'
    )

    if spec_df['is_outlier'].any():
        axis.plot(spec_df['wvl'], spec_df['scaled_intensity'],
                  '-', lw=1, c='grey', alpha=.8, label='source spec.')
        cleanup_spec_df = spec_df[~spec_df['is_outlier']]
        axis.plot(cleanup_spec_df['wvl'], cleanup_spec_df['scaled_intensity'],
                  '-', lw=1, c='k', alpha=.8, label='non-outlier spec.')
    else:
        axis.plot(spec_df['wvl'], spec_df['scaled_intensity'],
                  '-', lw=1, c='k', alpha=.8, label='spec.')

    axis.plot(spec_df['wvl'], spec_df['tilde_AS_alpha'],
              '-', lw=2, c='tab:cyan', alpha=.8, label='$\\widetilde{AS}_{\\alpha}$')
    axis.plot(spec_df[spec_df['is_intersect_with_tilde_AS_alpha']]['wvl'],
              spec_df[spec_df['is_intersect_with_tilde_AS_alpha']]['scaled_intensity'],
              'o', ms=6, mew=1, mec='tab:orange', mfc='None', alpha=.8,
              label='$\\widetilde{AS}_{\\alpha}$ $\cap$ spec.')

    axis.legend(fontsize='medium')

    set_axis_ticks(axis)
    axis.tick_params(labelsize='large')
    axis.set_xlabel('wavelength', fontsize='x-large')
    axis.set_ylabel('scaled intensity', fontsize='x-large')
    axis.set_ylim(None, int(axis.get_ylim()[1]) + 1)

    if isinstance(debug, str):
        export_figure(fig, filename=os.path.join(debug, 'alpha_shape.png'))
    elif debug:
        export_figure(fig)
