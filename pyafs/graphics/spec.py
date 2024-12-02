import os.path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd

from pyafs.graphics.utils import set_axis_ticks, export_figure


def plot_norm_spec(
        spec_df: pd.DataFrame,
        debug: Union[bool, str] = False
) -> None:
    """Plot the spectrum."""
    fig, axis = plt.subplots(1, 1, figsize=(10, 4), dpi=300)

    axis.plot(spec_df['wvl'], spec_df['primitive_norm_intensity'],
              '-', c='tab:red', lw=1, alpha=.8, label='spec.')
    axis.plot(spec_df['wvl'], spec_df['final_norm_intensity'],
              '-', c='tab:green', lw=1, alpha=.8, label='spec.')

    axis.axhline(1, ls=':', c='k', lw=1, alpha=.8)

    axis.set_ylim(0, 1.2)
    axis.yaxis.set_major_locator(plt.MultipleLocator(.3, offset=.1))

    set_axis_ticks(axis)
    axis.tick_params(labelsize='large')
    axis.set_xlabel('wavelength', fontsize='x-large')
    axis.set_ylabel('scaled intensity', fontsize='x-large')

    if isinstance(debug, str):
        export_figure(fig, filename=os.path.join(debug, 'final_norm.png'))
    elif debug:
        export_figure(fig)
