import os.path
from typing import Union

import pandas as pd

from pyafs.graphics.norm_blaze import plot_norm_spec
from pyafs.graphics.utils import export_figure


def plot_filtered_pixels(
        spec_df: pd.DataFrame,
        filter_quantile: float,
        quantile_dfs: list[pd.DataFrame],
        debug: Union[bool, str] = False
) -> None:
    """Plot the filtered pixels."""
    fig, (scaled_ax, norm_ax) = plot_norm_spec(spec_df, 'primitive')

    for axis, key_prefix in zip([scaled_ax, norm_ax], ['scaled_', 'primitive_norm_']):
        axis.plot(spec_df[spec_df['is_above_quantile']]['wvl'],
                  spec_df[spec_df['is_above_quantile']][f'{key_prefix}intensity'],
                  'o', ms=6, mew=1, mec='tab:green', mfc='None', alpha=.8,
                  label=f'pixels above {filter_quantile:.0%} quantile')

    for quantile_df in quantile_dfs:
        norm_ax.plot(quantile_df['wvl'], quantile_df['primitive_norm_intensity'],
                     '-', c='magenta', lw=1, label=f'{filter_quantile:.0%} quantile')

    for axis in [scaled_ax, norm_ax]:
        axis.legend(
            *[*zip(*{l: h for h, l in zip(*axis.get_legend_handles_labels())}.items())][::-1],
            fontsize='medium'
        )

    if isinstance(debug, str):
        export_figure(fig, filename=os.path.join(debug, 'filtered_pixels.png'))
    elif debug:
        export_figure(fig)
