import os.path
from typing import Union, Literal

import pandas as pd
from matplotlib import pyplot as plt

from pyafs.graphics.utils import set_axis_ticks, export_figure


def plot_norm_spec(
        spec_df: pd.DataFrame, key_prefix: Literal['primitive', 'final'],
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plot the spectrum in both scaled and normalised."""
    colour_dict = {'primitive': 'tab:red', 'final': 'tab:green'}

    fig, (scaled_ax, norm_ax) = plt.subplots(
        2, 1, figsize=(10, 7), dpi=300, height_ratios=[3, 2],
        gridspec_kw=dict(hspace=.2), sharex=True, sharey=False
    )

    if spec_df['is_outlier'].any():
        scaled_ax.plot(spec_df['wvl'], spec_df['scaled_intensity'],
                       '-', c='grey', lw=1, alpha=.8, label='source spec.')
        norm_ax.plot(spec_df['wvl'], spec_df[f'{key_prefix}_norm_intensity'],
                     '-', c='grey', lw=1, alpha=.8, label='source spec.')

        cleanup_spec_df = spec_df[~spec_df['is_outlier']]
        scaled_ax.plot(cleanup_spec_df['wvl'], cleanup_spec_df['scaled_intensity'],
                       '-', c='k', lw=1, alpha=.8, label='non-outlier spec.')
        norm_ax.plot(cleanup_spec_df['wvl'], cleanup_spec_df[f'{key_prefix}_norm_intensity'],
                     '-', c='k', lw=1, alpha=.8, label='non-outlier spec.')
    else:
        scaled_ax.plot(spec_df['wvl'], spec_df['scaled_intensity'],
                       'k-', lw=1, alpha=.8, label='spec.')
        norm_ax.plot(spec_df['wvl'], spec_df[f'{key_prefix}_norm_intensity'],
                     'k-', lw=1, alpha=.8, label='spec.')

    scaled_ax.plot(spec_df['wvl'], spec_df[f'{key_prefix}_blaze'],
                   '-', c=colour_dict[key_prefix], lw=1, alpha=.8, label=f'{key_prefix} blaze')

    for axis in [scaled_ax, norm_ax]:
        set_axis_ticks(axis)
        axis.tick_params(labelsize='large')

    norm_ax.axhline(1, ls=':', c='k', lw=1, alpha=.8)
    norm_ax.set_ylim(0, 1.2)
    norm_ax.yaxis.set_major_locator(plt.MultipleLocator(.3, offset=.1))

    scaled_ax.set_ylabel('scaled intensity', fontsize='x-large')
    scaled_ax.set_ylim(None, int(scaled_ax.get_ylim()[1]) + 1)

    norm_ax.set_ylabel('normalised intensity', fontsize='x-large')
    norm_ax.set_xlabel('wavelength', fontsize='x-large')

    return fig, [scaled_ax, norm_ax]


def plot_primitive_norm_spec(
        spec_df: pd.DataFrame,
        debug: Union[bool, str] = False
) -> None:
    """Plot the primitive normalised spectrum."""
    fig, (scaled_ax, norm_ax) = plot_norm_spec(spec_df, 'primitive')

    scaled_ax.plot(spec_df['wvl'], spec_df['tilde_AS_alpha'],
                   '-', lw=1, c='tab:cyan', alpha=.8, label='$\\widetilde{AS}_{\\alpha}$')
    scaled_ax.plot(spec_df['wvl'], spec_df['tilde_AS_alpha'] - spec_df['primitive_blaze'],
                   '-', lw=1, c='tab:purple', alpha=.8, label='$\\widetilde{AS}_{\\beta}$ - primitive blaze')
    scaled_ax.axhline(0, ls=':', c='k', lw=1, alpha=.8)
    scaled_ax.plot(spec_df[spec_df['is_intersect_with_tilde_AS_alpha']]['wvl'],
                   spec_df[spec_df['is_intersect_with_tilde_AS_alpha']]['scaled_intensity'],
                   'o', ms=6, mew=1, mec='tab:orange', mfc='None', alpha=.8,
                   label='$\\widetilde{AS}_{\\alpha}$ $\cap$ spec.')

    for axis in [scaled_ax, norm_ax]:
        axis.legend(fontsize='medium', ncol=3)

    if isinstance(debug, str):
        export_figure(fig, filename=os.path.join(debug, 'primitive_norm.png'))
    elif debug:
        export_figure(fig)


def plot_afs_final_norm_spec(
        spec_df: pd.DataFrame,
        debug: Union[bool, str] = False
) -> None:
    """Plot the final normalised spectrum."""
    fig, (scaled_ax, norm_ax) = plot_norm_spec(spec_df, 'final')

    scaled_ax.plot(spec_df[spec_df['is_above_quantile']]['wvl'],
                   spec_df[spec_df['is_above_quantile']]['scaled_intensity'],
                   'o', ms=6, mew=1, mec='tab:green', mfc='None', alpha=.8,
                   label=f'selected pixels')
    scaled_ax.plot(spec_df[spec_df['is_intersect_with_tilde_AS_alpha']]['wvl'],
                   spec_df[spec_df['is_intersect_with_tilde_AS_alpha']]['scaled_intensity'],
                   'x', ms=6, mew=1, mec='tab:orange', mfc='None', alpha=.8,
                   label='intersect with $AS_{\\alpha}$')
    scaled_ax.plot(spec_df['wvl'], spec_df['primitive_blaze'],
                   '-', c='tab:red', lw=1, alpha=.8, label='primitive blaze')

    for axis in [scaled_ax, norm_ax]:
        axis.legend(fontsize='medium', ncol=3)

    if isinstance(debug, str):
        export_figure(fig, filename=os.path.join(debug, 'final_norm.png'))
    elif debug:
        export_figure(fig)
