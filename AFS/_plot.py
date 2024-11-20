from typing import Dict, List, Union, Tuple

import matplotlib.pyplot as plt
import pandas as pd

replace_str_dict = {
    'tilde AS alpha': r'$\widetilde{AS}_\alpha$',
}


def export_figure(figure: plt.Figure, filename: str = None):
    """Show or save the figure."""
    if filename is None:
        plt.tight_layout()
        plt.show()
    else:
        figure.savefig(filename, bbox_inches='tight')


def _plot_spec(
        plot_obj_dicts: List[Dict[str, Union[str, float, pd.DataFrame]]],
        aux_plot_obj_dicts: List[Dict[str, Union[str, float, pd.DataFrame]]] = None,
        fig_title: str = None,
) -> Tuple[plt.Figure, plt.Axes]:
    fig, axis = plt.subplots(1, 1, figsize=(10, 4), dpi=300)

    for plot_obj_dict in plot_obj_dicts:
        x = plot_obj_dict['data'][plot_obj_dict['x_key']]
        y = plot_obj_dict['data'][plot_obj_dict['y_key']]
        # remove data, x, y keys from plot_obj_dict
        for key in ['data', 'x_key', 'y_key']:
            plot_obj_dict.pop(key)
        # replace the label with the corresponding string in replace_str_dict
        for key, value in replace_str_dict.items():
            try:
                plot_obj_dict['label'] = plot_obj_dict['label'].replace(key, value)
            except KeyError:
                pass
        axis.plot(x, y, **plot_obj_dict)

    # plot auxiliary lines
    for aux_plot_obj_dict in aux_plot_obj_dicts or []:
        plot_method = axis.axvline if aux_plot_obj_dict['orientation'] == 'v' else axis.axhline
        aux_plot_obj_dict.pop('orientation')
        plot_method(**aux_plot_obj_dict)

    axis.set_xlabel('wavelength', fontsize='x-large')
    axis.set_ylabel('intensity', fontsize='x-large')

    axis.legend(
        *[*zip(*{l: h for h, l in zip(*axis.get_legend_handles_labels())}.items())][::-1],
        prop={'size': 'medium'}, markerscale=1.2
    )

    if fig_title:
        axis.set_title(fig_title, fontsize='x-large')

    axis.tick_params(axis='both', which='both', labelsize='large', direction='in', top=True, right=True)

    return fig, axis


def plot_spectrum(
        plot_obj_dicts: List[Dict[str, Union[str, float, pd.DataFrame]]],
        aux_plot_obj_dicts: List[Dict[str, Union[str, float, pd.DataFrame]]] = None,
        fig_title: str = None, exp_filename: str = None
) -> None:
    """Visualise spectral data."""
    fig, axis = _plot_spec(plot_obj_dicts, aux_plot_obj_dicts, fig_title)
    axis.set_ylim(None, int(axis.get_ylim()[1]) + 1)

    export_figure(fig, exp_filename)


def plot_norm_spectrum(
        plot_obj_dicts: List[Dict[str, Union[str, float, pd.DataFrame]]],
        aux_plot_obj_dicts: List[Dict[str, Union[str, float, pd.DataFrame]]] = None,
        fig_title: str = None, exp_filename: str = None
) -> None:
    """Visualise normalised spectral data with horizontal auxiliary line at intensity=1."""
    fig, axis = _plot_spec(plot_obj_dicts, aux_plot_obj_dicts, fig_title)
    axis.axhline(y=1, ls=':', lw=1, c='k', alpha=.8, zorder=-1)
    axis.set_ylim(-.1, 1.2)

    export_figure(fig, exp_filename)
