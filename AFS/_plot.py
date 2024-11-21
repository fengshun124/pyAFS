from typing import Dict, List, Union, Tuple

import matplotlib.pyplot as plt
import pandas as pd

plot_label_converter = {
    'tilde_AS_alpha': r'$\widetilde{AS}_\alpha$',
}

plot_style_dict = {
    'alpha_shape': {
        'c': 'tab:orange', 'ls': ':', 'lw': 1,
        'marker': 'x', 'ms': 6, 'mec': 'tab:orange', 'mew': 1, 'mfc': 'None',
        'alpha': 1, 'label': r'$\alpha$-shape'
    },
    'tilde_AS_alpha': {
        'c': 'tab:green', 'ls': ':', 'lw': 1,
        'alpha': 1, 'label': r'$\widetilde{AS}_\alpha$'
    },
    'tilde_AS_alpha_intersect': {
        'ls': '', 'lw': 1,
        'marker': '+', 'ms': 6, 'mec': 'tab:green', 'mew': 1, 'mfc': 'None',
        'alpha': 1, 'label': r'$\widetilde{AS}_\alpha \cap \text{spec.}$'
    },
    'src_spec': {'c': 'grey', 'ls': '-', 'lw': 1, 'alpha': .4, 'label': 'spec. (source)'},
    'clean_spec': {'c': 'k', 'ls': '-', 'lw': 1, 'alpha': .8, 'label': 'spec. (cleaned)'},
    'outlier': {
        'ls': '', 'lw': 1,
        'marker': 'x', 'ms': 6, 'mec': 'grey', 'mew': 1, 'mfc': 'None',
        'alpha': .4, 'label': 'spec. outliers'
    },
    'selected_pixel': {
        'ls': '', 'lw': 1,
        'marker': '+', 'ms': 6, 'mec': 'tab:green', 'mew': 1, 'mfc': 'None',
        'alpha': 1, 'label': 'selected pixels'
    },
    'line': {'ls': '-', 'lw': 1, 'alpha': .8},
    'line_with_marker': {
        'ls': ':', 'lw': 1, 'marker': 'x', 'ms': 6, 'mec': 'tab:orange', 'mew': 1, 'mfc': 'None'
    },
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
        # extract x and y data from plot_obj_dict
        x = plot_obj_dict['data'][plot_obj_dict['x_key']]
        y = plot_obj_dict['data'][plot_obj_dict['y_key']]

        # extract plot style from plot_obj_dict
        if plot_obj_dict['style'] in plot_style_dict.keys():
            plot_arg_dict = plot_style_dict[plot_obj_dict['style']]
        else:
            if isinstance(plot_obj_dict['style'], dict):
                plot_arg_dict = plot_obj_dict['style']
            else:
                raise ValueError(f'Invalid plot style: {plot_obj_dict["style"]}')
        # remove unnecessary keys from plot_obj_dict
        for key in ['data', 'x_key', 'y_key', 'style']:
            plot_obj_dict.pop(key)
        # update plot_arg_dict if specific plot arguments are provided in plot_obj_dict
        plot_arg_dict.update(plot_obj_dict)
        # update plot label if it contains string patterns in plot_label_converter
        if 'label' in plot_arg_dict.keys():
            for k, v in plot_label_converter.items():
                plot_arg_dict['label'] = plot_arg_dict['label'].replace(k, v)

        axis.plot(x, y, **plot_arg_dict)

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
