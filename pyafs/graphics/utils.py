import os.path
from itertools import product

import matplotlib.pyplot as plt


def export_figure(figure: plt.Figure, filename: str = None):
    """Show or save the figure."""
    if filename is None:
        # plt.tight_layout()
        plt.show()
    else:
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        figure.savefig(filename, bbox_inches='tight')


def set_axis_ticks(
        target_axis: plt.Axes,
        x_ticks_position: str = 'both',
        y_ticks_position: str = 'both',
        x_tick_label_position: str = 'bottom',
        y_tick_label_position: str = 'left',
        x_label_position: str = 'bottom',
        y_label_position: str = 'left',
):
    """Customise the tick and label styles of the axis."""
    # convert the input strings to lower cases to enable case-insensitive comparison
    for prefix, suffix in product(
            ['x_', 'y_'], ['ticks_position', 'tick_label_position', 'label_position']):
        locals()[f'{prefix}{suffix}'] = locals()[f'{prefix}{suffix}'].lower()

    x_tick_label_pos_dict = {
        'top': dict(labeltop=True, labelbottom=False),
        'bottom': dict(labeltop=False, labelbottom=True),
        'both': dict(labeltop=True, labelbottom=True),
        'none': dict(labeltop=False, labelbottom=False),
    }
    y_tick_label_pos_dict = {
        'right': dict(labelright=True, labelleft=False),
        'left': dict(labelright=False, labelleft=True),
        'both': dict(labelright=True, labelleft=True),
        'none': dict(labelright=False, labelleft=False),
    }

    if x_tick_label_position not in x_tick_label_pos_dict:
        raise ValueError(f'invalid x_tick_label_position: \"{x_tick_label_position}\"')
    if y_tick_label_position not in y_tick_label_pos_dict:
        raise ValueError(f'invalid y_tick_label_position: \"{y_tick_label_position}\"')

    for tick_length, tick_type in zip([6, 4], ['major', 'minor']):
        target_axis.tick_params(
            axis='both', direction='in', which=tick_type,
            length=tick_length, pad=4, labelsize=10,
            **{**x_tick_label_pos_dict[x_tick_label_position],
               **y_tick_label_pos_dict[y_tick_label_position]}
        )

    for ax, ticks_position, label_position in zip(
            [target_axis.xaxis, target_axis.yaxis],
            [x_ticks_position, y_ticks_position],
            [x_label_position, y_label_position]
    ):
        ax.set_ticks_position(x_ticks_position)
        ax.set_label_position(label_position)
