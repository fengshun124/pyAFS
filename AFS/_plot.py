import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = 'monospace'


def export_figure(figure: plt.Figure, filename: str = None):
    """Show or save the figure."""
    if filename is None:
        plt.tight_layout()
        plt.show()
    else:
        figure.savefig(filename, bbox_inches='tight')


def plot_alpha_shape(
        spec_df: pd.DataFrame, boundary_df: pd.DataFrame, upper_boundary_df: pd.DataFrame,
        exp_filename: str = None
) -> None:
    """Plot the alpha shape of the spectrum, highlighting its upper boundary."""
    fig, axis = plt.subplots(1, 1, figsize=(8, 4), dpi=300)

    axis.plot(spec_df['wvl'], spec_df['scaled_intensity'], '-',
              c='tab:grey', lw=1, alpha=.8, label='spec.')
    axis.plot(boundary_df['wvl'], boundary_df['scaled_intensity'],
              'x-', c='tab:red', mew=2, ms=8, lw=1, alpha=.8, label=r'$\alpha$-shape')
    axis.plot(upper_boundary_df['wvl'], upper_boundary_df['scaled_intensity'],
              '+', c='tab:green', mew=2, ms=8, label=r'upper $\alpha$-shape')

    axis.set_xlabel('wavelength', fontsize='large')
    axis.set_ylabel('intensity', fontsize='large')

    axis.legend(prop={'size': 'medium'})

    axis.tick_params(axis='both', which='both', labelsize='large', direction='in',
                     top=True, right=True)
    axis.set_ylim(None, int(axis.get_ylim()[1]) + 1)

    export_figure(fig, exp_filename)
