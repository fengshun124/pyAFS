# pyAFS: A Python implementation for Alpha-shape Fitting to Spectrum (AFS) algorithm

`pyAFS` is a Python package that offers a third-party implementation of the Alpha-shape Fitting to Spectrum (AFS) algorithm.
Originally developed in R by [Xu *et. al.* (2019)](https://iopscience.iop.org/article/10.3847/1538-3881/ab1b47), AFS provides a data-driven method for continuum fitting and spectrum normalisation.
The original R implementation can be found on [GitHub](https://github.com/xinxuyale/AFS).

Please note that `pyAFS` is independently developed and maintained.
For any bugs or issues related to `pyAFS`, please report them in this repository.

If you find this package useful, please consider citing the original paper:

```latex
@ARTICLE{2019AJ....157..243X,
       author = {{Xu}, Xin and {Cisewski-Kehe}, Jessi and {Davis}, Allen B. and {Fischer}, Debra A. and {Brewer}, John M.},
        title = "{Modeling the Echelle Spectra Continuum with Alpha Shapes and Local Regression Fitting}",
      journal = {\aj},
     keywords = {instrumentation: spectrographs, methods: data analysis, methods: statistical, techniques: radial velocities, techniques: spectroscopic, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2019,
        month = jun,
       volume = {157},
       number = {6},
          eid = {243},
        pages = {243},
          doi = {10.3847/1538-3881/ab1b47},
archivePrefix = {arXiv},
       eprint = {1904.10065},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019AJ....157..243X},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Installation

To install `pyAFS`, you can use `pip`:

```bash
pip install pyafs-astro
```

Alternatively, you can clone this repository and install the package locally:

```bash
git clone https://github.com/fengshun124/pyAFS.git
cd pyAFS
pip install .
```

## Dependencies

`pyAFS` requires Python 3.9 or later, as well as the following packages:

```plaintext
numpy
pandas
alphashape
loess
scipy
shapely
matplotlib  
```

## Usage

A simple example of using `afs` is shown below:

```python
import pandas as pd
from pyafs import afs

# load the spectrum data
data = pd.read_csv('spectrum.csv')

# normalise the spectrum
normalised_flux = afs(data['wavelength'], data['flux'])
```

A more detailed example can be found in the [example notebook](examples/afs.ipynb).
