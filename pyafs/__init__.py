from .afs import afs

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version('pyafs-astro')
except PackageNotFoundError:
    __version__ = 'unknown'

__all__ = ['afs']
