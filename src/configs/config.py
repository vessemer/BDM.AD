from pathlib import Path
import addict

PATHS = addict.Dict()
PATHS.DATA = Path('../data')
PATHS.ASSETS = PATHS.DATA / 'assets'
PATHS.PEAKS = PATHS.DATA / 'peaks'
PATHS.SMINER = PATHS.DATA / 'sminer'
