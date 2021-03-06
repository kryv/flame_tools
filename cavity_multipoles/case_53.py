import mplexp as ml
from mplexp import GridData
from mplexp import ExtData
from mplexp import MplPos

fpath = './3DField/'
Epath = fpath+'b53HWR_E.dat'
Hpath = fpath+'b53HWR_H.dat'

no_grid_dump = True
no_expn_dump = False
make_grid_dump = False
make_expn_dump = False

dd = ml.MltExp(Epath,Hpath,suffix='53',scale=5.447,zmirror=True)

if no_grid_dump and no_expn_dump:
    dd.loadfile(skiprows=2)
elif not no_grid_dump and no_expn_dump:
    dd.loaddata()

if no_expn_dump:
    dd.calc_ezaxis()
    dd.calc_multipole()
    dd.calc_linterm()
else:
    dd.loadexpn()

if make_grid_dump and dd.d.dsize is not None:
    dd.dumpdata()

if make_expn_dump and dd.t.Rm is not None:
    dd.dumpexpn()

latline = ml.get_latline(dd.t)
ml.lat_gen('test53new.lat', dd.t, latline, 78.0/238.0, 931.49432, 322.0e6, ek_low=44.977, ek_high=300.0)
