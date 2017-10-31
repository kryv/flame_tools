import mplexp as ml
from mplexp import GridData
from mplexp import ExtData
from mplexp import MplPos

fpath = 'MEBT_bunchers/'
Epath = fpath+'MEBT_buncher_E-field_X.txt'
Hpath = fpath+'MEBT_buncher_H-field_X.txt'

no_grid_dump = False
no_expn_dump = False
make_grid_dump = False
make_expn_dump = False
force_cals = False

ml.cx = 2; ml.cy = 1; ml.cz = 0;
ml.xR = 5; ml.yR = 4; ml.zR = 3;
ml.xI = 8; ml.yI = 7; ml.zI = 6;

dd = ml.MltExp(Epath,Hpath,suffix='38',scale=0.1635,zmirror=False)
dd.nrad = 40
dd.nphi = 40

if no_grid_dump and no_expn_dump:
    dd.loadfile(skiprows=2)
elif not no_grid_dump and no_expn_dump:
    dd.loaddata()

if no_expn_dump or force_cals:
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
ml.lat_gen('test38s.lat', dd.t, latline, 33.0/238.0, 931.49432, 80.5e6, ek_low=0.2, ek_high=1.0, step=400)
