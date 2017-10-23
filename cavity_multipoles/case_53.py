import mlpexp as ml

fpath = '../test/3DField/'
Epath = fpath+'b53HWR_E.dat'
Hpath = fpath+'b53HWR_H.dat'

no_grid_dump = True
no_expn_dump = True
make_grid_dump = False
make_expn_dump = True

dd = ml.MltExp(Epath,Hpath,suffix='53',scale=5.447,zmirror=True)

if no_grid_dump and no_expn_dump:
    dd.loadfile(skiprows=2)
elif not no_grid_dump() and no_expn_dump:
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

pos1 = 160.099
posm = 250.0
pos2 = 339.901
pose = 500.0

latline = [
           {'name':'EF0001','type':'EFocus','range':[0.0, pos1]},
           {'name':'EF0002','type':'EFocus','range':[pos1, posm]},
           {'name':'EF0003','type':'EFocus','range':[posm, pos2]},
           {'name':'EF0004','type':'EFocus','range':[pos2, pose]},
           {'name':'EQ0001','type':'EQuad' ,'range':[0.0, posm]},
           {'name':'EQ0002','type':'EQuad' ,'range':[posm, pose]},
           {'name':'HM0001','type':'HMono' ,'range':[0.0, posm]},
           {'name':'HM0002','type':'HMono' ,'range':[posm, pose]},
           {'name':'HQ0001','type':'HQuad' ,'range':[0.0, posm]},
           {'name':'HQ0002','type':'HQuad' ,'range':[posm, pose]},
           {'name':'AC0001','type':'AccGap','range':[0.0, posm]},
           {'name':'AC0002','type':'AccGap','range':[posm, pose]}
         ]

ml.lat_gen('test53.lat', dd.t, latline, 78.0/238.0, 931.49432, 322.0e6, ek_low=44.977, ek_high=300.0)
