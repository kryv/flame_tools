from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

"""
3D field multipole expansion and FLAME input generation
"""

clight = 299792458.0
cx = 0; cy = 1; cz = 2;
xR = 3; yR = 4; zR = 5;
xI = 6; yI = 7; zI = 8;

import scipy as np
from scipy.optimize import leastsq

try :
    import cPickle as pkl
except:
    import pickle as pkl

class GridData(object):
    """Class for storing raw grid data
    """
    def __init__(self):
        self.efld = None
        self.hfld = None
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        self.zmin = None
        self.zmax = None

        self.dsize = None
        self.xsize = None
        self.ysize = None
        self.zsize = None

        self.dx = None
        self.dy = None
        self.dz = None

class ExtData(object):
    """Class for storing multipole terms which expanded from grid data
    """
    def __init__(self):
        self.zaxis = None
        self.ezaxis = None
        self.coefEr = None
        self.coefEp = None
        self.coefHr = None
        self.coefHp = None
        self.dz = None
        self.Rm = None

        self.Er12 = None
        self.Er21 = None
        self.Er23 = None
        self.Hp12 = None
        self.Hp21 = None
        self.Hp23 = None

class MplPos:
    """Class for storing field maximum and multipole matrix
    """
    def __init__(self, z, fmax, mat):
        self.z = z
        self.fmax = fmax
        self.mat = mat
        if np.isnan(mat).any():
            self.mat = np.zeros(mat.shape)

    def copy(self):
        mlp_copy = MplPos(self.z, self.fmax, self.mat.copy())
        return mlp_copy

class MltExp(object):
    """Class for calculating multipole terms form grid data
    """
    def __init__(self, efile=None, hfile=None, suffix='', scale=1.0, zmirror=False, rnorm=None):
        self.efile = efile
        self.hfile = hfile
        self.scale = scale
        self.suffix = str(suffix)
        self.zmirror = zmirror
        self.d = GridData()
        self.t = ExtData()
        self.nrad = 40
        self.nphi = 40
        self.orad = 4
        self.ophi = 4
        self.rnorm = rnorm

    def loadfile(self,skiprows=2):
        self.d.efld = np.loadtxt(self.efile,skiprows=skiprows)
        self.d.hfld = np.loadtxt(self.hfile,skiprows=skiprows)
        erows, ecols = np.shape(self.d.efld)
        hrows, hcols = np.shape(self.d.hfld)
        if (erows != hrows) or (ecols != hcols):
            raise TypeError('Input file size of E and H is inconsistent.')

        exl = np.unique(self.d.efld[:,cx])
        eyl = np.unique(self.d.efld[:,cy])
        ezl = np.unique(self.d.efld[:,cz])
        hxl = np.unique(self.d.hfld[:,cx])
        hyl = np.unique(self.d.hfld[:,cy])
        hzl = np.unique(self.d.hfld[:,cz])

        if any(exl != hxl) or any(eyl != hyl) or any(eyl != hyl):
            raise TypeError('Input data grid of E and H is inonsisitent.')

        self.d.xmin = np.amin(exl)
        self.d.xmax = np.amax(exl)
        self.d.ymin = np.amin(eyl)
        self.d.ymax = np.amax(eyl)
        self.d.zmin = np.amin(ezl)
        self.d.zmax = np.amax(ezl)

        self.d.dsize = erows
        self.d.xsize = len(exl)-1
        self.d.ysize = len(eyl)-1
        self.d.zsize = len(ezl)-1

        self.d.dx = (self.d.xmax - self.d.xmin)/float(self.d.xsize)
        self.d.dy = (self.d.ymax - self.d.ymin)/float(self.d.ysize)
        self.d.dz = (self.d.zmax - self.d.zmin)/float(self.d.zsize)

        self.zyxsort()

    def zyxsort(self):
        # x:0, y:1, z:2
        for col in [cz, cy, cx]:
            self.d.efld = self.d.efld[self.d.efld[:,col].argsort(kind='mergesort')]
            self.d.hfld = self.d.hfld[self.d.hfld[:,col].argsort(kind='mergesort')]

    def dumpdata(self,filename=None):
        if filename is None:
            filename = 'dumpdata_'+self.suffix+'.pkl'

        with open(filename, 'wb') as f:
            pkl.dump(self.d, f, protocol = 2)


    def loaddata(self,filename=None):
        if filename is None:
            filename = 'dumpdata_'+self.suffix+'.pkl'

        with open(filename, 'rb') as f:
            self.d = pkl.load(f)

    def dumpexpn(self,filename=None):
        if filename is None:
            filename = 'dumpexpn_'+self.suffix+'.pkl'

        with open(filename, 'wb') as f:
            pkl.dump(self.t, f, protocol = 2)


    def loadexpn(self,filename=None):
        if filename is None:
            filename = 'dumpexpn_'+self.suffix+'.pkl'

        with open(filename, 'rb') as f:
            self.t = pkl.load(f)


    def calc_ezaxis(self):
        pos = []
        Ez = []
        for fld in self.d.efld:
            if fld[cx] == 0.0 and fld[cy] == 0.0:
                pos.append(fld[cz])
                Ez.append(fld[zR] * self.scale)

        pos = np.asarray(pos)
        Ez = np.asarray(Ez)
        if self.zmirror:
            Eztot = np.concatenate([-Ez[:0:-1],Ez])
            pstrt = pos[0]
        else:
            Eztot = Ez
            pstrt = pos[0]

        self.t.zaxis = pos - pstrt
        self.t.ezaxis = Eztot
        self.t.dz = self.d.dz

    def calc_multipole(self):
        self.t.coefEr = []
        self.t.coefEp = []
        self.t.coefHr = []
        self.t.coefHp = []

        if self.rnorm != None:
            rmax = self.rnorm
        else :
            rmax = np.amax(self.d.efld[:,(cx,cy)])
        self.t.Rm = rmax

        zls = np.linspace(self.d.zmin, self.d.zmax, self.d.zsize+1)
        for i,z in enumerate(zls):
            print(z)
            exyt = self.d.efld[range(i, self.d.dsize, self.d.zsize+1),:]
            rfld, pfld = self._trans_poler(exyt, 1)
            self.t.coefEr.append(self._put_coefrp(z, rfld, 1))
            self.t.coefEp.append(self._put_coefrp(z, pfld, 2))

            hxyt = self.d.hfld[range(i, self.d.dsize, self.d.zsize+1),:]
            rfld, pfld = self._trans_poler(hxyt, 2)
            self.t.coefHr.append(self._put_coefrp(z, rfld, 2))
            self.t.coefHp.append(self._put_coefrp(z, pfld, 1))

        if self.zmirror:
            self.t.coefEr = self._coefmirror(self.t.coefEr, 1)
            self.t.coefEp = self._coefmirror(self.t.coefEp, 1)
            self.t.coefHr = self._coefmirror(self.t.coefHr, 2)
            self.t.coefHp = self._coefmirror(self.t.coefHp, 2)

    def _put_coefrp(self, z, rpfld, vlabel):
        emax, coefmat = self._get_coefmat(rpfld, vlabel)
        mlpin = MplPos(z, emax, coefmat)
        fld_re = self._rebuild_fld(coefmat, emax, vlabel)
        error = np.amax(np.absolute(rpfld-fld_re))/emax
        if error > 0.5:
            print('Warning: field error tends to be big.')
        elif error > 1.0:
            raise ValueError('coefmat calculation failed.')
        return mlpin


    def _trans_poler(self, xyt, vlabel):
        nr = self.nrad
        nph = self.nphi
        radls = np.linspace(self.t.Rm/float(nr), self.t.Rm, nr)
        dphi = 2.0*np.pi/float(nph)
        phils = np.linspace(dphi, 2.0*np.pi, nph)
        rfld = np.zeros((nr, nph))
        pfld = np.zeros((nr, nph))

        for i in xrange(nr):
            for j in xrange(nph):
                xtmp = radls[i]*np.cos(phils[j])
                ytmp = radls[i]*np.sin(phils[j])
                xyloc = self._poly_fld(xtmp, ytmp, xyt, vlabel)
                rfld[i,j] = xyloc[0]*np.cos(phils[j]) + xyloc[1]*np.sin(phils[j])
                pfld[i,j] = xyloc[1]*np.cos(phils[j]) - xyloc[0]*np.sin(phils[j])

        return rfld, pfld

    def _poly_fld(self, xtmp, ytmp, xyt, vlabel):
        ptb = np.zeros(4,dtype=int)
        ptb[0] = int(np.floor((xtmp-np.amin(xyt[:,cx])/self.d.dx))*(self.d.ysize+1) \
                   + np.floor((ytmp-np.amin(xyt[:,cy])/self.d.dy)))
        ptb[1] = int((np.floor((xtmp-np.amin(xyt[:,cx])/self.d.dx))+1)*(self.d.ysize+1) \
                   + np.floor((ytmp-np.amin(xyt[:,cy])/self.d.dy)))
        ptb[2] = int(np.floor((xtmp-np.amin(xyt[:,cx])/self.d.dx))*(self.d.ysize+1) \
                   + np.floor((ytmp-np.amin(xyt[:,cy])/self.d.dy))+1)
        ptb[3] = int((np.floor((xtmp-np.amin(xyt[:,cx])/self.d.dx))+1)*(self.d.ysize+1) \
                   + np.floor((ytmp-np.amin(xyt[:,cy])/self.d.dy))+1)

        xyloc = np.zeros(2)

        if vlabel == 1:
            xx = xR
            yy = yR
        elif vlabel == 2:
            xx = xI
            yy = yI

        if np.amax(ptb[1:4]) > len(xyt):
                xyloc[0] = xyt[ptb[0],xx]
                xyloc[1] = xyt[ptb[0],yy]
        else:
            x1 = xyt[ptb[0],cx]
            x2 = xyt[ptb[1],cx]
            y1 = xyt[ptb[0],cy]
            y2 = xyt[ptb[2],cy]
            xyu = (x2 - x1)*(y2 - y1)

            xyloc[0] = xyt[ptb[0],xx]*((x2-xtmp)*(y2-ytmp)/xyu) \
                     - xyt[ptb[1],xx]*((x1-xtmp)*(y2-ytmp)/xyu) \
                     - xyt[ptb[2],xx]*((x2-xtmp)*(y1-ytmp)/xyu) \
                     + xyt[ptb[3],xx]*((x1-xtmp)*(y1-ytmp)/xyu) \

            xyloc[1] = xyt[ptb[0],yy]*((x2-xtmp)*(y2-ytmp)/xyu) \
                     - xyt[ptb[1],yy]*((x1-xtmp)*(y2-ytmp)/xyu) \
                     - xyt[ptb[2],yy]*((x2-xtmp)*(y1-ytmp)/xyu) \
                     + xyt[ptb[3],yy]*((x1-xtmp)*(y1-ytmp)/xyu) \

        return xyloc

    def _get_coefmat(self, rpfldin, vlabel):
        fmax = np.amax(np.absolute(rpfldin))
        rpfld = rpfldin/fmax
        coefmat = []
        nr = self.nrad
        nph = self.nphi
        n = self.orad
        m = self.ophi

        X = np.linspace(1.0/float(nr), 1.0, nr)
        Y = np.zeros(nr)

        for i in xrange(nr):
            for j in xrange(nph):
                Y[i] += rpfld[i,j]/float(nph)

        P = np.polyfit(X, Y, n)
        Padj = np.zeros(n+1)
        for i in range(n+1):
            Padj[i] = P[n-i]

        coefmat.append(Padj.copy())

        for mi in xrange(m):
            Y = np.zeros(nr)
            if np.mod(mi+1,2) == np.mod(vlabel-1,2):
                for i in xrange(nr):
                    for j in xrange(nph):
                        Y[i] += np.cos(2.0*np.pi*(mi+1)*(j+1)/float(nph))*rpfld[i,j]*2.0/float(nph)
            elif np.mod(mi+1,2) == np.mod(vlabel,2):
                for i in xrange(nr):
                    for j in xrange(nph):
                        Y[i] += np.sin(2.0*np.pi*(mi+1)*(j+1)/float(nph))*rpfld[i,j]*2.0/float(nph)

            P = np.polyfit(X, Y, n)
            for i in xrange(n+1):
                Padj[i]=P[n-i]
            coefmat.append(Padj.copy())

        coefmat = np.transpose(np.array(coefmat))

        return fmax, coefmat

    def _rebuild_fld(self, coefmat, emax, vlabel):
        nr = self.nrad
        nph = self.nphi
        remap = np.zeros((nr, nph))
        n, m = coefmat.shape
        R = np.zeros(n)
        Phi = np.zeros(m)
        for i in xrange(nr):
            for j in xrange(nph):
                for ni in xrange(n):
                    R[ni] = np.power(float(i+1)/float(nr), ni)
                for mi in xrange(m):
                    if np.mod(mi,2) == np.mod(vlabel-1,2):
                        Phi[mi] = np.cos(2.0*np.pi*mi*(j+1)/float(nph))
                    elif np.mod(mi,2) == np.mod(vlabel,2):
                        Phi[mi] = np.sin(2.0*np.pi*mi*(j+1)/float(nph))
                remap[i,j] = emax*R.dot(coefmat.dot(Phi))
        return remap

    def _coefmirror(self, coefls, vlabel):
        coefm = []
        N = len(coefls)
        for i in xrange(N-1):
            elm = coefls[N-1-i].copy()
            elm.z *= -1.0
            if vlabel == 2:
                elm.mat *= -1.0
            coefm.append(elm)
        coeftot = coefm + coefls
        pstrt = coeftot[0].z
        for elm in coeftot:
            elm.z -= pstrt
        return coeftot

    def calc_linterm(self):
        self.t.Er12 = []
        self.t.Er21 = []
        self.t.Er23 = []
        self.t.Hp12 = []
        self.t.Hp21 = []
        self.t.Hp23 = []
        for mlp in self.t.coefEr:
            self.t.Er12.append(mlp.fmax*mlp.mat[0,1]*self.scale)
            self.t.Er21.append(mlp.fmax*mlp.mat[1,0]*self.scale)
            self.t.Er23.append(mlp.fmax*mlp.mat[1,2]*self.scale)
        for mlp in self.t.coefHp:
            self.t.Hp12.append(mlp.fmax*mlp.mat[0,1]*self.scale)
            self.t.Hp21.append(mlp.fmax*mlp.mat[1,0]*self.scale)
            self.t.Hp23.append(mlp.fmax*mlp.mat[1,2]*self.scale)

class ElemGen(object):
    """Class for generating lattice element from multipole terms
    """
    def __init__(self, expn, qm=None, es=None, frf=None, ek_low=None, ek_high=None,\
                 beta_low=None, beta_high=None, step=100, deg = 9):
        self.t = expn
        self.qm = qm
        self.es = es
        self.lmd = None
        self._frf = frf
        self._ek_low = ek_low
        self._ek_high = ek_high
        self._beta_low = beta_low
        self._beta_high = beta_high
        self.step = step
        self.deg = deg

        if ek_low is not None:
            self.ek_low = ek_low
        if ek_high is not None:
            self.ek_high = ek_high
        if beta_low is not None:
            self.beta_low = beta_low
        if beta_high is not None:
            self.beta_high = beta_high
        if frf is not None:
            self.frf = frf

        self.support = {'EDipole':self.t.Er12,
                        'EFocus':self.t.Er21,
                        'EQuad':self.t.Er23,
                        'HDipole':self.t.Hp12,
                        'HMono':self.t.Hp21,
                        'HQuad':self.t.Hp23,
                        'AccGap':self.t.ezaxis}

        self.ttflmd = clight/(80.5e6)

    @property
    def frf(self):
        return self._frf

    @frf.setter
    def frf(self, v):
        self._frf = v
        if v is not None:
            self.lmd = clight/(v)

    @property
    def ek_low(self):
        return self._ek_low

    @ek_low.setter
    def ek_low(self, ek):
        self._ek_low = ek
        if ek is not None:
            self._beta_low = self.ek2beta(ek)

    @property
    def ek_high(self):
        return self._ek_high

    @ek_high.setter
    def ek_high(self, ek):
        self._ek_high = ek
        if ek is not None:
            self._beta_high = self.ek2beta(ek)

    @property
    def beta_low(self):
        return self._beta_low

    @beta_low.setter
    def beta_low(self, beta):
        self._beta_low = beta
        if beta is not None:
            self._ek_low = self.beta2ek(beta)

    @property
    def beta_high(self):
        return self._beta_high

    @beta_high.setter
    def beta_high(self, beta):
        self._beta_high = beta
        if beta is not None:
            self._ek_high = self.beta2ek(beta)

    def ek2beta(self, ek):
        if self.es is None:
            raise ValueError('es is not defined.')
        w = self.es + ek
        gm = w/self.es
        return np.sqrt(1.0-1.0/gm/gm)

    def beta2ek(self, beta):
        if self.es is None:
            raise ValueError('es is not defined.')
        gm = 1.0/np.sqrt(1.0-beta*beta)
        return self.es*(gm-1.0)

    def get_v0pos(self, zax, fzax):
        k = 2.0*np.pi/(self.beta_high*self.ttflmd*1e3)
        out = self._calc_ttf(zax, fzax, k)
        return out[-1], out[0]

    def fit_ttf(self, zax, fzax):
        kh = 2.0*np.pi/(self.beta_high*self.ttflmd*1e3)
        kl = 2.0*np.pi/(self.beta_low*self.ttflmd*1e3)
        table = []
        for ik in np.linspace(kh, kl, self.step+1):
            out = self._calc_ttf(zax, fzax, ik)
            table += [[ik]+out]

        table = np.asarray(table)

        pt = np.polyfit(table[:,0], table[:,2], 9)
        ps = np.polyfit(table[:,0], table[:,4], 9)

        return pt, ps

    def _calc_ttf(self, zax, fzax, k):
        if len(zax) != len(fzax):
            raise ValueError('Worng input array length.')
        axdata = np.transpose([zax,fzax])
        axdata[0,:] -= axdata[0,0]
        dz = self.t.dz

        ecu = 0.0
        ecd = 0.0
        for i in xrange(len(axdata)-1):
            ecu += 0.5*(np.absolute(axdata[i,1])+np.absolute(axdata[i+1,1]))\
                  *0.5*(axdata[i,0]+axdata[i+1,0])*dz
            ecd += 0.5*(np.absolute(axdata[i,1])+np.absolute(axdata[i+1,1]))*dz
        ecen = ecu/ecd
        axdata[:,0] -= ecen

        ttf = 0.0
        ttfp = 0.0
        stf = 0.0
        stfp = 0.0

        for i in xrange(len(axdata)-1):
            dfac = 0.5*(axdata[i,0]+axdata[i+1,0])
            efac = 0.5*(axdata[i,1]+axdata[i+1,1])
            ttf += efac*np.cos(k*dfac)*dz
            ttfp -= dfac*efac*np.sin(k*dfac)*dz
            stf += efac*np.sin(k*dfac)*dz
            stfp += dfac*efac*np.cos(k*dfac)*dz

        ttf /= ecd
        ttfp /= ecd
        stf /= ecd
        stfp /= ecd
        v0 = ecd/1e9
        return [ecen, ttf, ttfp, stf, stfp, v0]

    def get_psync(self):
        wptab = []
        axdata = self.t.ezaxis[0:-1] + self.t.ezaxis[1:]
        for ek in np.linspace(self.ek_low, self.ek_high, self.step):
            ektab = []
            for ip in np.linspace(0.0, 2.0*np.pi, num=50):
                ek2, phi = self._calc_gap(ek, ip, axdata)
                ektab.append([ip, ek2])
            phi_c = self._calc_cosfit(ektab)
            wptab.append([ek, phi_c])
        wptab = self._check_jump(wptab)
        return self._calc_expfit(wptab)

    def _calc_gap(self, ek, phi0, axdata):
        dz = self.t.dz
        phi = phi0
        ek2 = ek
        beta = self.ek2beta(ek)
        k = 2.0*np.pi/(beta*self.lmd*1e3)
        for i in xrange(len(axdata)):
            philast = phi
            phi += k*dz
            efac = 0.5*(axdata[i])/1e6
            ek2 += self.qm*efac*np.cos(0.5*(philast+phi))*dz/1e3
            if (ek2) < 0 :
                ek2 = 0.0
                beta = 0.0
            else:
                beta = self.ek2beta(ek2)
            k = 2.0*np.pi/(beta*self.lmd*1e3)
        return ek2, phi

    @staticmethod
    def _calc_cosfit(ektab):
        xdata = np.array(ektab)[:,0]
        ydata = np.array(ektab)[:,1]
        ydata -= np.mean(ydata)
        xs1 = np.amax(ydata)
        xs2 = np.arccos(ydata[0]/xs1)
        optfunc = lambda x: x[0]*np.cos(xdata+x[1]) - ydata
        x1, x2 = leastsq(optfunc, [xs1, xs2])[0]
        if x1 > 0:
            phi_c = x2
        elif x1 <= 0:
            phi_c = x2 + np.pi
        return phi_c

    @staticmethod
    def _calc_expfit(wptab):
        xdata = np.array(wptab)[:,0]
        ydata = np.array(wptab)[:,1]
        optfunc = lambda x: x[0]*np.power(xdata,x[1]) - x[2] - ydata
        xs1 = 5.0
        xs2 =-0.5
        xs3 = 0.0
        x1, x2, x3 = leastsq(optfunc, [xs1, xs2, xs3])[0]
        return [x1, x2, x3]

    @staticmethod
    def _check_jump(wptab):
        wptab = np.array(wptab)
        for i in xrange(len(wptab)-1):
            thr = wptab[i+1,1] - wptab[i,1]
            if thr > np.pi:
                wptab[i+1:,1] -= 2.0*np.pi
            elif thr < -np.pi:
                wptab[i+1:,1] += 2.0*np.pi
        return wptab

    def parse_elem(self, etype, srange):
        zvec = self.t.zaxis
        istrt = np.argmin(np.absolute(zvec-srange[0]))
        istop = np.argmin(np.absolute(zvec-srange[1]))
        if etype in self.support.keys():
            zax = self.t.zaxis[istrt:istop+1]
            fzax = self.support[etype][istrt:istop+1]
            pt, ps = self.fit_ttf(zax, fzax)
            v0, pos = self.get_v0pos(zax, fzax)
            attr = list(pt)+list(ps)
            if etype == 'AccGap':
                attr += list(self.get_psync())

        return [pos, v0, attr]

def lat_gen(fname, expn, line, ionZ, ionEs, frf, **kws):
    """Generate cavity file for FLAME
    """
    ek_low = kws.get('ek_low', None)
    ek_high = kws.get('ek_high', None)
    beta_low = kws.get('beta_low', None)
    beta_high = kws.get('beta_high', None)
    step = kws.get('step', 100)

    if (ek_low is None) and (beta_low is None):
        raise ValueError('ek_low or beta_low must be set.')

    if (ek_high is None) and (beta_high is None):
        raise ValueError('ek_high or beta_high must be set.')

    eg = ElemGen(expn, qm=ionZ, es=ionEs, frf=frf, ek_low=ek_low, ek_high=ek_high,\
                 beta_low=beta_low, beta_high=beta_high, step=step)

    elems = []
    srange = []
    for l in line:
        print(l['name'])
        data = eg.parse_elem(l['type'],l['range'])
        elems.append([l]+list(data))
        srange += list(l['range'])

    elems = sorted(elems, key=lambda x: x[1])
    strt = min(srange)
    end = max(srange)

    with open(fname, 'wb') as fp:
        fp.write('Rm = ' + str(expn.Rm) + ';\n')
        drfcnt = 0
        cell = []
        for e in elems:
            drf = 'drift_'+str(drfcnt)
            fp.write(drf + ': drift, L = ' + str(e[1]-strt) +';\n')
            cell.append(drf)
            drfcnt += 1
            elm = e[0]['name']
            fp.write(elm + ': ' + e[0]['type'] + ', L = 0.0, V0 = ' + str(e[2]) \
                         + ', attr = ' + str(e[3]) + ';\n')
            cell.append(elm)
            strt = e[1]

        drf = 'drift_'+str(drfcnt)
        fp.write(drf + ': drift, L = ' + str(end - e[1]) +';\n')
        cell.append(drf)
        fp.write('cell: LINE = (\n')
        for nm in cell[:-1]:
            fp.write(nm + ', ')
        fp.write(cell[-1] + ');\n')
        fp.write('USE: cell;\n')
        fp.write('Ez = [\n')
        for z,ez in zip(expn.zaxis[:-1], expn.ezaxis[:-1]):
            fp.write(str(z) + ', ' + str(ez) + ',\n')
        fp.write(str(expn.zaxis[-1]) + ', ' + str(expn.ezaxis[-1]) + '];\n')


def get_latline(expn, types='all'):
    if types == 'all':
        tls = ['EDipole', 'EFocus', 'EQuad', 'HDipole', 'HMono', 'HQuad', 'AccGap']
    else:
        tls = list(types)

    lat_line = []

    stt = expn.zaxis[0]
    mid = expn.zaxis[-1]/2
    end = expn.zaxis[-1]

    for etype in tls:
        if etype == 'EDipole':
            ln = np.array(expn.Er12)
        elif etype == 'EFocus':
            ln = np.array(expn.Er21)
        elif etype == 'EQuad':
            ln = np.array(expn.Er23)
        elif etype == 'HDipole':
            ln = np.array(expn.Hp12)
        elif etype == 'HMono':
            ln = np.array(expn.Hp21)
        elif etype == 'HQuad':
            ln = np.array(expn.Hp23)
        elif etype == 'AccGap':
            ln = np.array(expn.ezaxis)
        else:
            raise TypeError('Worng input types :'+ etype)

        ln = np.around(ln/np.amax(np.absolute(ln))*1e7,1) + 1e-256
        iln = np.sign(ln)
        bln = np.array(np.absolute((iln[:-1] - iln[1:])/2), dtype=bool)
        ids = np.arange(len(ln))[bln]
        p0 = expn.zaxis[ids] + np.absolute(ln[ids]/(ln[ids]-ln[ids+1]))*expn.dz

        if not stt in p0:
            p0 = np.concatenate([p0, [stt]])

        if not mid in p0:
            p0 = np.concatenate([p0, [mid]])

        if not end in p0:
            p0 = np.concatenate([p0, [end]])

        p0.sort()

        bpls = np.array((p0[1:] - p0[:-1]) > expn.dz*10, dtype=bool)
        ids2 = np.arange(len(p0))[bpls]
        if not len(ids2) == 0:
            ids2 = np.concatenate([ids2, [ids2[-1]+1]])
            p0 = p0[ids2]

        print(etype, ids, p0)

        nn = 1
        for i in range(len(p0)-1):
            lat_line.append({'name':etype[0:2]+str(nn).zfill(5),
                             'type':etype,
                             'range':[p0[i],p0[i+1]]})
            nn += 1

    return lat_line
