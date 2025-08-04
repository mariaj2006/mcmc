import numpy as np 
from astropy.io import fits
# from cubelines.fitutils import get_lsf, convolve_lsf
from model import O3_1comp
from astropy.constants import c

clight = c.value/1e3 # in unit of angstrom/s

from os import path

# open the cube and return essential data
def readcube(cubefile):
    hdul = fits.open(cubefile)
    if len(hdul) < 3: warnings.warn('datacube missing variance extension')
    data = hdul[1].data
    err = np.sqrt(hdul[2].data)
    hdr = hdul[1].header
    wave = hdr['CRVAL3'] + hdr['CD3_3']*np.arange(hdr['NAXIS3'])
    hdul.close()
    return wave, data, err

path_prefix = '/carnegie/nobackup/scratch/msanchezrincon/'
# ratio I found fir my cube
noise_scale_ratio = 1.32

# replace with my data cube
cubefile = '/Users/mariasanchezrincon/mcmc/data/vacuum_data_cube.fits'
wave, data, err = readcube(cubefile)
err *= noise_scale_ratio
# replace with my OIII SNR map
flagmap = fits.getdata('/Users/mariasanchezrincon/mcmc/data/SNR_OIII.fits')
yy, xx = np.where(flagmap>3)
print(len(yy))

# skymask = np.ones(len(wave))
# skymask[(wave>9092)&(wave<9097)]=0
# skymask[(wave>9102)&(wave<9106)]=0
# skymask[(wave>9000)&(wave<9005)]=0
# skymask[(wave>8986)&(wave<8993)]=0

# what should I replace these wavelength values for?
wavemin1, wavemax1 = 7380,7422
mask1 = (wave>wavemin1) & (wave<wavemax1)
wavemin2, wavemax2 = 7445,7497
mask2 = (wave>wavemin2) & (wave<wavemax2)
wavefit = np.concatenate((wave[mask1], wave[mask2]))
print(len(wavefit))

# replace z0
line = [4960.295, 5008.240]
z0 = 0.0477
line_z = np.asarray(line)*(1+z0)

# lsf = get_muse_lsf(line_z) # what is the purpose of this? rather:
l0, r0 = np.loadtxt('mcmc/data/muse_lsf.dat',unpack=True)
r = interp1d(l0, r0)(line_z)
lsf = 2.998e5/r

func = O3_1comp(line, lsf).model
ndim = 3
n = len(wavefit)

nx, ny = flagmap.shape[1], flagmap.shape[0]

# v0_map = fits.getdata(path_prefix+'vmap_OIIIonly_1comp.fits') #np.zeros(shape=(ny, nx))

# best_fit = fits.getdata(path_prefix+'OIIIonly_bestfit_1comp.fits') #np.zeros(shape=(ndim, ny, nx))
# bic_map = fits.getdata(path_prefix+'bic_1comp_v0.fits')#np.zeros(shape=flagmap.shape)

v0_map = np.zeros(shape=(ny, nx))

best_fit = np.zeros(shape=(ndim, ny, nx))
bic_map = np.zeros(shape=flagmap.shape)

v0_map[:,:] = np.nan
best_fit[:,:,:] = np.nan
bic_map[:,:] = np.nan

def auto_corr_func(timeseries, lagmax):
    """
    compute auto correlation function
    """
    ts = np.asarray(timeseries)
    n = np.size(ts) - 1
    ts -= np.mean(ts) # set to mean 0
    corr_func = np.zeros(lagmax)
    # compute xi(j) for different j
    for j in range(lagmax):
        # sum of ts[t+dt]*ts[t]
        corr_func[j] = (np.dot(timeseries[0:n-j],timeseries[j:n])) 
    if (corr_func[0] > 0):
        corr_func /= corr_func[0] # normalize
    return corr_func

def compute_tcorr(timeseries, maxcorr):
    """
    compute correlation time
    """
    timeseries = np.copy(timeseries)
    mean = np.mean(timeseries)
    corrfxn = auto_corr_func(timeseries,maxcorr)
    tau = np.sum(corrfxn)-1 # auto-correlation time
    var = np.var(timeseries)
    sigma = np.sqrt(var * tau / len(timeseries))
    return tau, mean, sigma

# def acceptance_rate(chain):
    # return len(np.unique(chain))/len(chain)
    # return len(np.unique(chain))/len(chain)

def accept_rate(chains):
    nwalkers = chains.shape[0]
    ar_all = np.zeros(nwalkers)
    for i in range(nwalkers):
        ar_all[i] = len(np.unique(chains[i,-250:,0].flatten()))/len(chains[i,-250:,0].flatten())
    return ar_all

def lnlm(fluxfit, errfit, model):
    s_tot_sq = errfit**2
    chi2 = (model - fluxfit)**2/(2*s_tot_sq)
    lnlm = - np.sum(chi2)
    return lnlm

def bic(k, n, lnlm):
    # k: number of parameters
    # n: number of data points
    # L: max. likelihood
    return k*np.log(n) - 2.*lnlm

def model_vec(func, x, pars_vec):
    nmodels = pars_vec.shape[0]
    models = np.zeros(shape = (nmodels, len(x)))
    for i in range(nmodels):
        models[i, :] = func(x, *pars_vec[i,:])
    return models

def lnlm_vec(fluxfit, errfit, model_vec):
    nmodels = model_vec.shape[0]
    fluxfit_vec = np.repeat(fluxfit[None,:], nmodels, axis=0)
    errfit_vec = np.repeat(errfit[None,:], nmodels, axis=0)
    s_tot_sq = errfit_vec**2
    chi2 = (model_vec - fluxfit_vec)**2/(2*s_tot_sq)
    lnlm = - np.sum(chi2, axis=1)
    return lnlm

chain_quality = np.zeros(shape=(flagmap.shape[0], flagmap.shape[1]))

for x, y in zip(xx, yy):
    print(x, y)
    checkpath = path_prefix + '/chains_cleaned/mc_%d_%d.fits' % (x,y)
    if path.exists(checkpath): 
        continue
    dataspec = data[:,y,x]
    errspec = err[:,y,x]

    fluxfit = np.concatenate((dataspec[mask1], dataspec[mask2]))
    errfit = np.concatenate((errspec[mask1], errspec[mask2]))

    chains = fits.getdata(path_prefix + 'chains/mc_%d_%d.fits' % (x,y))
    
    nwalkers0 = chains.shape[0]

    ar_all = accept_rate(chains)
    good_chain = ar_all > (np.median(ar_all) - 2*np.std(ar_all))
    chains = chains[good_chain,:,:]

    nwalkers1 = chains.shape[0]
    print('nwalkers after accpt.rate selection:', nwalkers1)

    ### likelihood clustering analysis
    lnlm_map = np.zeros((nwalkers1, 250))
    minus_lnlm = np.zeros(nwalkers1)
    for i in range(nwalkers1):
        pars250 = chains[i,-250:,:] # 250 sets of parameters
        models250 = model_vec(func, wavefit, pars250)
        lnlm_map[i,:] = lnlm_vec(fluxfit, errfit, models250)
        minus_lnlm[i] = -np.mean(lnlm_vec(fluxfit, errfit, models250))
    walkerbest, samplebest = np.where(lnlm_map == np.max(lnlm_map))
    popt = chains[walkerbest[0], -250 + samplebest[0], :]
    best_fit[:,y,x] = popt
    v0_map[y,x] = (popt[0] - z0)/(1+z0)*clight
    # v1_map[y,x] = (popt[0] - z0)/(1+z0)*clight
    # v2_map[y,x] = (popt[3] - z0)/(1+z0)*clight
    # v3_map[y,x] = (popt[6] - z0)/(1+z0)*clight

    l = lnlm(fluxfit, errfit, func(wavefit, *popt))
    bic_val = bic(ndim, n, l)
    bic_map[y, x] = bic_val

    s_index = np.argsort(minus_lnlm) # sorting index
    minus_lnlm_s = minus_lnlm[s_index]
    dlnlm = (minus_lnlm_s[1:] - minus_lnlm_s[0])/np.arange(1,nwalkers1)

    const_thresh = 3
    try: 
        bad_i = np.where(dlnlm[10:]/dlnlm[9:-1]>const_thresh)[0][0]+11
        good_chains = s_index[:bad_i]
        chains_good = chains[good_chains,:,:]
    except: chains_good = chains

    nwalkers2 = chains_good.shape[0]
    print('nwalkers after likelihood selection:', nwalkers2)

    chain_quality[y,x] = nwalkers0 - nwalkers2
    fits.writeto(checkpath, chains_good, overwrite=True)

fits.writeto(path_prefix+'OIIIonly_bestfit_1comp.fits', best_fit, overwrite=True)
fits.writeto(path_prefix+'bic_1comp_v0.fits', bic_map, overwrite=True)
fits.writeto(path_prefix+'vmap_OIIIonly_1comp.fits', v0_map, overwrite=True)
# fits.writeto(path_prefix+'vmap_OIIIonly_1comp_1.fits', v1_map, overwrite=True)
# fits.writeto(path_prefix+'vmap_OIIIonly_1comp_2.fits', v2_map, overwrite=True)
# fits.writeto(path_prefix+'vmap_OIIIonly_2comp_3.fits', v3_map, overwrite=True)
# fits.writeto(path_prefix+'vmap_OIIIonly_2comp_4.fits', v4_map, overwrite=True)
fits.writeto(path_prefix+'chain_quality.fits', chain_quality, overwrite=True)
