import os
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from model import O3_1comp
import emcee
import time
import sys
from os import path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

##test
#test
'''
sig/c = width of gaussian/lambda observed
3:1 ratio for amplitudes
resolution = lambda obs/ FWHM = c/FWHM vel
'''

#/Users/mariasanchezrincon/Downloads/muse_lsf.dat
# what does this do?
array_num = int(sys.argv[1])

def readcube(cubefile):
    hdul = fits.open(cubefile)
    if len(hdul) < 3: warnings.warn('datacube missing variance extension')
    data = hdul[1].data
    err = np.sqrt(hdul[2].data)
    hdr = hdul[1].header
    wave = hdr['CRVAL3'] + hdr['CD3_3']*np.arange(hdr['NAXIS3'])
    hdul.close()
    return wave, data, err

#cubefile = '../qsub_HRSDI_eso_w_cubexqso_cont_sub_v2_center_replaced_smoothed_sig1.5.fits'
cubefile = 'mcmc_fitting/data/vac_cont_sub_mini.fits'
#flagmap = fits.getdata('../flagmap_3sig_OIII.fits')
flagmap = fits.getdata('mcmc_fitting/data/SNR_OIII.fits')
noise_scale_ratio = 1.32 # my number is approx. 1.3240177979590504, not 1.6

#popt_prev = fits.getdata('/Users/mariasanchezrincon/CASSI_SURF/UGC7342_IFS_data/OIII_best_fit.fits')

wave, data, err = readcube(cubefile)
err *= noise_scale_ratio # why do we need to multiply the error by 1.6?
 # this was only a mask to test a few spaxels
loc_mask = np.zeros((322,324)) # do 2x2
loc_mask[106:108,130:132] = 1
tot_mask = np.full((322,324),np.nan)

for x in range(324):
    for y in range(322):
        if (flagmap[y,x]>3) and (loc_mask[y,x]>0):
            tot_mask[y,x] = True
        else:
            tot_mask[y,x] = False

yy, xx = np.where(tot_mask) # change after
print(len(yy))

# remember to check how the mask is defined inside the fitting loop
wavemin1, wavemax1 = 5180, 5210
mask1 = (wave>wavemin1) & (wave<wavemax1)
wavemin2, wavemax2 = 5220, 5267
mask2 = (wave>wavemin2) & (wave<wavemax2)

line = [4960.295, 5008.240]
z0 = 0.0477
line_z = np.asarray(line)*(1+z0)

# lsf = get_muse_lsf(line_z) # what is the purpose of this? rather:
l0, r0 = np.loadtxt('mcmc_fitting/data/muse_lsf.dat',unpack=True)
r = interp1d(l0, r0)(line_z)
lsf = 2.998e5/r
func = O3_1comp(line, lsf).model # does the .model make this into a function of some sort?


##mod = O3_1comp(line,lsf).model_display(x,z,sig,w)


def lnpi_global(pars):
    model = func(wavefit, *pars)
    s_tot_sq = errfit**2
    chi2 = (model - fluxfit)**2/(2*s_tot_sq)
    lnlm = - np.sum(chi2)

    # check bounds
    f = check_bound_global(pars)
    lnlm += f    
    return lnlm

def check_bound_global(pars):
    f = 0
    for i in range(ndim):
        if pars[i] < bound[i,0] or pars[i] > bound[i, 1]:
            f = -np.inf
            
    # check component redshifts in order
    if len(pars) > 3:
        for i in range(3, len(pars), 3):
            if pars[i] < pars[i-3]: f = -np.inf
        
    return f

s1, s2 = (array_num-1)*50, array_num*50
if s2>len(yy): s2 = len(yy)
#print(xx[s1:s2])
#print(yy[s1:s2])
bad_pixels = []
for x, y in zip(xx[s1:s2], yy[s1:s2]):
    print('x, y', x, y)
    print(flagmap[y,x])
    checkpath = 'mcmc_fitting/results/mc_%d_%d.fits' % (x, y) 
    if path.exists(checkpath): continue
    t0 = time.time()
    dataspec = data[:,y,x]
    errspec = err[:,y,x]

    # wavefit = wave[mask2]
    # fluxfit = dataspec[mask2]
    # errfit = errspec[mask2]
    wavefit = np.concatenate((wave[mask1], wave[mask2]))
    fluxfit = np.concatenate((dataspec[mask1], dataspec[mask2]))
    errfit = np.concatenate((errspec[mask1], errspec[mask2]))

    ndim, nwalkers, nsteps = 3, 200, 4000

    bound = np.array([[0.0400, 0.055383],
                    [1., 500.],
                    [0.2, 4000.]]) 

    p0 = [0.0477, 75, 620]
    # p0 = [0.8103, 41.66, 2.61, 0.00217, 154.34, 12.96]
    plim = ((bound[0,0], bound[1,0], bound[2,0]),
            (bound[0,1], bound[1,1], bound[2,1]))

    #try: popt, pcov = curve_fit(func, wavefit, fluxfit, p0=p0, sigma=errfit, bounds = plim)
    #except: popt = p0

    popt = np.array(p0)


#km/s / c * (1+z) = sig in vel
# initialization
    sigs = np.array([0.0007, 40, 10])
    p = np.zeros(shape=(nwalkers, ndim))
    for i in range(ndim):
        rand_p = np.random.normal(popt[i], sigs[i], size=nwalkers)
        rand_p[rand_p<bound[i, 0]] = bound[i, 0]
        rand_p[rand_p>bound[i, 1]] = bound[i, 1]
        p[:,i] = rand_p

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpi_global)
    #run for a given nsteps
    bad_pixels = []
    try: 
        sampler.run_mcmc(p, nsteps)
    except:
        bad_pixels.append([x,y])
        continue
    
    params = ['z','sig','amp']
    flat_samples = sampler.get_chain(discard=500, flat=True)
    
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        center = mcmc[1]
        left_sd = mcmc[1] - mcmc[0]
        right_sd = mcmc[2] - mcmc[1]
        text_to_save = f'{params[i]} is -{left_sd:.5f} {center:.5f} +{right_sd:.5f}'
        file_name = f'mcmc_fitting/results/mc_{x}_{y}.txt'
        with open(file_name, 'w') as file:
            file.write(text_to_save)
        
    print(time.time() - t0,'s')
    
    mcmc_z = np.percentile(flat_samples[:, 0], [16, 50, 84])
    mcmc_sig = np.percentile(flat_samples[:, 1], [16, 50, 84])
    mcmc_amp = np.percentile(flat_samples[:, 2], [16, 50, 84])
    z = mcmc_z[1]
    sig = mcmc_sig[1]
    amp = mcmc_amp[1]
    print(z,sig,amp)
    
    mod = O3_1comp(line,lsf).model_display(wave,z,sig,amp)
    fig,ax = plt.subplots()
    ax.plot(wave,mod)
    ax.step(wave,dataspec,where='mid')
    plt.show()
    
    fits.writeto(checkpath, sampler.chain.astype(np.float32),overwrite=True)
np.save('bad_pixel.npy', bad_pixels)
# plot the spectra of the pixels above the best fit params to see the accuracy visually
# use model_display
