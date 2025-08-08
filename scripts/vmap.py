import numpy as np
from astropy.io import fits
#from cubelines.modelutils import get_muse_lsf, convolve_lsf
#from mylib.turbulence import VSF
from scipy.interpolate import interp1d
import time
from astropy.constants import c 

def convolve_lsf(sig, lsf):
    a = 2*np.sqrt(2*np.log(2))
    return np.sqrt(sig**2+(lsf/a)**2)

clight = c.value/1e3 # km/s

def get_ave_vel(popt):
    # get flux-weighted mean velocity
    npar = popt.shape[-1]
    ncomp = int(npar/3)
    if ncomp < 1:
        return (popt[:,:,0] - z0)/(1 + z0)*clight
    n_index = np.arange(2, npar, 3)
    s_index = np.arange(1, npar, 3)
    z_index = np.arange(0, npar, 3)
    vs = (popt[:,:,z_index] - z0)/(1 + z0)*clight
    s_convol = convolve_lsf(popt[:,:,s_index], lsf)
    fs = popt[:,:,n_index]*s_convol
    vels = np.sum(vs*fs, axis=2)/np.sum(fs, axis=2)
    return vels.flatten()

def get_ave_vel_nowide(popt):
    # get flux-weighted mean velocity after excluding wide components 
    # with a velocity dispersion >300km/s
    npar = popt.shape[-1]
    ncomp = int(npar/3)
    if ncomp < 2:
        return (popt[:,:,0] - z0)/(1 + z0)*clight
    n_index = np.arange(2, npar, 3)
    s_index = np.arange(1, npar, 3)
    z_index = np.arange(0, npar, 3)
    vs = (popt[:,:,z_index] - z0)/(1 + z0)*clight
    s_convol = convolve_lsf(popt[:,:,s_index], lsf)
    s_convol[(popt[:,:,s_index] > 300)] = 0
    fs = popt[:,:,n_index]*s_convol
    vels = np.sum(vs*fs, axis=2)/np.sum(fs, axis=2)
    return vels.flatten()


def get_strongest_vel(popt):
    # get the velocity of the strongest (most flux) component
    npar = popt.shape[-1]
    ncomp = int(npar/3)
    if ncomp < 1:
        return (popt[:, :, 0] - z0)/(1 + z0)*clight
    n_index = np.arange(2, npar, 3)
    s_index = np.arange(1, npar, 3)
    z_index = np.arange(0, npar, 3)
    vs = (popt[:, :, z_index] - z0)/(1 + z0)*clight
    s_convol = convolve_lsf(popt[:, :, s_index], lsf)
    fs = popt[:, :, n_index]*s_convol
    output_vs = []
    for i in range(popt.shape[0]):
        for j in range(popt.shape[1]):
            allfs = fs[i, j, :]
            max_comp = np.where(allfs == np.max(allfs))
            output_vs.append(vs[i, j, :][max_comp][0])
    # print(output_vs)
    return output_vs

# some basic set up 
line = 5008.240
z0 = 0.0477148
line_z = np.asarray(line)*(1+z0)

#lsf = get_muse_lsf(line_z)
l0, r0 = np.loadtxt('mcmc/data/muse_lsf.dat',unpack=True)
r = interp1d(l0, r0)(line_z)
lsf = 2.998e5/r

dBIC_cut = -30 # threshold in BIC value
           # below this value, use 2 comp


nsample=1000
flagmap = fits.getdata('mcmc/data/SNR_OIII.fits')
path_prefix = '/carnegie/nobackup/scratch/msanchezrincon/'
velmap_stack = np.zeros(shape=(nsample, flagmap.shape[0], flagmap.shape[1]))
velmap_stack[:,:,:] = np.nan

veldispmap_stack = np.zeros(shape=(nsample, flagmap.shape[0], flagmap.shape[1]))
veldispmap_stack[:,:,:] = np.nan

yy_val, xx_val = np.where(flagmap > 0)
print(len(xx_val))


# Monte Carlo sample of the velocity fields, using only one component fitting results
t0 = time.time()
for xi, yi in zip(xx_val, yy_val):
    chain_file = path_prefix + 'chains_cleaned/vel_chains/cleaned_mc_%d_%d.fits' % (xi,yi)
    try:
        chains = fits.getdata(chain_file)
    except FileNotFoundError:
        print(f"{xi} {yi} not found")
        continue
    redshifts = chains[:, -250:, 0].flatten() # take the last -250 samples from all chains
    vels = (redshifts - z0)/(1 + z0) * clight
    velmap_stack[:, yi, xi] = np.random.choice(vels, size=nsample, replace=True)

    veldisps = chains[:, -250:, 1].flatten()
    veldispmap_stack[:, yi, xi] = np.random.choice(veldisps, size=nsample, replace=True)

velmap_med = np.median(velmap_stack, axis=0)
velmap_mean = np.mean(velmap_stack, axis=0)
velmap_std = np.std(velmap_stack, axis=0)

veldispmap_med = np.median(veldispmap_stack, axis=0)
veldispmap_mean = np.mean(veldispmap_stack, axis=0)
veldispmap_std = np.std(veldispmap_stack, axis=0)

print('Populating velocity map takes %.1f mins' % ((time.time()-t0)/60.))
fits.writeto(path_prefix + 'velocity/velmap_stack_OIII_1comp.fits'.format(dBIC_cut), velmap_stack, overwrite=True)
fits.writeto(path_prefix + 'velocity/velmap_stack_OIII_1comp_median.fits'.format(dBIC_cut), velmap_med, overwrite=True)
fits.writeto(path_prefix + 'velocity/velmap_stack_OIII_1comp_mean.fits'.format(dBIC_cut), velmap_mean, overwrite=True)
fits.writeto(path_prefix + 'velocity/velmap_stack_OIII_1comp_std.fits'.format(dBIC_cut), velmap_std, overwrite=True)

print(velmap_stack.shape)

# not needed for now
'''
# exit()


# for fitting with multiple components
nsample=1000
flagmap = fits.getdata('../OIII_mask3d_global_proj2d.fits')
path_prefix = '../'
BICmap = fits.getdata('../OIIIonly_num_of_comp_dBIC<{}.fits'.format(dBIC_cut))
velmap_stack = np.zeros(shape=(nsample, flagmap.shape[0], flagmap.shape[1]))
velmap_stack[:,:,:] = np.nan

yy_val, xx_val = np.where(flagmap > 0)
print(len(xx_val))


# Monte Carlo sample of the velocity fields
t0 = time.time()
for xi, yi in zip(xx_val, yy_val):
    if BICmap[yi, xi] == 1: 
        chain_file = '../OIIIonly_1comp/chains_cleaned/%d_%d_OIIIonly_1comp_chain.fits' % (xi,yi)
        chains = fits.getdata(chain_file)
        redshifts = chains[:, -250:, 0].flatten() # take the last -250 samples from all chains
        vels = (redshifts - z0)/(1 + z0) * clight
        velmap_stack[:, yi, xi] = np.random.choice(vels, size=nsample, replace=True)
    elif BICmap[yi, xi] == 2:
        chain_file = '../OIIIonly_2comp/chains_cleaned/%d_%d_OIIIonly_2comp_chain.fits' % (xi,yi)
        chains = fits.getdata(chain_file)
        chains = chains[:, -250:, :] # take the last -250 samples from all chains
        vels = get_ave_vel(chains)
        velmap_stack[:, yi, xi] = np.random.choice(vels, size=nsample, replace=True)
    elif BICmap[yi, xi] == 3:
        chain_file = '../OIIIonly_3comp/chains_cleaned/%d_%d_OIIIonly_3comp_chain.fits' % (xi,yi)
        chains = fits.getdata(chain_file)
        chains = chains[:, -250:, :] # take the last -250 samples from all chains
        vels = get_ave_vel(chains)
        velmap_stack[:, yi, xi] = np.random.choice(vels, size=nsample, replace=True)
    elif BICmap[yi, xi] == 4:
        chain_file = '../OIIIonly_4comp/chains_cleaned/%d_%d_OIIIonly_4comp_chain.fits' % (xi,yi)
        chains = fits.getdata(chain_file)
        chains = chains[:, -250:, :] # take the last -250 samples from all chains
        vels = get_ave_vel(chains)
        velmap_stack[:, yi, xi] = np.random.choice(vels, size=nsample, replace=True)

velmap_med = np.median(velmap_stack, axis=0)
velmap_mean = np.mean(velmap_stack, axis=0)
velmap_std = np.std(velmap_stack, axis=0)

print('Populating velocity map takes %.1f mins' % ((time.time()-t0)/60.))
fits.writeto(path_prefix + 'velmap_stack_ave_dBIC<{}.fits'.format(dBIC_cut), velmap_stack, overwrite=True)
fits.writeto(path_prefix + 'velmap_stack_ave_dBIC<{}_median.fits'.format(dBIC_cut), velmap_med, overwrite=True)
fits.writeto(path_prefix + 'velmap_stack_ave_dBIC<{}_mean.fits'.format(dBIC_cut), velmap_mean, overwrite=True)
fits.writeto(path_prefix + 'velmap_stack_ave_dBIC<{}_std.fits'.format(dBIC_cut), velmap_std, overwrite=True)

print(velmap_stack.shape)

# exit()


nsample=1000
flagmap = fits.getdata('../OIII_mask3d_global_proj2d.fits')
path_prefix = '../'
BICmap = fits.getdata('../OIIIonly_num_of_comp_dBIC<{}.fits'.format(dBIC_cut))
velmap_stack = np.zeros(shape=(nsample, flagmap.shape[0], flagmap.shape[1]))
velmap_stack[:,:,:] = np.nan

yy_val, xx_val = np.where(flagmap > 0)
print(len(xx_val))


# Monte Carlo sample of the velocity fields
t0 = time.time()
for xi, yi in zip(xx_val, yy_val):
    if BICmap[yi, xi] == 1: 
        chain_file = '../OIIIonly_1comp/chains_cleaned/%d_%d_OIIIonly_1comp_chain.fits' % (xi,yi)
        chains = fits.getdata(chain_file)
        redshifts = chains[:, -250:, 0].flatten() # take the last -250 samples from all chains
        vels = (redshifts - z0)/(1 + z0) * clight
        velmap_stack[:, yi, xi] = np.random.choice(vels, size=nsample, replace=True)
    elif BICmap[yi, xi] == 2:
        chain_file = '../OIIIonly_2comp/chains_cleaned/%d_%d_OIIIonly_2comp_chain.fits' % (xi,yi)
        chains = fits.getdata(chain_file)
        chains = chains[:, -250:, :] # take the last -250 samples from all chains
        vels = get_ave_vel_nowide(chains)
        velmap_stack[:, yi, xi] = np.random.choice(vels, size=nsample, replace=True)
    elif BICmap[yi, xi] == 3:
        chain_file = '../OIIIonly_3comp/chains_cleaned/%d_%d_OIIIonly_3comp_chain.fits' % (xi,yi)
        chains = fits.getdata(chain_file)
        chains = chains[:, -250:, :] # take the last -250 samples from all chains
        vels = get_ave_vel_nowide(chains)
        velmap_stack[:, yi, xi] = np.random.choice(vels, size=nsample, replace=True)
    elif BICmap[yi, xi] == 4:
        chain_file = '../OIIIonly_4comp/chains_cleaned/%d_%d_OIIIonly_4comp_chain.fits' % (xi,yi)
        chains = fits.getdata(chain_file)
        chains = chains[:, -250:, :] # take the last -250 samples from all chains
        vels = get_ave_vel_nowide(chains)
        velmap_stack[:, yi, xi] = np.random.choice(vels, size=nsample, replace=True)

velmap_med = np.median(velmap_stack, axis=0)
velmap_mean = np.mean(velmap_stack, axis=0)
velmap_std = np.std(velmap_stack, axis=0)

print('Populating velocity map takes %.1f mins' % ((time.time()-t0)/60.))
fits.writeto(path_prefix + 'velmap_stack_ave_nowide_dBIC<{}.fits'.format(dBIC_cut), velmap_stack, overwrite=True)
fits.writeto(path_prefix + 'velmap_stack_ave_nowide_dBIC<{}_median.fits'.format(dBIC_cut), velmap_med, overwrite=True)
fits.writeto(path_prefix + 'velmap_stack_ave_nowide_dBIC<{}_mean.fits'.format(dBIC_cut), velmap_mean, overwrite=True)
fits.writeto(path_prefix + 'velmap_stack_ave_nowide_dBIC<{}_std.fits'.format(dBIC_cut), velmap_std, overwrite=True)

print(velmap_stack.shape)

exit()
Collapse
'''
