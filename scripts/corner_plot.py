import numpy as np
import corner
from astropy.io import fits
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plot_title = "MCMC Corner Plot for UGC7342"

params = ['Redshift z','Width','Amplitude']
with fits.open('/Users/mariasanchezrincon/CASSI_SURF/UGC7342_IFS_data/mc_173_155.fits') as hdul:
    # Assuming the MCMC chain is in the first extension (HDU 1) as a binary table
    mcmc_data = hdul[0].data
    for i in range(len(mcmc_data[0,:,0])):
        fig = corner.corner(mcmc_data[:,i,:],labels=params,color='black')
        fig.suptitle(plot_title, fontsize=16, y=1.02)
        fig.subplots_adjust(top=.9)
        plt.savefig('/Users/mariasanchezrincon/CASSI_SURF/Plots/poster_figs/corner.pdf')
        plt.show()

