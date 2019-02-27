import os
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import gaussian
from mpl_toolkits.axes_grid1 import make_axes_locatable


def minmax(x):
    mnx = x.min()
    mxx = x.max()
    return ((x - mnx) / (mxx - mnx)) - 0.5


def renorm(x, pos, neg):
    return (np.maximum(x, 0) / pos) + (((np.minimum(x, 0) * -1) / neg) * -1)


def zscore(x, mu=None, sd=None):
    if not mu:
        mu = x.mean()
    if not sd:
        sd = x.std()
    return (x - mu) / sd


def prep_subplot(f, im, ax, ticks=[-4, 4]):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    f.colorbar(im, cax=cax, orientation='horizontal', ticks=ticks)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])


def get_grad_plots(
        nne,
        nni,
        e,
        i,
        nims,
        ims,
        logits,
        nlogits,
        out_dir,
        ran=None,
        save_figs=False,
        dpi=300,
        sig=0):
    """Gradients from the hGRU on pathfinder.

    Blur the image and use to mask the gradients?
    Recolor the image by h2 + h1
    """
    font = {'family' : 'normal',
            # 'weight' : 'bold',
            'size'   : 8}
    matplotlib.rc('font', **font)
    gmax, gmin = 1, -1  # 8, 0 # np.max([nze, nzi, ze, zi])
    smax, smin = 1, -1  # 4, -4
    if ran is None:
        ran = range(len(nne))
    count = 0
    for im in ran:
        f = plt.figure()

        nze = zscore(nne[im], munne, sdnne)  # zscore(nni[im], munni, sdnni)
        nzi = zscore(nni[im], munni, sdnni)  # zscore(nne[im], munne, sdnne)
        ze = zscore(e[im], mue, sde)
        zi = zscore(i[im], mui, sdi)

        # -
        im_mask = gaussian(nims[im], 1) == 0
        nzi[im_mask] = 0
        nze[im_mask] = 0
        ax1 = plt.subplot(2, 4, 2)
        plt.title('$H^{(1)}$')
        im1 = ax1.imshow(gaussian(nzi, sig, preserve_range=True), cmap='PuOr_r', vmin=smin, vmax=smax)
        prep_subplot(f=f, im=im1, ax=ax1)

        ax2 = plt.subplot(2, 4, 3)
        plt.title('$H^{(2)}$')
        im2 = ax2.imshow(gaussian(nze, sig, preserve_range=True), cmap='PuOr_r', vmin=smin, vmax=smax)
        prep_subplot(f=f, im=im2, ax=ax2)

        ax3 = plt.subplot(2, 4, 4)
        ndif = gaussian((nze), sig, preserve_range=True) + gaussian((nzi), sig, preserve_range=True)
        plt.title('$H^{(2)} + H^{(1)}, max=%s$' % np.around(ndif.max(), 2))
        im3 = ax3.imshow((ndif), cmap='RdBu_r', vmin=gmin, vmax=gmax)
        prep_subplot(f=f, im=im3, ax=ax3, ticks=[gmin, gmax])
        plt.subplot(2, 4, 1)
        plt.title('Decision: %s' % nlogits[im])
        plt.imshow(nims[im], cmap='Greys_r')
        plt.axis('off')

        # +
        im_mask = gaussian(ims[im], 1) == 0
        zi[im_mask] = 0
        ze[im_mask] = 0
        ax4 = plt.subplot(2, 4, 6)
        plt.title('$H^{(1)}$')
        im4 = ax4.imshow(gaussian(zi, sig, preserve_range=True), cmap='PuOr_r', vmin=smin, vmax=smax)
        prep_subplot(f=f, im=im4, ax=ax4)
        ax5 = plt.subplot(2, 4, 7)
        plt.title('$H^{(2)}$')
        im5 = ax5.imshow(gaussian(ze, sig, preserve_range=True), cmap='PuOr_r', vmin=smin, vmax=smax)
        prep_subplot(f=f, im=im5, ax=ax5)
        ax6 = plt.subplot(2, 4, 8)
        dif = gaussian((ze), sig, preserve_range=True) + gaussian((zi), sig, preserve_range=True)
        plt.title('$H^{(2)} + H^{(1)}, max=%s$' % np.around(dif.max(), 2))
        im6 = ax6.imshow((dif), cmap='RdBu_r', vmin=gmin, vmax=gmax)
        prep_subplot(f=f, im=im6, ax=ax6, ticks=[gmin, gmax])
        plt.subplot(2, 4, 5)
        plt.title('Decision: %s' % logits[im])
        plt.imshow(ims[im], cmap='Greys_r')
        plt.axis('off')
        if save_figs:
            out_path = os.path.join(
                out_dir,
                '%s.png' % count)
            count += 1
            plt.savefig(out_path, dpi=dpi)
        else:
            plt.show()
        plt.close(f)


####
# data = np.load('movies/hgru_2018_07_31_10_02_24_209188_val_gradients.npz')
data = np.load('results/all_gradients_hgru_pathfinder_14_2018_08_13_14_08_27_565654_val_gradients.npz')
use_activities = False

# e = data['e_grads'][0]
# i = data['i_grads'][0]
# ims = data['og_ims'].squeeze()

e = data['exc_val_gradients'][0]
i = data['inh_val_gradients'][0]
ims = data['val_images'].squeeze()
if use_activities:
    e = (data['e_acts'][0]).mean(-1)#  / data['e_acts'][0].std(-1)
    i = (data['i_acts'][0]).mean(-1)#  / data['e_acts'][0].std(-1)
    ims = data['proc_ims'].squeeze()
e_copy = np.copy(e)
i_copy = np.copy(i)
classes = data['files']
logits = np.argmax(data['val_logits'], axis=-1).ravel()
labels = ['Path' if x else 'No path' for x in logits]

#

f_names = [int(f.split(os.path.sep)[-1].split('_')[0]) for f in classes]
f_sort = np.argsort(f_names)
e = e[f_sort]
i = i[f_sort]
ims = ims[f_sort]
logits = logits[f_sort]
labels = np.asarray(labels)[f_sort]

#

fac = len(logits) / 2
idx = np.arange(2).reshape(1, 2).repeat(fac, axis=0).reshape(-1)
nne = e[idx == 1]
nni = i[idx == 1]
e = e[idx == 0]
i = i[idx == 0]
nims = ims[idx == 1]
ims = ims[idx == 0]
nlogits = logits[idx == 1]
logits = logits[idx == 0]
nlabels = labels[idx == 1]
labels = labels[idx == 0]

#

munne = e_copy.mean()
munni = i_copy.mean()
sdnne = e_copy.std()
sdnni = i_copy.std()

mue = e_copy.mean()
mui = i_copy.mean()
sde = e_copy.std()
sdi = i_copy.std()

#

ran = None
get_grad_plots(
        nne=nne,
        nni=nni,
        e=e,
        i=i,
        nims=nims,
        ims=ims,
        logits=labels,
        nlogits=nlabels,
        ran=ran,
        save_figs=True,
        out_dir='grad_ims',
        dpi=300)


