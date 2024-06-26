
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import os

def batch_histogram(data_tensor, num_classes=-1,  weights=None):
    """
    From. https://github.com/pytorch/pytorch/issues/99719#issuecomment-1760112194
    Computes histograms of integral values, even if in batches (as opposed to torch.histc and torch.histogram).
    Arguments:
        data_tensor: a D1 x ... x D_n torch.LongTensor
        num_classes (optional): the number of classes present in data.
                                If not provided, tensor.max() + 1 is used (an error is thrown if tensor is empty).
    Returns:
        A D1 x ... x D_{n-1} x num_classes 'result' torch.LongTensor,
        containing histograms of the last dimension D_n of tensor,
        that is, result[d_1,...,d_{n-1}, c] = number of times c appears in tensor[d_1,...,d_{n-1}].
    """
    maxd = data_tensor.max()
    nc = (maxd+1) if num_classes <= 0 else num_classes
    hist = torch.zeros((*data_tensor.shape[:-1], nc), dtype=data_tensor.dtype, device=data_tensor.device)
    if weights is not None:
        wts = weights
    else:
        wts = torch.tensor(1, dtype=hist.dtype, device=hist.device).expand(data_tensor.shape)
    hist.scatter_add_(-1, ((data_tensor * nc) // (maxd+1)).long(), wts)
    return hist


def powerspect(x):
    c, h, w = x.shape

    # 2D power
    pwr = torch.fft.fftn(x, dim=(-2,-1), norm='ortho').abs()**2
    pwr = torch.fft.fftshift(pwr, dim=(-2,-1)).to(torch.float32)
    
    # Azimuthal average
    xx, yy = torch.meshgrid(torch.arange(h, device=pwr.device), torch.arange(w, device=pwr.device), indexing='ij')
    k = torch.hypot(xx - h//2, yy - w/2).to(torch.float32)

    sort = torch.argsort(k.flatten())
    k_sort = k.flatten()[sort]
    pwr_sort = pwr.reshape(c,-1)[:,sort]

    nbins = min(h//2, w//2)
    k_bins = torch.linspace(0, k_sort.max() + 1, nbins)
    k_bin_centers = 0.5*(k_bins[1:] + k_bins[:-1])
    k_sort_stack = torch.tile(k_sort, dims=(c,1))

    pwr_binned = batch_histogram(k_sort_stack, weights=pwr_sort, num_classes=nbins-1)
    count_binned = batch_histogram(k_sort_stack, num_classes=nbins-1)

    return k_bin_centers.detach().cpu().numpy(), (pwr_binned/count_binned).detach().cpu().numpy()

def compute_ps1d(generated, target, fields, diffusion_channels):

    assert generated.shape == target.shape

    # Comppute PS1D, all channels
    with torch.no_grad():
        k, Pk_gen = powerspect(generated)
        _, Pk_tar = powerspect(target)

    # Make plots and save metrics
    figs = {}
    ratios = {}
    for i, _f in enumerate(fields):
        cidx = diffusion_channels.index(_f)
        f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2,1], 'hspace':0}, figsize=(6,4))
        a0.plot(k, Pk_tar[cidx], 'k-', label=_f)
        a0.plot(k, Pk_gen[cidx], 'r-', label='prediction')
        a0.set_yscale('log')
        a0.set_xscale('log')
        a0.set_xlabel('Wavenumber')
        a0.set_ylabel('PS1D')
        a0.tick_params(axis='x', direction='in', labelbottom=False, which='both')
        a0.tick_params(axis='x', length=5, which='major')
        a0.tick_params(axis='x', length=3, which='minor')
        a0.legend()

        ratio = Pk_gen[cidx]/Pk_tar[cidx]
        a1.plot(k, ratio, 'r-')
        a1.plot(k, np.ones(k.shape), 'k--')
        a1.set_xlabel('Wavenumber')
        a1.set_ylabel('Ratio')
        a1.set_xscale('log')
        a1.set_ylim((0,2))
        a1.minorticks_on()
        a1.tick_params(axis='x', top=True, direction='inout', labeltop=False, which='both')
        a1.tick_params(axis='x', length=5, which='major')
        a1.tick_params(axis='x', length=3, which='minor')
        figs['PS1D_'+_f] = f
        ratios['specratio_'+_f] = np.mean(ratio)
    
    return figs, ratios 




# def plot_ps1d(pk_gen, pk_tar, k, fields, plot_dir="./", nyquist_lim: int=None):
#     assert pk_gen.shape == pk_tar.shape

#     # Ensure plot directory exists
#     if not os.path.exists(plot_dir):
#         os.makedirs(plot_dir)

#     # Make plots and save metrics
#     figs = {}
#     ratios = {}
#     for i, _f in enumerate(fields):
#         cidx = i

#         # Apply Nyquist limit if provided
#         if nyquist_lim is not None:
#             k_lim = k[k <= nyquist_lim]
#             pk_gen_lim = pk_gen[cidx, :len(k_lim)]
#             pk_tar_lim = pk_tar[cidx, :len(k_lim)]
#         else:
#             k_lim = k
#             pk_gen_lim = pk_gen[cidx]
#             pk_tar_lim = pk_tar[cidx]

#         f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2,1], 'hspace':0}, figsize=(6,4))
#         a0.plot(k_lim, pk_tar_lim, 'k-', label=_f)
#         a0.plot(k_lim, pk_gen_lim, 'r-', label='prediction')
#         a0.set_yscale('log')
#         a0.set_xscale('log')
#         a0.set_xlabel('Wavenumber')
#         a0.set_ylabel('PS1D')
#         a0.tick_params(axis='x', direction='in', labelbottom=False, which='both')
#         a0.tick_params(axis='x', length=5, which='major')
#         a0.tick_params(axis='x', length=3, which='minor')
#         a0.legend()

#         ratio = pk_gen_lim / pk_tar_lim
#         a1.plot(k_lim, ratio, 'r-')
#         a1.plot(k_lim, np.ones(k_lim.shape), 'k--')
#         a1.set_xlabel('Wavenumber')
#         a1.set_ylabel('Ratio')
#         a1.set_xscale('log')
#         a1.set_ylim((0,2))
#         a1.minorticks_on()
#         a1.tick_params(axis='x', top=True, direction='inout', labeltop=False, which='both')
#         a1.tick_params(axis='x', length=5, which='major')
#         a1.tick_params(axis='x', length=3, which='minor')

#         # Save figure
#         plot_path = os.path.join(plot_dir, f'PS1D_{_f}.png')
#         f.savefig(plot_path)

#         figs['PS1D_'+_f] = f
#         ratios['specratio_'+_f] = np.mean(ratio)

#     return figs, ratios



def plot_ps1d(pk_gen, pk_tar, k, fields, plot_dir="./", nyquist_lim: int=None):
    assert pk_gen.shape == pk_tar.shape

    # Ensure plot directory exists
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Setting font sizes
    plt.rcParams['axes.labelsize'] = 16  # Font size for axis labels
    plt.rcParams['axes.titlesize'] = 18 # Font size for titles
    plt.rcParams['xtick.labelsize'] = 16  # Font size for X-tick labels
    plt.rcParams['ytick.labelsize'] = 16  # Font size for Y-tick labels
    plt.rcParams['legend.fontsize'] = 16  # Font size for legend
    linewidth=3
    # Make plots and save metrics
    figs = {}
    ratios = {}
    for i, _f in enumerate(fields):
        cidx = i

        # Apply Nyquist limit if provided
        if nyquist_lim is not None:
            k_lim = k[k <= nyquist_lim]
            pk_gen_lim = pk_gen[cidx, :len(k_lim)]
            pk_tar_lim = pk_tar[cidx, :len(k_lim)]
        else:
            k_lim = k
            pk_gen_lim = pk_gen[cidx]
            pk_tar_lim = pk_tar[cidx]

        f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2,1], 'hspace':0}, figsize=(6,4))
        a0.plot(k_lim, pk_tar_lim, 'k-', label=_f,linewidth=linewidth)
        a0.plot(k_lim, pk_gen_lim, 'r-', label='prediction',linewidth=linewidth)
        a0.set_yscale('log')
        a0.set_xscale('log')
        a0.set_xlabel('Wavenumber')
        a0.set_ylabel('PS1D')
        a0.legend()

        ratio = pk_gen_lim / pk_tar_lim
        a1.plot(k_lim, ratio, 'r-',linewidth=linewidth)
        a1.plot(k_lim, np.ones(k_lim.shape), 'k--',linewidth=linewidth)
        a1.set_xlabel('Wavenumber')
        a1.set_ylabel('Ratio')
        a1.set_xscale('log')
        a1.set_ylim((0,2))

        # Save figure
        plot_path = os.path.join(plot_dir, f'PS1D_{_f}.png')
        f.savefig(plot_path)

        figs['PS1D_'+_f] = f
        ratios['specratio_'+_f] = np.mean(ratio)

    # Reset to default (optional, if you want to revert the changes after plotting)
    plt.rcdefaults()

    return figs, ratios