import matplotlib.pyplot as plt
import numpy as np

def plot_dispersion(ft2dArr, dt, dx, fmax, kedge, saveDirName="."):
    Nt, Nx = ft2dArr.shape
    df = 1 / (dt * Nt)
    dk = (2 * np.pi) / (dx * Nx)

    nfmax = int(np.ceil(fmax / df))
    nkmax = int(np.ceil(kedge / dk))

    # ft2dArr_plot = ft2dArr[:nfmax,:nkmax]
    ft2dArr_plot = np.sqrt(ft2dArr/np.max(ft2dArr[:nfmax,:nkmax]))
    nfExpect = np.where(ft2dArr_plot == np.max(ft2dArr_plot))[0][0]
    fExpect = nfExpect * df
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yticks([0, nfmax])
    ax.set_yticklabels(["0", f"{fmax/1e9:.2f}"])
    ax.set_xticks([0, nkmax])
    ax.set_xticklabels(["0", r"$\pi/a$"])
    im=ax.imshow(ft2dArr_plot[:nfmax+1,:nkmax+1],aspect="auto",origin="lower",interpolation='none')
    plt.colorbar(im)

    plt.savefig(saveDirName + "/" + "dispersion.png")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticks([0, nfmax])
    ax.set_xticklabels(["0", f"{fmax/1e9:.2f}"])
    ax.plot(ft2dArr_plot[:nfmax, nkmax+1])
    ax.annotate(f"{fExpect/1e9:.2f}GHz", (nfExpect, 1))

    plt.savefig(saveDirName + "/" + "spectrum.png")

    
    return fExpect
