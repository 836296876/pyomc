import numpy as np
import pyomc

@pyomc.get_runtime
def sw_ft2d_peraxis(LoadDirName:str, Ny:int, Nz:int, na:int):
    """
    (过程函数)对频谱做二维FFT后求和。
    """
    LoadDirName = pyomc.windowsPath(LoadDirName)
    M_y0_z0 = np.load(LoadDirName + f"M^{na}_y0_z0.npy")
    Nt, Nx = M_y0_z0.shape
    Nt = Nt - 1

    sw_ft2d_Arr = np.zeros([100000, Nx])
    # sw_ft2d_yArr = np.zeros([100000, Nx], dtype=np.complex128)
    # sw_ft2d_zArr = np.zeros([100000, Nx], dtype=np.complex128)

    for ny in range(Ny):
        for nz in range(Nz):
            fileName = LoadDirName + f"M^{na}_y{ny}_z{nz}.npy"
            file = np.load(fileName)
            sw_ft2d_Arr += np.abs(np.fft.fft2(file, [100000, Nx]))**2
            # sw_ft2d_yArr += np.abs(np.fft.fft2(file[1], [100000, Nx]))
            # sw_ft2d_zArr += np.abs(np.fft.fft2(file[2], [100000, Nx]))

            print(f"complete fft of M^{na}_y{ny}_z{nz}.npy.")

    sw_ft2d_Arr = np.sqrt(sw_ft2d_Arr)
    
    np.save(LoadDirName + f"sw_ft2d_{na}.npy", sw_ft2d_Arr)
    # np.save(LoadDirName + f"sw_ft2d_y.npy", sw_ft2d_yArr)
    # np.save(LoadDirName + f"sw_ft2d_z.npy", sw_ft2d_zArr)

    # return sw_ft2d_xArr, sw_ft2d_yArr, sw_ft2d_zArr
    return sw_ft2d_Arr

@pyomc.get_runtime
def sw_ft2d(LoadDirName:str, Ny:int, Nz:int):
    """
    对频谱做二维FFT后求和。
    """
    for na in range(3):
        sw_ft2d_peraxis(LoadDirName, Ny, Nz, na)

    return 0

@pyomc.get_runtime
def sw_ft1d(LoadDirName, Nt, dt, f):
    """
    仅对时间做一维傅里叶变换。
    """
    LoadDirName = pyomc.windowsPath(LoadDirName)
    m000000 = np.load(LoadDirName + "m000000.npy")
    Nz, Ny, Nx = m000000[0].shape
    resultArr = np.zeros([3, Nz, Ny, Nx], dtype=np.complex128)
    tArr = np.arange(Nt+1) * dt
    expiwtArr = np.exp(2j*np.pi*f*tArr)

    for nt in range(0, Nt+1):
        if nt % 100 == 0:
            print(f"Already calculate {nt}/{Nt}.")
        filename = LoadDirName + '/m' + ('000000' + str(nt))[-6:] + '.npy'
        npyfile = np.load(filename) - m000000
        resultArr += npyfile * expiwtArr[nt] * dt
        del npyfile, filename

    print(f"Already calculate {Nt}/{Nt} files.")
    np.save(LoadDirName + "ft1d.npy", resultArr)

    return resultArr