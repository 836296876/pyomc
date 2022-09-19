from msilib.schema import Directory
import numpy as np
import platform
import os
import pyomc

def windowsPath(pathName:str):
    """
    将文件路径转化为当前系统可识别的路径。
    """
    if platform.system() == "Windows":
        return pathName.replace("/","\\") + "\\"
    else:
        return pathName + "/"

@pyomc.get_runtime
def mumax3_convert(LoadDirName:str, index="all", type="-numpy"):
    """
    调用mumax3-convert快捷地转化.ovf文件为.npy文件。
    """
    LoadDirName = windowsPath(LoadDirName)
    if index == "all":
        os.system(f"mumax3-convert {type} {LoadDirName}m*.ovf")
    else:
        os.system(f"mumax3-convert {type} {LoadDirName}m" + ("000000" + str(index))[-6:] + ".ovf")
    return 0

@pyomc.get_runtime
def fold_2_axis_peraxis(LoadDirName:str, na:int, Nt:int):
    """
    (过程函数)将数据转化为变量，然后压缩，用来计算色散。
    """
    LoadDirName = windowsPath(LoadDirName)
    m000000 = np.load(LoadDirName + "m000000.npy")[na]
    Nz, Ny, Nx = m000000.shape

    if na >= 3:
        raise ValueError(f"n_axis out of range! Has 3, but call {na}.")
    fold_2_axis_Arr = np.zeros([Nt+1, Nz, Ny, Nx])
    for nt in range(Nt+1):
        fileName = LoadDirName + "m" + ("000000" + str(nt))[-6:] + ".npy"
        file = np.load(fileName)[na]
        fold_2_axis_Arr[nt, ...] = file - m000000
    
    for ny in range(Ny):
        for nz in range(Nz):
            np.save(LoadDirName + f"M^{na}_y{ny}_z{nz}.npy", fold_2_axis_Arr[:, nz, ny, :])
    return 0

@pyomc.get_runtime
def fold_2_axis(LoadDirName:str, Nt:int):
    """
    将数据转化为变量，然后压缩，用来计算色散。
    """
    # Nz, Ny, Nx = np.load(windowsPath(LoadDirName) + "m000000.npy")[0].shape
    for na in range(3):
        fold_2_axis_peraxis(LoadDirName, na, Nt)

    return 0

@pyomc.get_runtime
def tocomsol(LoadDirName, SaveDirName, dx, dy, dz):
    LoadDirName = pyomc.windowsPath(LoadDirName)
    SaveDirName = pyomc.windowsPath(SaveDirName)

    ft1dArr = np.load(LoadDirName + "ft1d.npy")
    Nz, Ny, Nx = ft1dArr.shape[1:4]
    dmXYZ = open(SaveDirName + 'tocomsol.txt', "w")

    dmXYZ.write(f"x(m)\ty(m)\tz(m)\treal(dmX)\timag(dmX)\treal(dmY)\timag(dmY)\treal(dmZ)\imag(dmZ)\n")

    for nx in range(Nx):
        x = nx * dx + dx / 2 - Nx * dx / 2
        for ny in range(Ny):
            y = ny * dy + dy / 2 - Ny * dy / 2
            for nz in range(Nz):
                z = nz * dz + dz / 2 - Nz * dz / 2
                dmXYZ.write(f"{x}\t{y}\t{z}\t{np.real(ft1dArr[0][nz][ny][nx])}\t"
                            f"{np.imag(ft1dArr[0][nz][ny][nx])}\t"
                            f"{np.real(ft1dArr[1][nz][ny][nx])}\t"
                            f"{np.imag(ft1dArr[1][nz][ny][nx])}\t"
                            f"{np.real(ft1dArr[2][nz][ny][nx])}\t"
                            f"{np.imag(ft1dArr[2][nz][ny][nx])}\n")
    dmXYZ.close()

    return 0
