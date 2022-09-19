import numpy as np
import matplotlib.pyplot as plt
import os

def npyfold(loadDirName, saveDirName, stepbegin, stepend, nybegin, nyend, nzbegin, nzend, ifAbs=False, saveName='folded.npy'):
    """
    对y，z轴进行求和，所得数组用于计算能谱和色散关系。
    经过试验，对y轴的一半、z轴的全部进行求和，可以得到较好的效果。
    """
    print(f"@function npyfold():\n"
          f"loadDirName={loadDirName}\n"
          f"saveDirName={saveDirName}\n"
          f"steps=[{stepbegin},{stepend}]\n"
          f"ny=[{nybegin}, {nyend}]\n"
          f"nz=[{nzbegin}, {nzend}]\n"
          f"ifAbs={ifAbs}\n"
          f"saveName={saveName}")
    filename = loadDirName + '/m000000.npy'
    Nx = np.load(filename).shape[3]
    Nt = stepend - stepbegin + 1
    resultArr = np.empty([Nt, Nx])

    for i in range(stepbegin, stepend+1):
        if i%100==0:
            print(f"Already load and save {i}/{Nt} files.")
        filename = loadDirName + '/m' + ('000000' + str(i))[-6:] + '.npy'
        npyfile = np.load(filename)[2,nzbegin:nzend,nybegin:nyend,:]
        if ifAbs==True:
            resultArr[i, :] = np.abs(npyfile.sum(axis=0)).sum(axis=0)[:]
        else:
            resultArr[i, :] = npyfile.sum(axis=0).sum(axis=0)[:]
        del npyfile, filename

    print(f"Already load and save {Nt}/{Nt} files.")
    np.save(saveDirName+'/'+saveName, resultArr)

    return resultArr

def ft2d(loadDirName, loadName, saveDirName, newNt, saveName='ft2d.npy'):
    """
    二维快速傅里叶变换。
    """
    npyfile = np.load(loadDirName+'/'+loadName)
    Nt, Nx = npyfile.shape
    fftArr = np.fft.fft2(npyfile, [newNt, Nx])
    np.save(saveDirName+'/'+saveName, fftArr)
    return fftArr

def ft1d(loadDirName, saveDirName, stepbegin, stepend, fExpect, dt, saveName='ft1d.npy'):
    r"""
    仅对时间做一维傅里叶变换。
    """
    print(f"@function ft1d():\n"
          f"loadDirName={loadDirName}\n"
          f"saveDirName={saveDirName}\n"
          f"steps=[{stepbegin},{stepend}]\n"
          f"fExpect={fExpect}\n"
          f"dt={dt}\n"
          f"saveName={saveName}")

    Nt = stepend - stepbegin + 1
    Nz, Ny, Nx = np.load(loadDirName + '/m000000.npy')[0].shape
    resultArr = np.zeros([3, Nz, Ny, Nx], dtype=np.complex128)
    tArr = np.arange(Nt) * dt
    expiwtArr = np.exp(2j*np.pi*fExpect*tArr)

    for i in range(stepbegin, stepend+1):
        if i%100==0:
            print(f"Already calculate {i}/{Nt}.")
        filename = loadDirName + '/m' + ('000000' + str(i))[-6:] + '.npy'
        npyfile = np.load(filename)
        resultArr += npyfile * expiwtArr[i] * dt
        del npyfile, filename

    # resultArr = resultArr / (Nt * dt)

    print(f"Already calculate {Nt}/{Nt} files.")
    np.save(saveDirName+'/'+saveName, resultArr)

    return resultArr

def plot2(loadDirName, loadName, saveDirName, dt, dx, kExpect, fMin, fMax, ifScan=True, NfTest=2001, dfTest=0.00001e9):
    """
    绘制频谱。
    """
    print(f"@function plot2():\n"
          f"loadDirName={loadDirName}\n"
          f"loadName={loadName}\n"
          f"saveDirName={saveDirName}\n"
          f"dt={dt}\n"
          f"dx={dx}\n"
          f"kExpect={kExpect}\n"
          f"f=[{fMin},{fMax}]\n"
          f"ifScan={ifScan}\n"
          f"NfTest={NfTest}\n"
          f"dfTest={dfTest}"
          )

    foldArr = np.load(loadDirName+'/'+loadName)
    Nt, Nx = foldArr.shape
    ft1Arr = np.zeros(Nt, dtype=np.complex128)  # 第一步，对指定k(kExpect)做连续傅里叶变换
    xArr = (np.arange(Nx) - Nx/2 + 1/2) * dx
    expikxArr = np.exp(1j * kExpect * xArr)
    for i in range(Nx):
        ft1Arr += foldArr[:, i] * expikxArr[i] * dx
    ft2Arr = np.fft.fft(ft1Arr, 100000)         # 第二步，对时间做快速傅里叶变换，默认补零到100000个时间步
    nfMin = int(fMin * 100000 * dt)
    nfMax = int(fMax * 100000 * dt)
    ft2ArrPlot = np.sqrt(np.abs(np.real(ft2Arr[nfMin:nfMax]))/np.max(np.abs(np.real(ft2Arr[nfMin:nfMax]))))
    ft2ArrPlot_imag = np.sqrt(np.abs(np.imag(ft2Arr[nfMin:nfMax]))/np.max(np.abs(np.imag(ft2Arr[nfMin:nfMax]))))
    nfExpect = np.where(ft2ArrPlot==np.max(ft2ArrPlot))[0][0] + nfMin
    fExpect = nfExpect / (100000 * dt)
    plt.figure()
    plt.plot(ft2ArrPlot)
    plt.plot(ft2ArrPlot_imag)
    plt.xlabel("f")
    plt.ylabel(r"$FT_{2d}[mZ(t,x)]$")
    plt.xticks([])
    plt.yticks([])
    plt.title("Mode Spectrum")
    plt.annotate(f"{fExpect/1e9:.2f}GHz", (nfExpect, 1))
    plt.savefig(saveDirName+'/'+'Mode Spectrum.png')

    # 绘制细化的频谱，得到精确的极大共振频率
    if ifScan==True:
        nfTestArr = np.arange(-NfTest//2,NfTest//2+1)
        fTestArr = nfTestArr * dfTest + fExpect
        tArr = np.arange(Nt) * dt
        expiwtArr = np.exp(2j*np.pi*np.outer(fTestArr, tArr))
        ft2TestArr = np.zeros(NfTest, dtype=np.complex128)
        for i in range(NfTest):
            ft2TestArr[i] = np.dot(ft1Arr, expiwtArr[i]) * dt
        ft2TestArrPlot = np.sqrt(np.abs(np.real(ft2TestArr))/np.max(np.abs(np.real(ft2TestArr))))
        nfExpect = np.where(ft2TestArrPlot==np.max(ft2TestArrPlot))[0][0]
        fExpect = fExpect + (nfExpect-NfTest//2)*dfTest
        plt.figure()
        plt.plot(ft2TestArrPlot)
        plt.xlabel("f")
        plt.ylabel(r"$FT_{2d}[mZ(t,x)]$")
        plt.xticks([])
        plt.yticks([])
        plt.title("Mode Spectrum")
        plt.annotate(f"{fExpect/1e9:.5f}GHz", (nfExpect, 1))
        plt.savefig(saveDirName+'/'+'Mode Spectrum (scan).png')

    print(f"The max energy mode is at {fExpect/1e9:.9f}GHz")

    return fExpect

def plot3(loadDirName, loadName, saveDirName):
    """
    绘制共振模式。
    """
    print(f"@function plot3():\n"
          f"loadDirName={loadDirName}\n"
          f"loadName={loadName}\n"
          f"saveDirName={saveDirName}\n"
          )

    if os.path.exists(loadDirName+'/'+loadName):
        modeShapeArr = np.load(loadDirName+'/'+loadName)
    else:
        # modeShapeArr = ft1d(loadDirName, saveDirName, 0, Nt-1, fExpect, dt)
        print(f"No such file: {loadDirName+'/'+loadName}")
    modeShapeArrPlot = modeShapeArr[2,int(modeShapeArr.shape[1]//2+modeShapeArr.shape[1]%2),...]
    modeShapeArrPlotAbs = np.abs(modeShapeArrPlot) / np.abs(np.max(modeShapeArrPlot))
    modeShapeArrPlotReal = np.real(modeShapeArrPlot) / np.real(np.max(modeShapeArrPlot))
    modeShapeArrPlotImag = np.imag(modeShapeArrPlot) / np.imag(np.max(modeShapeArrPlot))
    plt.figure()
    plt.imshow(modeShapeArrPlotAbs, cmap='jet')
    plt.axis('off')
    plt.title(r'Mode Shape Simulation($|\delta m_z|$)')
    plt.colorbar(orientation='horizontal', fraction=0.02, pad=0.17)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(saveDirName+'/'+'Mode Shape Simulation(abs(dmZ)).png')
    plt.figure()
    plt.imshow(modeShapeArrPlotReal, cmap='jet')
    plt.axis('off')
    plt.title(r'Mode Shape Simulation($Re[\delta m_z]$)')
    plt.colorbar(orientation='horizontal', fraction=0.02, pad=0.17)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(saveDirName+'/'+'Mode Shape Simulation(Re[dmZ]).png')
    plt.figure()
    plt.imshow(modeShapeArrPlotImag, cmap='jet')
    plt.axis('off')
    plt.title(r'Mode Shape Simulation($Im[\delta m_z]$)')
    plt.colorbar(orientation='horizontal', fraction=0.02, pad=0.17)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(saveDirName+'/'+'Mode Shape Simulation(Im[dmZ]).png')

    return 0

    """
    自动处理全部内容。
    """

    # 绘制频谱
    foldArr = np.load(loadDirName+'/'+loadName)
    Nt, Nx = foldArr.shape
    ft1Arr = np.zeros(Nt, dtype=np.complex128)  # 第一步，对指定k(kExpect)做连续傅里叶变换
    xArr = (np.arange(Nx) - Nx/2 + 1/2) * dx
    expikxArr = np.exp(1j * kExpect * xArr)
    for i in range(Nx):
        ft1Arr += foldArr[:, i] * expikxArr[i] * dx
    ft2Arr = np.fft.fft(ft1Arr, 100000)         # 第二步，对时间做快速傅里叶变换，默认补零到100000个时间步
    nfMin = int(fMin * 100000 * dt)
    nfMax = int(fMax * 100000 * dt)
    ft2ArrPlot = np.sqrt(np.abs(ft2Arr[nfMin:nfMax])/np.max(np.abs(ft2Arr[nfMin:nfMax])))
    nfExpect = np.where(ft2ArrPlot==np.max(ft2ArrPlot))[0][0]
    fExpect = nfExpect / (100000 * dt)
    plt.figure()
    plt.plot(ft2ArrPlot)
    plt.xlabel("f")
    plt.ylabel(r"$FT_{2d}[mZ(t,x)]$")
    plt.xticks([])
    plt.yticks([])
    plt.title("Mode Spectrum")
    plt.annotate(f"{fExpect/1e9:.2f}GHz", (nfExpect, 1))
    plt.savefig(saveDirName+'/'+'Mode Spectrum.png')

    # 绘制细化的频谱，得到精确的极大共振频率
    if ifScan==True:
        nfTestArr = np.arange(-NfTest//2,NfTest//2+1)
        fTestArr = nfTestArr * dfTest + fExpect
        tArr = np.arange(Nt) * dt
        expiwtArr = np.exp(2j*np.pi*np.outer(fTestArr, tArr))
        ft2TestArr = np.zeros(NfTest, dtype=np.complex128)
        for i in range(NfTest):
            ft2TestArr[i] = np.dot(ft1Arr, expiwtArr[i]) * dt
        ft2TestArrPlot = np.sqrt(np.abs(ft2TestArr)/np.max(np.abs(ft2TestArr)))
        nfExpect = np.where(ft2TestArrPlot==np.max(ft2TestArrPlot))[0][0]
        fExpect = fExpect + (nfExpect-NfTest//2)*dfTest
        plt.figure()
        plt.plot(ft2TestArrPlot)
        plt.xlabel("f")
        plt.ylabel(r"$FT_{2d}[mZ(t,x)]$")
        plt.xticks([])
        plt.yticks([])
        plt.title("Mode Spectrum")
        plt.annotate(f"{fExpect/1e9:.5f}GHz", (nfExpect, 1))
        plt.savefig(saveDirName+'/'+'Mode Spectrum (scan).png')

    print(f"The max energy mode is at {fExpect/1e9:.9f}GHz")

    # 绘制共振模式
    if os.path.exists(loadDirName+'/ft1d.npy'):
        modeShapeArr = np.load(loadDirName+'/ft1d.npy')
    else:
        modeShapeArr = ft1d(loadDirName, saveDirName, 0, Nt-1, fExpect, dt)
    modeShapeArrPlot = modeShapeArr[2,int(modeShapeArr.shape[1]//2+modeShapeArr.shape[1]%2),...]
    modeShapeArrPlotAbs = np.abs(modeShapeArrPlot) / np.abs(np.max(modeShapeArrPlot))
    modeShapeArrPlotReal = np.real(modeShapeArrPlot) / np.real(np.max(modeShapeArrPlot))
    modeShapeArrPlotImag = np.imag(modeShapeArrPlot) / np.imag(np.max(modeShapeArrPlot))
    plt.figure()
    plt.imshow(modeShapeArrPlotAbs, cmap='jet')
    plt.axis('off')
    plt.title(r'Mode Shape Simulation($|\delta m_z|$)')
    plt.colorbar(orientation='horizontal', fraction=0.02, pad=0.17)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(saveDirName+'/'+'Mode Shape Simulation(abs(dmZ)).png')
    plt.figure()
    plt.imshow(modeShapeArrPlotReal, cmap='jet')
    plt.axis('off')
    plt.title(r'Mode Shape Simulation($Re[\delta m_z]$)')
    plt.colorbar(orientation='horizontal', fraction=0.02, pad=0.17)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(saveDirName+'/'+'Mode Shape Simulation(Re[dmZ]).png')
    plt.figure()
    plt.imshow(modeShapeArrPlotImag, cmap='jet')
    plt.axis('off')
    plt.title(r'Mode Shape Simulation($Im[\delta m_z]$)')
    plt.colorbar(orientation='horizontal', fraction=0.02, pad=0.17)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(saveDirName+'/'+'Mode Shape Simulation(Im[dmZ]).png')

    return 0

def tocomsol(loadDirName, loadName, saveDirName, saveName, dx, dy, dz):
    """
    将模态转化为可供comsol读取的表格。
    """
    print(f"@function tocomsol():\n"
          f"loadDirName={loadDirName}\n"
          f"loadName={loadName}\n"
          f"saveDirName={saveDirName}\n"
          f"saveName={saveName}\n"
          f"dx={dx}\n"
          f"dy={dy}\n"
          f"dz={dz}\n"
          )

    ft1dArr = np.load(loadDirName+'/'+loadName)
    Nz, Ny, Nx = ft1dArr.shape[1:4]
    realX = open(saveDirName+'/'+saveName+'_RX.txt', "w")
    imagX = open(saveDirName+'/'+saveName+'_IX.txt', "w")
    realY = open(saveDirName+'/'+saveName+'_RY.txt', "w")
    imagY = open(saveDirName+'/'+saveName+'_IY.txt', "w")
    realZ = open(saveDirName+'/'+saveName+'_RZ.txt', "w")
    imagZ = open(saveDirName+'/'+saveName+'_IZ.txt', "w")

    realX.write(f"x(m)\ty(m)\tz(m)\treal(dmX)\n")
    imagX.write(f"x(m)\ty(m)\tz(m)\timag(dmX)\n")
    realY.write(f"x(m)\ty(m)\tz(m)\treal(dmY)\n")
    imagY.write(f"x(m)\ty(m)\tz(m)\timag(dmY)\n")
    realZ.write(f"x(m)\ty(m)\tz(m)\treal(dmZ)\n")
    imagZ.write(f"x(m)\ty(m)\tz(m)\timag(dmZ)\n")

    for nx in range(Nx):
        x = nx * dx + dx / 2 - Nx * dx / 2
        for ny in range(Ny):
            y = ny * dy + dy / 2 - Ny * dy / 2
            for nz in range(Nz):
                z = nz * dz + dz / 2 - Nz * dz / 2
                realX.write(f"{x}\t{y}\t{z}\t{np.real(ft1dArr[0][nz][ny][nx])}\n")
                imagX.write(f"{x}\t{y}\t{z}\t{np.imag(ft1dArr[0][nz][ny][nx])}\n")
                realY.write(f"{x}\t{y}\t{z}\t{np.real(ft1dArr[1][nz][ny][nx])}\n")
                imagY.write(f"{x}\t{y}\t{z}\t{np.imag(ft1dArr[1][nz][ny][nx])}\n")
                realZ.write(f"{x}\t{y}\t{z}\t{np.real(ft1dArr[2][nz][ny][nx])}\n")
                imagZ.write(f"{x}\t{y}\t{z}\t{np.imag(ft1dArr[2][nz][ny][nx])}\n")

    realX.close()
    imagX.close()
    realY.close()
    imagY.close()
    realZ.close()
    imagZ.close()

    return 0

def auto_run(loadDirName, saveDirName, Nt, nybegin, nyend, nzbegin, nzend, ifAbs, dt, dx, dy, dz, kExpect, fMin, fMax):
    print(f"@function auto_run():\n"
          f"loadDirName={loadDirName}\n"
          f"saveDirName={saveDirName}\n"
          f"Nt={Nt}\n"
          f"ny=[{nybegin}, {nyend}]\n"
          f"nz=[{nzbegin}, {nzend}]\n"
          f"ifAbs={ifAbs}\n"
          f"dt={dt}\n"
          f"dx={dx}\n"
          f"dy={dy}\n"
          f"dz={dz}\n"
          f"kExpect={kExpect}\n"
          f"f=[{fMin},{fMax}]\n"
          )
    foldedArr = npyfold(loadDirName, loadDirName, 0, Nt-1, nybegin, nyend, nzbegin, nzend, ifAbs)
    fExpect = plot2(loadDirName, 'folded.npy', saveDirName, dt, dx, kExpect, fMin, fMax)
    ft1dArr = ft1d(loadDirName, loadDirName, 0, Nt-1, fExpect, dt)
    plot3(loadDirName, 'ft1d.npy', saveDirName)
    tocomsol(loadDirName, 'ft1d.npy', saveDirName, 'tocomsol.txt', dx, dy, dz)