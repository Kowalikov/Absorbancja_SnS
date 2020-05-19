import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("macosx")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def readData(workdir, title):
    wavlen = []
    inten = []
    phase = []
    f = open(workdir + title, "r")
    for x in f:
        st = x
        st1, st2, st3 = st.split('\t')
        wavlen.append(float(st1))
        inten.append(float(st2))
        phase.append(float(st3))

    wavlennp = np.asarray(wavlen)
    intennp = np.asarray(inten)
    phasenp = np.asarray(phase)

    d = {'wavelength': wavlen, \
         'intensivity': inten, \
         'detector phase': phase}
    df = pd.DataFrame(data=d)
    return wavlen, inten, phase, df


class MathTextSciFormatter(mticker.Formatter):
    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt
    def __call__(self, x, pos=None):
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
        return "${}$".format(s)


def plotChart(wlnp, insnp, title):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(wlnp, insnp, '-', color='firebrick')
    ax.set_title(title)
    plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.2e"))
    ax.grid(b=True, color='silver', which="both", linestyle='-', linewidth=0.5)
    plt.ylim(0, 8)
    plt.show()


def plotRegression(wlnp, absor, title, regRange1, regRange2, plot=True):
    title+=" with regression"
    xr1 = wlnp[regRange1[0]:regRange1[1]]
    yr1 = absor[regRange1[0]:regRange1[1]]
    xr2 = wlnp[regRange2[0]:regRange2[1]]
    yr2 = absor[regRange2[0]:regRange2[1]]
    coef1 = np.polyfit(xr1, yr1, 1)
    poly1d_fn1 = np.poly1d(coef1)
    coef2 = np.polyfit(xr2, yr2, 1)
    poly1d_fn2 = np.poly1d(coef2)
    x1 = wlnp[regRange1[0]:]
    x2 = wlnp[regRange2[0]:]

    if plot==True:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(wlnp, absor, '-', color='firebrick')
        ax.plot(x1, poly1d_fn1(x1), '--', color= "lightsteelblue")
        ax.plot(x2, poly1d_fn2(x2), '--', color = "cornflowerblue")
        ax.set_title(title)
        plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.2e"))
        ax.grid(b=True, color='silver', which="both", linestyle='-', linewidth=0.5)
        plt.ylim(0, 8)
        plt.show()

    return coef1, coef2


def Absorbance(i0, i):
    R=0.35
    i = np.multiply(i, 10 ** 8)
    i0 = np.add(np.multiply(i0, 10 ** 8), 0.00001)
    T = np.divide(i, i0)
    Treal = np.divide(T, (1-R)*(1-R))
    ln = np.log(Treal)
    absorbance = np.multiply(ln, -1)
    return absorbance


def absorptionAnalyse(regRange1=[551-319, 551-289], regRange2=[551-369, 551-341], t=0, material='Sns', plotIntens=False, plotReg = False, plotAny=False):
    if t==0:
        t=10
    workdir = "./data/"+material+"/"
    if material=='SnS':
        filenameRef= "ref 10K 300ms FEL650 slits300um lamp19V InGaAs.dat"
        if t<100:
            filename = material + "_0"+str(t)+"K.dat"
        else:
            filename = material + "_" + str(t) + "K.dat"
    elif material == 'Ges':
        filename = t+ "K o1_5mm szcz430u f550nm 100ms s300_800.dat"
        filenameRef = "ref o1_5mm szcz430u f550nm 100ms s300_800.dat"

    wlnp, insnp, phnp, df = readData(workdir, filename)
    wlRefnp, insRefnp, phRefnp, dfRef= readData(workdir, filenameRef)

    if plotIntens==True and plotAny==True:
        title = "Intensity for wavelength T="+str(t)+"K"
        plotChart(wlnp, insnp, title)

    absor = Absorbance(insRefnp, insnp)
    title = "Absorpion for wavelength T="+str(t)+"K"
    if plotAny==True:
        plotChart(wlnp, absor, title)

    global Eg
    if plotReg==False or plotAny==False:
        coe1, coe2 = plotRegression(wlnp, absor, title, regRange1, regRange2, plot=False)
        Eg.append([-coe1[1]/coe1[0], -coe2[1]/coe2[0], coe1[0], coe1[1], coe2[0], coe2[1]])
    else:
        coe1, coe2 = plotRegression(wlnp, absor, title, regRange1, regRange2, plot=True)
        Eg.append([-coe1[1] / coe1[0], -coe2[1] / coe2[0], coe1[0], coe1[1], coe2[0], coe2[1]])


def writeBandGaps(Eg):
    f = open("./bandgaps.txt", "w+")
    for i in range(len(Eg)):
        f.write(str(Eg[i][0]) + "\t" + str(Eg[i][1]) + "\n")
    f = open("./regression_coefficients.txt", "w+")
    for i in range(len(Eg)):
        f.write(str(Eg[i][2]) + "\t" + str(Eg[i][3]) + "\t" + str(Eg[i][4]) + "\t" + str(Eg[i][5]) + "\n")


def arrayPrep():
    temps = []
    regRan1 = []
    regRan2 = []

    for x in range(0,16):
        temps.append(x*20)

    f = open("./wlrange.txt", "r")
    for x in f:
        st = x
        st1, st2, st3, st4 = st.split('\t')
        regRan1.append([int(551-(int(st1)-700)/2), int(551-(int(st2)-700)/2)])
        regRan2.append([int(551-(int(st4)-700)/2), int(551-(int(st3)-700)/2)])

    return temps, regRan1, regRan2


Eg=[]
temps, regRan1, regRan2 = arrayPrep()
for x in temps:
    absorptionAnalyse(regRan1[int(x/20)], regRan2[int(x/20)], t=x, material='SnS', plotIntens = False, plotReg=True, plotAny=True)

writeBandGaps(Eg)
print(Eg[:][0])
Egnp=np.asarray(Eg).transpose()

temps[0]=10
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
line1, = ax.plot(temps, Egnp[:][0], 'o', color='seagreen')
line2, = ax.plot(temps, Egnp[:][1], 'o', color='deepskyblue')
line1.set_label("Eg")
line2.set_label("Eg+Ek")
ax.legend()
ax.set_title("Eg and Eg+Ek form the temperature")
plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.2e"))
ax.grid(b=True, color='silver', which="both", linestyle='-', linewidth=0.5)
plt.show()
#absorptionAnalyse(t=10, material='SnS', plotIntens=False)
