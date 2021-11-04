"""

Take LIRA output, convert chains to format that GetDist likes, and make plots.

"""

import os
import sys
import numpy as np
import astropy.table as atpy
import getdist
from getdist import plots
import pylab as plt
from cycler import cycler
from astLib import *
import IPython

#------------------------------------------------------------------------------------------------------------
def update_rcParams(dict={}):
    """Based on Cristobal's preferred settings. Updates matplotlib rcParams in place.

    """
    default = {}
    for tick in ('xtick', 'ytick'):
        default['{0}.major.size'.format(tick)] = 8
        default['{0}.minor.size'.format(tick)] = 4
        default['{0}.major.width'.format(tick)] = 2
        default['{0}.minor.width'.format(tick)] = 2
        default['{0}.labelsize'.format(tick)] = 20
        default['{0}.direction'.format(tick)] = 'in'
    default['xtick.top'] = True
    default['ytick.right'] = True
    default['axes.linewidth'] = 2
    default['axes.labelsize'] = 22
    default['font.size'] = 22
    default['font.family']='sans-serif'
    default['legend.fontsize'] = 18
    default['lines.linewidth'] = 2

    for key in default:
        plt.rcParams[key] = default[key]
    # if any parameters are specified, overwrite anything previously
    # defined
    for key in dict:
        plt.rcParams[key] = dict[key]

    # From https://github.com/mhasself/rg_friendly
    plt.rcParams['axes.prop_cycle']=cycler(color=['#2424f0','#df6f0e','#3cc03c','#d62728','#b467bd','#ac866b','#e397d9','#9f9f9f','#ecdd72','#77becf'])

#------------------------------------------------------------------------------------------------------------
def convertLIRAChainsToGetDist(inFileName, outChainsDirName):
    """Convert LIRA MCMC chain into GetDist format.

    Args:
        inFileName (str): Name of file containing output from export.mcmc in LIRA.
        outChainsDirName (str): Name of the output chains directory in GetDist format that will be created.

    Returns:
        None

    """

    os.makedirs(outChainsDirName, exist_ok = True)

    with open(inFileName, "r") as inFile:
        lines=inFile.readlines()
    parNames=lines[0].replace('"', '').split()

    with open(outChainsDirName+os.path.sep+outChainsDirName+".paramnames", "w") as outFile:
        for p in parNames:
            outFile.write("%s\n" % (p))

    with open(outChainsDirName+os.path.sep+outChainsDirName+"_1.txt", "w") as outFile:
        for line in lines[1:]:
            bits=line.split()
            outLine="1 0"
            for b in bits:
                outLine=outLine+" %.6e" % (float(b))
            outFile.write(outLine+"\n")

#------------------------------------------------------------------------------------------------------------
def parseMargeStatsFile(fileName):
    """Parse getdist *.margestats file.

    Args:
        fileName (str): Name of getdist produced *.margestats file

    Returns:
        Dictionary, indexed by parameter name.

    """

    with open(fileName, "r") as inFile:
        lines=inFile.readlines()

    parKeyNames=lines[2].split()
    fitDict={}
    for line in lines[3:]:
        bits=line.split()
        parName=bits[0]
        fitDict[parName]={}
        for b, p in zip(bits[1:], parKeyNames[1:]):
            try:
                fitDict[parName][p]=float(b)
            except:
                fitDict[parName][p]=b

    return fitDict

#------------------------------------------------------------------------------------------------------------
# Main

# LX-T fit, corresponds to line 1 in Table 4 of Lovisari et al. 2020
rootName="LX-T_line1"
xPivot=5    # T, keV
yPivot=5    # LX, 1e44 erg/s

# Lovisari et al. best fit as written in their paper
# This corresponds to line 1 of LIRA LX-T fits in Table 4
lovDict={'alpha.YIZ': {'mean': -0.250, 'sddev': 0.045},
         'beta.YIZ': {'mean': 3.110, 'sddev': 0.422},
         'gamma.YIZ': {'mean': 0.398, 'sddev': 0.939},
         'sigma.XIZ.0': {'mean': 0.051, 'sddev': 0.010},
         'sigma.YIZ.0': {'mean': 0.052, 'sddev': 0.041}}

# Load/convert chains
LIRAChainsFileName="%s.dat" % (rootName)
getDistChainsDir="%s_chains" % (rootName)
convertLIRAChainsToGetDist(LIRAChainsFileName, getDistChainsDir)

# Triangle plot
analysis_settings = {'ignore_rows': '0'} # I _think_ burn-in is already removed - if not, add here
g=plots.get_subplot_plotter(chain_dir = os.path.abspath(getDistChainsDir), analysis_settings = analysis_settings)
roots=[getDistChainsDir]
params=['alpha.YIZ', 'beta.YIZ', 'gamma.YIZ', 'sigma.YIZ.0', 'sigma.XIZ.0']
g.triangle_plot(roots, params, filled = True)
g.export("triangle_%s.png" % (rootName))

# Extract best fit values from chains
os.system("getdist %s/%s" % (getDistChainsDir, getDistChainsDir))
fitDict=parseMargeStatsFile(getDistChainsDir+".margestats")

# Check our LIRA fit results against Lovisari et al. fit results --------------------------------------------
for parName in ['alpha.YIZ', 'beta.YIZ', 'gamma.YIZ', 'sigma.XIZ.0', 'sigma.YIZ.0']:
    us=fitDict[parName]['mean']
    errUs=fitDict[parName]['sddev']
    them=lovDict[parName]['mean']
    errThem=lovDict[parName]['sddev']
    diffSigma=(us-them) / np.sqrt(errUs**2 + errThem**2)
    print("%s = %.3f +/- %.3f [this run] ; %.3f +/- %.3f [Lovisari+2020] ; diff = %.3f sigma" % (parName, us, errUs, them, errThem, diffSigma))

# Make scaling relation comparison plot - compare with left panel of Fig. 4 in Lovisari et al. --------------

# Load data
tab=atpy.Table().read("XMM-PSZ_Lovisari2020.fits")
Ez=[]
for z in tab['z']:
    Ez.append(astCalc.Ez(z))
tab['Ez']=Ez

# Plot data
# NOTE: plotting x E(z)^{-1} here because Lovisari et al. do - but x E(z)^{gamma} would make more sense
update_rcParams()
fig=plt.figure(figsize=(10,8))
plt.errorbar(tab['kT'], np.power(tab['Ez'], -1)*tab['LX']*1e44, xerr = tab['E_kT'], yerr = tab['E_LX']*1e44, fmt = 'D')

# Plot fits
plotRange=np.linspace(0.1, 15, 1000)
plt.plot(plotRange*xPivot, 1e44*yPivot*np.power(10, lovDict['alpha.YIZ']['mean'])*np.power(plotRange, fitDict['beta.YIZ']['mean']),
         '-', label = 'Lovisari+2020 best fit')
plt.plot(plotRange*xPivot, 1e44*yPivot*np.power(10, fitDict['alpha.YIZ']['mean'])*np.power(plotRange, fitDict['beta.YIZ']['mean']),
         '-', label = 'Reproduced LIRA fit')

plt.loglog()
plt.ylabel("$E(z)^{-1}$ $L_{\\rm X[0.1-2.4\,keV]}$ (erg/s)")
plt.xlabel("$kT$ (keV)")
plt.ylim(5e43, 5e45)
plt.xlim(1, 20)
plt.legend()
plt.savefig("scalingRelationPlot_%s.png" % (rootName))
plt.close()

IPython.embed()
sys.exit()

