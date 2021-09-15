"""

Checking pressure profiles and scaling relation models for Arnaud+2010 and Battaglia+2012.

"""

import os
import sys
import numpy as np
import pylab as plt
from scipy import interpolate
from scipy.optimize import fmin
from scipy import stats
from nemo import gnfw
from nemo import signals
from nemo import plotSettings
import pyccl as ccl
import astropy.constants as constants
import IPython

#------------------------------------------------------------------------------------------------------------
# Constants and cosmology

# SI units
sigmaT=6.6524586e-29
me=9.10938188e-31
c=299792458.

# For P500
mu=0.59
mu_e=1.14
fB=0.175
    
# Cosmology
Om0=0.3
Ob0=0.05
H0=70
sigma8=0.8
ns=0.95
transferFunction="boltzmann_camb"
cosmoModel=ccl.Cosmology(Omega_c=Om0-Ob0, Omega_b=Ob0, h=0.01*H0, sigma8=sigma8, n_s=ns,
                         transfer_function=transferFunction)
M200mDef=ccl.halos.MassDef(200, "matter", c_m_relation = 'Bhattacharya13')
M200cDef=ccl.halos.MassDef(200, "critical", c_m_relation = 'Bhattacharya13')

#------------------------------------------------------------------------------------------------------------
def calcP(x, P0, c500, alpha, beta, gamma):
    """Actually P/P500 (dimensionless p from fitting average pressure profile in e.g. A10).
    
    This is just for checking against nemo.gnfw
    
    """
    
    p=P0 / (np.power(c500*x, gamma)* (  np.power( 1+ np.power(c500*x, alpha) , (beta-gamma)/alpha)))
    return p

#------------------------------------------------------------------------------------------------------------
def calcP_B12(x, P0, xc, alpha, beta, gamma):
    """This is just for checking against nemo.gnfw.
    
    """
    
    p=P0 * np.power(x/xc, gamma)*np.power(1+np.power(x/xc, alpha) , -beta)
    return p

#------------------------------------------------------------------------------------------------------------
def calcP500(z, M500c):
    """Calculate P500, return value in keV cm^-3
    
    """
    
    Ez=ccl.h_over_h0(cosmoModel, 1/(1+z))
    Hz=Ez*cosmoModel['H0']
    P500z_SI=(3/(8*np.pi))*np.power((500*np.power(6.67e-11, -1/4)*np.power((Hz*1000)/(3.09e16*1e6), 2))/2, 4/3)*(mu/mu_e)*fB*np.power(M500c*1.99e30, 2/3)
    P500z=P500z_SI/(1000*1.6e-19)/(100**3) # keV cm^-3
    
    return P500z

#------------------------------------------------------------------------------------------------------------
def integratedP_to_y0(integratedP, z, M500c):
    """Convert integrated pressure into y0, given z and mass to get R500c.
    
    """
    
    y0=(((sigmaT*(100**2)) / ( (me*c**2)/(1.6e-19*1000) )) * integratedP) * \
        (2.0*signals.calcR500Mpc(z, M500c, cosmoModel)*3.09e16*1e6*100.)
    
    return y0
    
#------------------------------------------------------------------------------------------------------------
def calc_y0_A10(z, M500c):
    """Given a mass and a redshift, calculate y0 assuming the Arnaud et al. (2010) pressure profile
    (completely self-similar version).
    
    Args:
        z (float): Redshift
        M500c (float): Mass (M500c) in units of solar mass.
        
    Returns:
        y0 (central Comptonization parameter)
        
    Note:
        Uses globally set cosmological model
    
    """
    
    # Here x is r/R500
    x=np.linspace(1e-5, 5.0, 1000)

    # UPP
    P0=8.403
    c500=1.177
    gamma=0.3081
    alpha=1.0510
    beta=5.4905

    P500z=calcP500(z, M500c)
    integratedP_A10_z=P500z*gnfw.integrated(0.0, params = {'gamma': gamma, 'alpha': alpha, 
                                                            'beta': beta, 'P0': P0, 'c500': c500, 
                                                            'tol': 1e-7, 'npts': 100})
    y0=integratedP_to_y0(integratedP_A10_z, z, M500c)
    
    return y0

#------------------------------------------------------------------------------------------------------------
def calc_y0_B12(z, M500c):
    """Given a mass and a redshift, calculate y0 assuming the Battaglia et al. (2012) pressure profile.
    
    Args:
        z (float): Redshift
        M500c (float): Mass (M500c) in units of solar mass.
        
    Returns:
        y0 (central Comptonization parameter)
    
    Note:
        Uses globally set cosmological model. Also, pressure profile parameters used here are for M200c, so
        mass is converted assuming the Bhattacharya et al. (2013) c-M relation (see globally set mass
        definitions used for conversion).
    
    """

    # Here x is r/R500
    x=np.linspace(1e-5, 5.0, 1000)

    P0=7.49
    P0_alpha_m=0.226
    P0_alpha_z=-0.957
    xc=0.710
    xc_alpha_m=-0.0833
    xc_alpha_z=0.853
    beta=4.19
    beta_alpha_m=0.0480
    beta_alpha_z=0.615

    # B12 M,z evolution is all specified in terms of M200c
    M200c=signals.M500cToMdef(M500c, z, M200cDef, cosmoModel)        
    P0z=P0*np.power(M200c/1e14, P0_alpha_m)*np.power(1+z, P0_alpha_z)
    xcz=xc*np.power(M200c/1e14, xc_alpha_m)*np.power(1+z, xc_alpha_z)
    betaz=beta*np.power(M200c/1e14, beta_alpha_m)*np.power(1+z, beta_alpha_z)
    # B12 -> A10 notation conversion
    betaz=betaz+0.3
    c500z=1/xcz
    gamma=0.3
    alpha=1.0
    
    P500z=calcP500(z, M500c)
    integratedP_B12_z=P500z*gnfw.integrated(0.0, params = {'gamma': gamma, 'alpha': alpha, 
                                                            'beta': betaz, 'P0': P0z, 'c500': c500z, 
                                                            'tol': 1e-7, 'npts': 100})
    y0=integratedP_to_y0(integratedP_B12_z, z, M500c)

    return y0

#------------------------------------------------------------------------------------------------------------
def calc_y0_B12_approx(z, M500c):
    """Given a mass and a redshift, estimate y0 via a simplified model of how the Battaglia et al. (2012)
    scaling relation evolves (redshift evolution has been put into the scaling relation normalization, i.e.,
    taken out of the profile calculation).
    
    Args:
        z (float): Redshift
        M500c (float): Mass (M500c) in units of solar mass.
        
    Returns:
        y0 (central Comptonization parameter)
    
    Note:
        Uses globally set cosmological model. Also, pressure profile parameters used here are for M200c, so
        mass is converted assuming the Bhattacharya et al. (2013) c-M relation (see globally set mass
        definitions used for conversion).
    
    """

    # Here x is r/R500
    x=np.linspace(1e-5, 5.0, 1000)

    P0=7.49
    P0_alpha_m=0.226
    xc=0.710
    xc_alpha_m=-0.0833
    beta=4.19
    beta_alpha_m=0.0480

    # B12 M,z evolution is all specified in terms of M200c
    M200c=signals.M500cToMdef(M500c, z, M200cDef, cosmoModel)
    P0z=P0*np.power(M200c/1e14, P0_alpha_m)
    xcz=xc*np.power(M200c/1e14, xc_alpha_m)
    betaz=beta*np.power(M200c/1e14, beta_alpha_m)
    # B12 -> A10 notation conversion
    betaz=betaz+0.3
    c500z=1/xcz
    gamma=0.3
    alpha=1.0

    P500z=calcP500(z, M500c)
    integratedP_B12_z=P500z*gnfw.integrated(0.0, params = {'gamma': gamma, 'alpha': alpha, 
                                                            'beta': betaz, 'P0': P0z, 'c500': c500z, 
                                                            'tol': 1e-7, 'npts': 100})
    y0=integratedP_to_y0(integratedP_B12_z, z, M500c)
    
    # The extra evolution compared to self-similar (approx.; see Section 5 of the main part of the script)
    y0=y0*np.power(1+z, -0.5928)
    
    return y0

#------------------------------------------------------------------------------------------------------------
def calc_y0_from_scaling_params(z, M500c, scalingRelationDict):
    """Given a mass and a redshift, estimate y0 directly from a scaling relation, without integrating any
    pressure profile.

    Args:
        z (float): Redshift
        M500c (float): Mass (M500c) in units of solar mass.
        scalingRelationDict (dict): Scaling relation dictionary in standard nemo format.
        
    Returns:
        y0 (central Comptonization parameter)
    
    """
    
    tenToA0=scalingRelationDict['tenToA0']
    B0=scalingRelationDict['B0']
    if 'Mpivot' in scalingRelationDict.keys():
        Mpivot=scalingRelationDict['Mpivot']
    else:
        Mpivot=3e14
    if 'Ez_gamma' in scalingRelationDict.keys():
        Ez_gamma=scalingRelationDict['Ez_gamma']
    else:
        Ez_gamma=2
    if 'onePlusRedshift_power' in scalingRelationDict.keys():
        onePlusRedshift_power=scalingRelationDict['onePlusRedshift_power']
    else:
        onePlusRedshift_power=0
    
    Ez=ccl.h_over_h0(cosmoModel, 1/(1+z))
    Hz=Ez*cosmoModel['H0']
    
    y0=np.power(Ez, Ez_gamma)*tenToA0*np.power(M500c/Mpivot, 1+B0)*np.power(1+z, onePlusRedshift_power)
    
    return y0
    
#------------------------------------------------------------------------------------------------------------
def makeScalingRelationPlot(y0Func, plotTitle, outFileName, plotRelativeToSelfSimilar = True, 
                            scalingRelation_zs = [0.0, 0.5, 1.0, 1.5], scalingRelation_colors = ['b', 'g', 'r', 'm'],
                            scalingRelationDict = None):
    """Makes a plot of the scal
    
    y0Func (function): The function to use to calculate y0 for a given mass, redshift (e.g., calc_y0_B12, calc_y0_A10).
    
    """

    plotSettings.update_rcParams()    
    M500cRange=np.linspace(1.0e14, 50e14, 30)

    plt.figure(figsize=(10, 8))
    plt.axes([0.18, 0.12, 0.8, 0.8])
    for z, color in zip(scalingRelation_zs, scalingRelation_colors):
        y0s=[]
        for m in M500cRange:
            if scalingRelationDict is None:
                y0s.append(y0Func(z, m))
            else:
                y0s.append(y0Func(z, m, scalingRelationDict))
        y0s=np.array(y0s)

        fitResult=stats.linregress(np.log10(M500cRange/3e14), np.log10(y0s))

        fit_tenToA0=np.power(10, fitResult.intercept)
        fit_B0=fitResult.slope-1
        
        yFit=fit_tenToA0*np.power(M500cRange/3e14, 1+fit_B0)
        if plotRelativeToSelfSimilar == True:
            Ez=ccl.h_over_h0(cosmoModel, 1/(1+z))
            plt.plot(M500cRange, np.power(Ez, -2)*y0s, '%so' % (color))
            yFit=yFit*np.power(Ez, -2)
        else:
            plt.plot(M500cRange, y0s, '%so' % (color)) 
            
        #plt.plot(M500cRange, np.power(10, fitResult.intercept)*np.power(M500cRange/3e14, fitResult.slope), 'k-')
        plt.plot(M500cRange, yFit, '%s-' % (color), 
                label = "z = %.2f; $10^{A_0}$ = %.3e; $B_0$ = %.3f" % (z, fit_tenToA0, fit_B0))
    plt.loglog()
    plt.ylim(1e-6, 1e-1)
    plt.xlabel("$M_{\\rm 500c}$ (M$_{\\odot}$)")
    if plotRelativeToSelfSimilar == True:
        plt.ylabel("$E(z)^{-2}\,y_0$")
    else:
        plt.ylabel("$y_0$")
    plt.legend()
    plt.title(plotTitle)
    plt.savefig(outFileName)
    plt.close()
    
#------------------------------------------------------------------------------------------------------------
# Main

# 1. Pressure profile plots (checking Nemo implementation) --------------------------------------------------

# Here x is r/R500
x=np.linspace(1e-5, 5.0, 1000)

# P500 is the average pressure within R500, given in keV cm^-3 (see Appendix of A10)
#P500=1.65e-3    # keV cm^-3
P500=calcP500(0, 3e14)

# PPP
P0=6.41
c500=1.81
gamma=0.31
alpha=1.33
beta=4.13
p_PPP=calcP(x, P0, c500, alpha, beta, gamma)
integratedP_PPP=P500*gnfw.integrated(0.0, params = {'gamma': gamma, 'alpha': alpha, 'beta': beta, 'P0': P0, 'c500': c500, 'tol': 1e-7, 'npts': 100})
p_nemoModified_PPP=gnfw.func(x, params = {'gamma': gamma, 'alpha': alpha, 'beta': beta, 'P0': P0, 'c500': c500, 'tol': 1e-7, 'npts': 100})

# UPP
P0=8.403
c500=1.177
gamma=0.3081
alpha=1.0510
beta=5.4905
p_UPP=calcP(x, P0, c500, alpha, beta, gamma)
integratedP_UPP=P500*gnfw.integrated(0.0, params = {'gamma': gamma, 'alpha': alpha, 'beta': beta, 'P0': P0, 'c500': c500, 'tol': 1e-7, 'npts': 100})
p_nemoModified_UPP=gnfw.func(x, params = {'gamma': gamma, 'alpha': alpha, 'beta': beta, 'P0': P0, 'c500': c500, 'tol': 1e-7, 'npts': 100})
# Section 3.3 of H13 states that pressure integrated along line of sight is 20% lower for PPP cf. UPP
pressureRatio_PPPOverUPP=integratedP_PPP/integratedP_UPP

# B12 (M500c)
P0=7.49
xc=0.710
gamma=-0.3
alpha=1.0
beta=4.19
p_B12=calcP_B12(x, P0, xc, alpha, beta, gamma)

# B12 (M500c) - A10 notation (xc == 1/c500, gamma <-> -gamma, beta = beta + 0.3)
# (this was just a check to make sure we understood the differences between A10 and B12)
P0=7.49
c500=1/0.710
gamma=0.3
alpha=1.0
beta=4.19+0.3 
p_B12_transformed=calcP(x, P0, c500, alpha, beta, gamma)
integratedP_B12=P500*gnfw.integrated(0.0, params = {'gamma': gamma, 'alpha': alpha, 'beta': beta, 'P0': P0, 'c500': c500, 'tol': 1e-7, 'npts': 100})
p_nemoModified_B12=gnfw.func(x, params = {'gamma': gamma, 'alpha': alpha, 'beta': beta, 'P0': P0, 'c500': c500, 'tol': 1e-7, 'npts': 100})

# Plot of the 1d dimensionless pressure profiles
# Top ones here (before p_B12_transformed) are just for checking implementation in Nemo
plt.figure(figsize=(7, 8))
plt.plot(x, p_UPP, label = 'UPP', zorder = 920)
plt.plot(x, p_PPP, label = 'PPP', zorder = 910)
plt.plot(x, p_B12, label = 'B12', zorder = 1000)
plt.plot(x, p_B12_transformed, label = 'B12-trans', lw =  6, zorder = 900)
plt.xlabel("$R/R_{500}$")
plt.ylabel("$P/P_{500}$")
plt.legend()
plt.loglog()
plt.ylim(2.5e-4, 350.)  # Match scales in A10 Fig. 8
plt.xlim(7e-3, 5.1)
plt.savefig("profiles.png")
plt.show()
plt.close()


# 2. Calculating scaling relation normalization (tenToA0 in nemo config file speak) -------------------------

refM500c=3e14   # i.e., pivot mass in scaling relation formulation
tenToA0_UPP=integratedP_to_y0(integratedP_UPP, 0, refM500c)
tenToA0_PPP=integratedP_to_y0(integratedP_PPP, 0, refM500c)
tenToA0_B12=integratedP_to_y0(integratedP_B12, 0, refM500c)

print("Normalisation factors (tenToA0):")
print("...  UPP = %.3e" % (tenToA0_UPP))
print("...  PPP = %.3e" % (tenToA0_PPP))
print("...  B12 = %.3e" % (tenToA0_B12))
print("NB: B12 result above does NOT account for mass scaling done by B12 equation 11")


# 3. Redshift evolution of the A10 and B12 scaling relations ------------------------------------------------

makeScalingRelationPlot(calc_y0_B12, "B12 Scaling", "B12Scaling_M500c_multi-z.png")
makeScalingRelationPlot(calc_y0_A10, "A10 Scaling", "A10Scaling_M500c_multi-z.png")
makeScalingRelationPlot(calc_y0_B12, "B12 Scaling", "B12Scaling_M500c_multi-z_noSelfSimScaling.png",
                        plotRelativeToSelfSimilar = False)
makeScalingRelationPlot(calc_y0_A10, "A10 Scaling", "A10Scaling_M500c_multi-z_noSelfSimScaling.png",
                        plotRelativeToSelfSimilar = False)


# 4. Modelling the evolution of the B12 scaling relation normalization --------------------------------------

# Normalisation is for a fixed mass - we'll use M500c = 3e14 MSun for comparison with A10
plotRelativeToSelfSimilar=True
Mpivot=3e14
zs=np.linspace(0, 2, 21)
tenToA0_B12_z=[]
Ezs=[]
for z in zs:
    tenToA0_B12_this_z=calc_y0_B12(z, Mpivot)
    tenToA0_B12_z.append(tenToA0_B12_this_z)
    Ezs.append(ccl.h_over_h0(cosmoModel, 1/(1+z)))
tenToA0_B12_z=np.array(tenToA0_B12_z)
Ezs=np.array(Ezs)

# Plot
plotSettings.update_rcParams()
plt.figure(figsize=(10, 8))
plt.axes([0.20, 0.12, 0.78, 0.8])
if plotRelativeToSelfSimilar == True:
    plt.plot(zs, np.power(Ezs, -2)*tenToA0_B12_z, 'o')
else:
    plt.plot(zs, tenToA0_B12_z, 'o') 
plt.xlabel("$z$")
if plotRelativeToSelfSimilar == True:
    plt.ylabel("$E(z)^{-2}\,10^{A_0}$")
else:
    plt.ylabel("$10^{A_0}$")
plt.savefig("B12_normalization_evolution.png")
plt.close()

# Fit for the evolution that differs from self-similar
extraEvo=tenToA0_B12_z/(tenToA0_B12_z[0]*np.power(Ezs, 2))
extraEvoIndex=np.linspace(-0.5, -0.7, 10000)
minSumSqRes=1e9
for i in extraEvoIndex:
    sumSqRes=np.sum(np.power(extraEvo-np.power(1+zs, i), 2))
    if sumSqRes < minSumSqRes:
        bestFitIndex=i
        minSumSqRes=sumSqRes

# Check plot
plotSettings.update_rcParams()
plt.figure(figsize=(10, 8))
plt.plot(zs, tenToA0_B12_z/(tenToA0_B12_z[0]*np.power(Ezs, 2)), 'ko')
plt.plot(zs, (1+zs)**bestFitIndex, 'r-', label = "$(1+z)^%.3f$" % (bestFitIndex))
plt.xlabel("$z$")
plt.ylabel("$10^{A_0}(z) / [E(z)^2\,10^{A_0}(z = 0)]$")
plt.savefig("B12_extra_evolution.png")
plt.close()


# 5. An approximate B12 scaling relation model --------------------------------------------------------------

# Uses the result above, but approximates the additional beyond self-similar z evolution
makeScalingRelationPlot(calc_y0_B12_approx, "B12 Approximate Scaling", "B12ApproxScaling_M500c_multi-z.png")
makeScalingRelationPlot(calc_y0_B12_approx, "B12 Approximate Scaling", "B12ApproxScaling_M500c_multi-z_noSelfSimScaling.png",
                        plotRelativeToSelfSimilar = False)


# 6. Simple scaling relations from parameters ---------------------------------------------------------------

# Here we just plot scaling relations, with no integration of pressure profiles, to check our scaling models
A10Dict={'tenToA0': 4.95e-05, 'B0': 0.0}
B12Dict={'tenToA0': 4.62e-05, 'B0': 0.10, 'onePlusRedshift_power': -0.5928}

makeScalingRelationPlot(calc_y0_from_scaling_params, "Simple Scaling - A10", "simpleScaling_A10_M500c_multi-z.png",
                        scalingRelationDict = A10Dict)
makeScalingRelationPlot(calc_y0_from_scaling_params, "Simple Scaling - B12", "simpleScaling_B12_M500c_multi-z.png",
                        scalingRelationDict = B12Dict)
makeScalingRelationPlot(calc_y0_from_scaling_params, "Simple Scaling - A10", "simpleScaling_A10_M500c_multi-z_noSelfSimScaling.png",
                        scalingRelationDict = A10Dict, plotRelativeToSelfSimilar = False)
makeScalingRelationPlot(calc_y0_from_scaling_params, "Simple Scaling - B12", "simpleScaling_B12_M500c_multi-z_noSelfSimScaling.png",
                        scalingRelationDict = B12Dict, plotRelativeToSelfSimilar = False)


IPython.embed()
sys.exit()
