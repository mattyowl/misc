"""

Mass/radius conversion/calculation code using CCL - taken out of Nemo

"""

import pyccl as ccl
import numpy as np

#------------------------------------------------------------------------------------------------------------
Om0=0.3
Ob0=0.05
H0=70
sigma8=0.8
ns=0.95
transferFunction="boltzmann_camb"

fiducialCosmoModel=ccl.Cosmology(Omega_c=Om0-Ob0, Omega_b=Ob0, h=0.01*H0, sigma8=sigma8, n_s=ns,
                                transfer_function=transferFunction)

M200mDef=ccl.halos.MassDef(200, "matter", c_m_relation = 'Bhattacharya13')
M200cDef=ccl.halos.MassDef(200, "critical", c_m_relation = 'Bhattacharya13')
M500cDef=ccl.halos.MassDef(500, "critical")

#------------------------------------------------------------------------------------------------------------
def calcRDeltaMpc(z, MDelta, cosmoModel, delta = 500, wrt = 'critical'):
    """Calculate RDelta (e.g., R500c, R200m etc.) in Mpc, for a halo with the given mass and redshift.

    Args:
        z (float): Redshift.
        MDelta (float): Halo mass in units of solar masses, using the definition set by `delta` and `wrt`.
        cosmoModel (:obj:`pyccl.Cosmology`): Cosmology object.
        delta (float, optional): Overdensity (e.g., typically 500 or 200).
        wrt (str, optional): Use 'critical' or 'mean' to set the definition of density with respect to the
            critical density or mean density at the given redshift.

    Returns:
        RDelta (in Mpc)

    """

    if type(MDelta) == str:
        raise Exception("MDelta is a string - use, e.g., 1.0e+14 (not 1e14 or 1e+14)")

    Ez=ccl.h_over_h0(cosmoModel, 1/(1+z))
    if wrt == 'critical':
        wrtDensity=ccl.physical_constants.RHO_CRITICAL*(Ez*cosmoModel['h'])**2
    elif wrt == 'mean':
        wrtDensity=ccl.omega_x(cosmoModel, 1/(1+z), 'matter')*ccl.physical_constants.RHO_CRITICAL*(Ez*cosmoModel['h'])**2
    else:
        raise Exception("wrt should be either 'critical' or 'mean'")
    RDeltaMpc=np.power((3*MDelta)/(4*np.pi*delta*wrtDensity), 1.0/3.0)

    return RDeltaMpc

#------------------------------------------------------------------------------------------------------------
def M500cToMdef(M500c, z, massDef, cosmoModel):
    """Convert M500c to some other mass definition.

    massDef (`obj`:ccl.halos.MassDef): CCL halo mass definition

    """

    M500cDef=ccl.halos.MassDef(500, "critical")

    tolerance=1e-5
    scaleFactor=3.0
    ratio=1e6
    count=0
    while abs(1.0-ratio) > tolerance:
        testM500c=massDef.translate_mass(cosmoModel, scaleFactor*M500c, 1/(1+z), M500cDef)
        ratio=M500c/testM500c
        scaleFactor=scaleFactor*ratio
        count=count+1
        if count > 10:
            raise Exception("M500c -> massDef conversion didn't converge quickly enough")

    massX=scaleFactor*M500c

    return massX

#------------------------------------------------------------------------------------------------------------
# Main

# Examples
M500c=2e14
z=0.5

# M200m, R200m
M200m=M500cToMdef(M500c, z, M200mDef, fiducialCosmoModel)
R200m=calcRDeltaMpc(z, M200m, fiducialCosmoModel, delta = 200, wrt = 'mean')

# M200c, R200c
M200c=M500cToMdef(M500c, z, M200cDef, fiducialCosmoModel)
R200c=calcRDeltaMpc(z, M200c, fiducialCosmoModel, delta = 200, wrt = 'critical')

print("M500c, z = %.3e MSun, %.3f" % (M500c, z))
print("M200m = %.3e MSun" % (M200m))
print("R200m = %.3f Mpc" % (R200m))
print("M200c = %.3e MSun" % (M200c))
print("R200c = %.3f Mpc" % (R200c))
