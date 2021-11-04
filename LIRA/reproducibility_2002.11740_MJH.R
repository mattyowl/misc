library('lira')

# Modified by MJH - using .tsv downloaded from Vizier
tab <- read.table('XMM-PSZ_Lovisari2020.tsv', header = TRUE, sep = "\t", dec = ".")

# LIRA run by MJH
# L-T relation
C1 <- 5.0   # units in catalog are 1e44 erg/s
C2 <- 5.0   # units in catalog are keV
x <- log10(tab$kT/C2)
y <- log10(tab$LX/C1)
xerr <- tab$e_kT/(tab$kT*log(10))
yerr <- tab$e_LX/(tab$LX*log(10))
z <- tab$z

# Corresponds to line 3 of LIRA LX-T fits in Table 4
# Fixed gamma = 1; fitting for beta only (no sigmaX|Z)
mcmc <- lira (x, y, delta.x = xerr, delta.y = yerr, z = z, z.ref = 0.2,
         gamma.YIZ = 1.0, gamma.mu.Z.Fz = 0.0, gamma.sigma.Z.D = "dt", n.chains = 4, n.adapt = 2e3,
         n.iter = 2e4, export = TRUE, export.mcmc = 'LX-T_line3.dat')

# Corresponds to line 1 of LIRA LX-T fits in Table 4
# Gamma, beta, sigmaX|Z all free to vary
mcmc <- lira (x, y, delta.x = xerr, delta.y = yerr, z = z, z.ref = 0.2,
              sigma.XIZ.0 = "prec.dgamma", gamma.mu.Z.Fz = 0.0, gamma.sigma.Z.D = "dt",
              n.chains = 4, n.adapt = 2e3, n.iter = 2e4, export = TRUE, export.mcmc = 'LX-T_line1.dat')
