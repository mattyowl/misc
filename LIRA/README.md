Matt's attempt at reproducing the LX-T result in Lovisari et al. (2020)
-----------------------------------------------------------------------

Here we attempt to reproduce results from

    https://ui.adsabs.harvard.edu/abs/2020ApJ...892..102L/abstract

using LIRA.

Run everything with:

    sh RUN_ALL.sh

Or see below for more info.


Data
----

These files contain the same info, as downloaded from Vizier:

    XMM-PSZ_Lovisari2020.fits
    XMM-PSZ_Lovisari2020.tsv

The FITS table is easier to use with Python, while the tab-separated-values
plain-text file is easier to use with R.


R scripts
---------

This is Matt's version, based on the script from Phumlani:

    reproducibility_2002.11740_MJH.R

This can be run directly from the command line using:

    R CMD BATCH reproducibility_2002.11740_MJH.R


Post processing
---------------

Rather than use R, post processing to get marginalised parameter constraints
is done in Python using the GetDist package.

Run:

    python3 postLIRAProcessing.py

Parameter values and uncertainties are found in, e.g., `LX-T_line1_chains.margestats`.
This code also produces a scaling relation plot called, e.g., `scalingRelationPlot_LX-T_line1.png`

Only the LX-T relation (line 1 in the LIRA LX-T fits in Table 4 of Lovisari
et al. 2020) is implemented at the moment, but all parameter values agree with
the ones quoted in the paper.
