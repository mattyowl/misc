"""

eROSITA images grabber

"""

import os
import sys
import numpy as np
from astLib import *
import astropy.table as atpy
import astropy.io.fits as pyfits
import urllib
import IPython

tab=atpy.Table().read("DR5_cluster-catalog_v1.1.fits")
sizeArcmin=8

# Store clips here
os.makedirs('images', exist_ok = True)

# test
tab=tab[tab['redshift'] < 0.5]
tab=tab[tab['redshift'] > 0.4]

tileTab=atpy.Table().read("SKYMAPS_052022_MPE.fits")

count=0
cube=[]
for row in tab:
    count=count+1
    print("... %d/%d ..." % (count, len(tab)))
    mask1=np.logical_and(row['RADeg'] > tileTab['RA_MIN'], row['RADeg'] < tileTab['RA_MAX'])
    mask2=np.logical_and(row['decDeg'] > tileTab['DE_MIN'], row['decDeg'] < tileTab['DE_MAX'])
    maskTileTab=tileTab[np.logical_and(mask1, mask2)]

    if len(maskTileTab) > 0:
        outFileName="images/%s_eRASS.fits.gz" % (row['name'].replace(" ", "_"))
        if os.path.exists(outFileName) == False:
            tileRow=maskTileTab[0]
            # 022 gets 0.6-2.3 keV
            # Some really daft rounding stuff to deal with here
            if tileRow['RA_CEN'] % 1 == 0.5:
                R=int(np.ceil(tileRow['RA_CEN']))
            else:
                R=round(tileRow['RA_CEN'])
            if R < 1:
                RRR='000'
            elif R > 1 and R < 10:
                RRR='00'+str(R)[:1]
            elif R >= 10 and R < 100:
                RRR='0'+str(R)[:2]
            else:
                RRR=str(R)[:3]
            # DDD should be safe as ACT in southern hemisphere - otherwise, need to implement handling for stuff at < 10 deg of N pole...
            if tileRow['DE_CEN'] > 0:
                # DDD=str(90-tileRow['DE_CEN']).replace(".", "")[:3]
                DDD='0'+str(90-round(tileRow['DE_CEN'])).replace(".", "")[:2]
            elif tileRow['DE_CEN'] == 0:
                DDD='090'
            elif tileRow['DE_CEN'] < 0 and tileRow['DE_CEN'] > -10:
                DDD='0'+str(90+abs(tileRow['DE_CEN'])).replace(".", "")[:2]
            else:
                DDD=str(90+abs(tileRow['DE_CEN'])).replace(".", "")[:3]
            if tileRow['OWNER'] == 0:
                ownStr='b' # was 'c'
            elif tileRow['OWNER'] == 1:
                ownStr='c' # was 'b'
            elif tileRow['OWNER'] == 2:
                ownStr='m'
            urlStr="https://erosita.mpe.mpg.de/dr1/erodat/data/download/%s/%s/EXP_010/e%s01_%s_022_Image_c010.fits.gz" % (DDD, RRR, ownStr, RRR+DDD)
            try:
                urllib.request.urlretrieve(urlStr, 'tmp.fits')
            except urllib.error.HTTPError:
                print("404", tileRow['OWNER'], 'Russia?')
                if tileRow['OWNER'] == 0:
                    print("might be German? - skipping anyway")
                # continue
                import IPython
                IPython.embed()
                sys.exit()
            with pyfits.open('tmp.fits') as img:
                d=img[0].data
                wcs=astWCS.WCS(img[0].header, mode = 'pyfits')
            clip=astImages.clipImageSectionWCS(d, wcs, row['RADeg'], row['decDeg'], sizeArcmin/60)
            astImages.saveFITS(outFileName, clip['data'], clip['wcs'])
            cube.append(clip['data'])
        else:
            with pyfits.open(outFileName) as img:
                d=img[0].data
            cube.append(d)

# Stack
cube=np.array(cube)
print("N = %d" % (cube.shape[0]))
stack=np.sum(cube, axis = 0)
astImages.saveFITS("stack.fits", stack, None)

IPython.embed()
sys.exit()


# url=https://erosita.mpe.mpg.de/dr1/erodat/data/download/075/091/EXP_010/em01_091075_021_Image_c010.fits.gz
