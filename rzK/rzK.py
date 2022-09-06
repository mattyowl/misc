#! /usr/bin/env python3

"""

Given a K-band image, make rzK RGB image, with rz images from DECaLS (legacysurvey.org at any rate)
Let's try doing it by reshuffling the grz .png images so that we don't have to do any resampling

"""

import os
import sys
import numpy as np
import astropy.io.fits as pyfits
from astLib import *
from PIL import Image, UnidentifiedImageError
import pylab as plt
import IPython
import urllib3

#------------------------------------------------------------------------------------------------------------
# Global config dictionary - like sourcery config file
configDict={'figSize': [8.25, 7.5],
            'contourSmoothingArcsec': 1.0,
            'contour1Sigma': "measureFromImage",
            # Used for supercluster figs
            #'contourSigmaLevels': [3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            # Used for DES/HSC/DECaLS figs
            #'contourSigmaLevels': [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0],
            # El Gordo MeerKAT (so not too crowded)
            'contourSigmaLevels': [3.0, 5.0, 10.0, 15.0, 20.0],
            # Used for super high SNR
            #'contourSigmaLevels': [15.0, 20.0, 25.0, 30.0],
            'contourColour': 'cyan',
            'contourWidth': 1,
            'gamma': 1.0}

#------------------------------------------------------------------------------------------------------------
def fetchLegacySurveyImage(RADeg, decDeg, sizeArcmin, sizePix, refetch = False, layer = 'ls-dr9',
                           bands = 'grz'):
    """Fetches .jpg cut-out from legacysurvey.org sky viewer. Based on the code in sourcery.

    Valid layers include e.g. decals-dr7, des-dr1 etc..

    """

    decalsCacheDir="jpgCache"
    os.makedirs(decalsCacheDir, exist_ok = True)
    http=urllib3.PoolManager()

    outFileName=decalsCacheDir+os.path.sep+"%s_%.6f_%.6f_%.1f_%s.jpg" % (layer, RADeg, decDeg, sizeArcmin, bands)

    decalsWidth=sizePix
    decalsPixScale=(sizeArcmin*60.0)/float(decalsWidth)
    if os.path.exists(outFileName) == False or refetch == True:
        #http://legacysurvey.org/viewer/jpeg-cutout?ra=52.102810&dec=-21.670020&size=2048&layer=des-dr1&pixscale=0.3809&bands=grz
        urlString="http://legacysurvey.org/viewer/jpeg-cutout?ra=%.6f&dec=%.6f&size=%d&layer=%s&pixscale=%.4f&bands=%s" % (RADeg, decDeg, decalsWidth, layer, decalsPixScale, bands)
        print("... fetching: %s" % (urlString))
        resp=http.request('GET', urlString)
        with open(outFileName, 'wb') as f:
            f.write(resp.data)
            f.close()

    return outFileName

#------------------------------------------------------------------------------------------------------------
def rzKPlot(name, RADeg, decDeg, inJPGPath, KData, KWCS, jpgSizeArcmin, contourImgPath,
            plotNEDObjects = "false", plotSpecObjects = "false", plotSourcePos = "false",
            plotXMatch = "false", plotContours = "false", showAxes = "false", clipSizeArcmin = None,
            plotRedshift = "false", redshift = "none", gamma = 1.0):
    """Makes plot of .jpg image with coordinate axes and NED, SDSS objects overlaid.

    Based on the Sourcery routine.

    """

    # Just in case they are passed as strings (e.g., if direct from the url)
    RADeg=float(RADeg)
    decDeg=float(decDeg)

    # This is only used for scaling the size of plotted points
    if clipSizeArcmin == None:
        sizeDeg=jpgSizeArcmin/60.
    else:
        sizeDeg=float(clipSizeArcmin)/60.

    try:
        im=Image.open(inJPGPath)
    except UnidentifiedImageError:
        print("Server error - outside legacy survey area? Skipping")
        return None

    data=np.array(im)
    data=np.power(data, 1.0/float(gamma))
    try:
        data=np.flipud(data)
        #data=np.fliplr(data)
    except:
        #"... something odd about image (1d?) - aborting ..."
        return None

    R=data[:, :, 0]
    G=data[:, :, 1]
    B=data[:, :, 2]

    # HACK: for ACT maps, with huge pixels, we can get offsets in .jpg relative to original
    # So, if we have a .fits image, load that and use to set centre coords
    #fitsFileName=inJPGPath.replace(".jpg", ".fits")
    #if os.path.exists(fitsFileName) == True:
        #hackWCS=astWCS.WCS(fitsFileName)
        #CRVAL1, CRVAL2=hackWCS.getCentreWCSCoords()
    #else:
        #CRVAL1, CRVAL2=RADeg, decDeg
    # Make a WCS
    CRVAL1, CRVAL2=RADeg, decDeg
    sizeArcmin=jpgSizeArcmin
    xSizeDeg, ySizeDeg=sizeArcmin/60.0, sizeArcmin/60.0
    xSizePix=float(R.shape[1])
    ySizePix=float(R.shape[0])
    xRefPix=xSizePix/2.0
    yRefPix=ySizePix/2.0
    xOutPixScale=xSizeDeg/xSizePix
    yOutPixScale=ySizeDeg/ySizePix
    newHead=pyfits.Header()
    newHead['NAXIS']=2
    newHead['NAXIS1']=xSizePix
    newHead['NAXIS2']=ySizePix
    newHead['CTYPE1']='RA---TAN'
    newHead['CTYPE2']='DEC--TAN'
    newHead['CRVAL1']=CRVAL1
    newHead['CRVAL2']=CRVAL2
    newHead['CRPIX1']=xRefPix+1
    newHead['CRPIX2']=yRefPix+1
    newHead['CDELT1']=-xOutPixScale
    newHead['CDELT2']=xOutPixScale    # Makes more sense to use same pix scale
    newHead['CUNIT1']='DEG'
    newHead['CUNIT2']='DEG'
    wcs=astWCS.WCS(newHead, mode='pyfits')

    KScale=astStats.biweightScale(KData.flatten(), 9)
    d=KData/KScale
    minSigmaStretch=-1
    d[d < minSigmaStretch]=minSigmaStretch
    d[d > 10]=10
    d=d-d.min()+1e-3
    d=np.log(d)
    d=d/d.max()
    d=np.array(d*255, dtype = int)
    R=R[:d.shape[0], :d.shape[1]]
    G=G[:d.shape[0], :d.shape[1]]
    B=B[:d.shape[0], :d.shape[1]]
    Gt=np.zeros(d.shape)+R
    Bt=np.zeros(d.shape)+G
    R=d
    B=Bt
    G=Gt
    R=d
    #G=R
    #B=R
    #print("fiddle")
    #import IPython
    #IPython.embed()
    #sys.exit()

    #cutLevels=[[R.min(), R.max()], [G.min(), G.max()], [B.min(), B.max()]]
    cutLevels=[[0, 255], [0, 255], [0, 255]]

    # Optional zoom
    if clipSizeArcmin != None:
        clipSizeArcmin=float(clipSizeArcmin)
        RClip=astImages.clipImageSectionWCS(R, wcs, RADeg, decDeg, clipSizeArcmin/60.0)
        GClip=astImages.clipImageSectionWCS(G, wcs, RADeg, decDeg, clipSizeArcmin/60.0)
        BClip=astImages.clipImageSectionWCS(B, wcs, RADeg, decDeg, clipSizeArcmin/60.0)
        R=RClip['data']
        G=GClip['data']
        B=BClip['data']
        wcs=RClip['wcs']
    #astImages.saveFITS("test.fits", R, wcs)

    # Make plot
    if showAxes == "true":
        axes=[0.1,0.085,0.9,0.85]
        axesLabels="sexagesimal"
        figSize=configDict['figSize']
    else:
        axes=[0, 0, 1, 1]
        axesLabels="sexagesimal"    # Avoid dealing with axis flips
        figSize=(max(configDict['figSize']), max(configDict['figSize']))
    fig=plt.figure(figsize = figSize)

    p=astPlots.ImagePlot([R, G, B], wcs, cutLevels = cutLevels, title = name.replace("_", " "), axes = axes,
                        axesLabels = axesLabels)

    if showAxes != "true":
        p.addScaleBar('NW', 60, color='yellow', fontSize=16, width=2.0, label = "1'")
        plt.figtext(0.025, 0.98, name.replace("_", " "), ha = 'left', va = 'top', size = 24, color = 'yellow')
        #if plotTitle != None:
        #plt.figtext(0.965, 0.88, plotTitle, ha = 'right', size = 24)

    if plotSourcePos == "true":
        p.addPlotObjects([RADeg], [decDeg], 'clusterPos', symbol='cross', size=sizeDeg/20.0*3600.0, color='white')

    if plotNEDObjects == "true":
        # We should already have the files for this from doing addNEDInfo earlier
        nedFileName=self.nedDir+os.path.sep+name.replace(" ", "_")+".txt"
        nedObjs=catalogTools.parseNEDResult(nedFileName, onlyObjTypes = configDict['NEDObjTypes'])
        if len(nedObjs['RAs']) > 0:
            p.addPlotObjects(nedObjs['RAs'], nedObjs['decs'], 'nedObjects', objLabels = nedObjs['labels'],
                                size = sizeDeg/40.0*3600.0, color = "#7cfc00")

    if plotSpecObjects == "true":
        specRedshifts=catalogTools.fetchSpecRedshifts(name, RADeg, decDeg,
                                                        redshiftsTable = self.specRedshiftsTab)
        if specRedshifts is not None:
            specRAs=[]
            specDecs=[]
            specLabels=[]
            specCount=0
            for specObj in specRedshifts:
                specCount=specCount+1
                specRAs.append(specObj['RADeg'])
                specDecs.append(specObj['decDeg'])
                specLabels.append(str(specCount))
            if len(specRAs) > 0:
                p.addPlotObjects(specRAs, specDecs, 'specObjects', objLabels = specLabels,
                                size = sizeDeg/40.0*3600.0, symbol = 'box', color = "red")

    if plotXMatch == "true":
        obj=self.sourceCollection.find_one({'name': name})
        xMatchRAs=[]
        xMatchDecs=[]
        xMatchLabels=[]
        if 'crossMatchCatalogs' in configDict.keys():
            for xMatchDict in configDict['crossMatchCatalogs']:
                if "plotXMatch" in xMatchDict.keys() and xMatchDict['plotXMatch'] == False:
                    continue
                label=xMatchDict['label']
                RAKey='%s_RADeg' % (label)
                decKey='%s_decDeg' % (label)
                if RAKey in obj.keys() and decKey in obj.keys():
                    # We only want to show coords that are different, and not exactly cross-matched
                    # (e.g., cross matched zCluster results would have exact same RADeg, decDeg - useless to show)
                    if obj[RAKey] != obj['RADeg'] and obj[decKey] != obj['decDeg']:
                        xMatchRAs.append(obj[RAKey])
                        xMatchDecs.append(obj[decKey])
                        xMatchLabels.append(label)
        # Other coords in obj dictionary that weren't yet picked up (useful for e.g. editable BCG coords)
        # Editable fields
        fieldsList=[]
        for fieldDict in configDict['fields']:
            fieldsList.append(str(fieldDict['name']))
        for key in fieldsList:
            if key.split("_")[-1] == 'RADeg':
                if key.split("_")[0] not in xMatchLabels and key != "RADeg":
                    xMatchRAs.append(obj[key])
                    xMatchDecs.append(obj[key.replace("RADeg", "decDeg")])
                    xMatchLabels.append(str(key.split("_")[0]))

        if len(xMatchRAs) > 0:
            p.addPlotObjects(xMatchRAs, xMatchDecs, 'xMatchObjects', objLabels = xMatchLabels,
                                size = sizeDeg/40.0*3600.0, symbol = "diamond", color = 'cyan')

    if plotRedshift == "true" and redshift != "none":
            plt.figtext(0.025, 0.03, "z = %.2f" % (float(redshift)), ha = 'left', size = 24, color = 'white')

    if plotContours == "true" and contourImgPath != "none":
        clipFileName=contourImgPath
        if os.path.exists(clipFileName):
            # NOTE: Adapted for MeerKAT (and measure RMS on fly)
            #--
            with pyfits.open(clipFileName) as img:
                shape=img[0].data[0, 0].shape
                contourWCS=astWCS.WCS(img[0].header, mode = 'pyfits').copy()
                contourData=img[0].data[0, 0]
            if configDict['contour1Sigma'] == "measureFromImage":
                # Choose level from clipped stdev
                sigmaCut=3.0
                mean=0
                sigma=1e6
                for i in range(20):
                    #nonZeroMask=np.not_equal(contourData, 0)
                    mask=np.less(abs(contourData-mean), sigmaCut*sigma)
                    #mask=np.logical_and(nonZeroMask, mask)
                    mean=np.mean(contourData[mask])
                    sigma=np.std(contourData[mask])
            else:
                sigma=configDict['contour1Sigma']
            contourSigmaLevels=np.array(configDict['contourSigmaLevels'])
            contourLevels=contourSigmaLevels*sigma
            #contourLevels=[configDict['contour1Sigma'], 2*configDict['contour1Sigma'],
                            #4*configDict['contour1Sigma'], 8*configDict['contour1Sigma'],
                            #16*configDict['contour1Sigma']]
            #contourLevels=np.linspace(configDict['contour1Sigma'],
                                        #20*configDict['contour1Sigma'], 20)
            p.addContourOverlay(contourData, contourWCS, 'contour', levels = contourLevels,
                                width = configDict['contourWidth'],
                                color = configDict['contourColour'],
                                smooth = configDict['contourSmoothingArcsec'],
                                highAccuracy = False)
        else:
            plt.figtext(0.05, 0.05, "Adding contours failed - missing file: %s" % (clipFileName), color = 'red', backgroundcolor = 'black')

    return p

#------------------------------------------------------------------------------------------------------------
# Main
if len(sys.argv) < 2:
    print("Run: rzK.py <Ks-band-image-1.fits> ... <Ks-band-image-N.fits>")
    print("Output will be written to 'rzK' directory")
    sys.exit()

os.makedirs("rzK", exist_ok = True)

for inFileName in sys.argv[1:]:

    print("Making rzK image for %s" % (inFileName))

    # Parse Ks-band image to figure out size, coords etc.
    with pyfits.open(inFileName) as img:
        d=img[0].data
        wcs=astWCS.WCS(img[0].header, mode = 'pyfits')
    RADeg, decDeg=wcs.getCentreWCSCoords()
    clip=astImages.clipImageSectionWCS(d, wcs, RADeg, decDeg, 5.3/60)
    d=clip['data']
    wcs=clip['wcs']
    pixSizeDeg=wcs.getPixelSizeDeg()
    sizeArcmin=np.mean(wcs.getFullSizeSkyDeg())*60
    sizePix=max(d.shape)

    jpgFileName=fetchLegacySurveyImage(RADeg, decDeg, sizeArcmin, sizePix, refetch = False)
    p=rzKPlot(img[0].header['OBJNAME'], RADeg, decDeg, jpgFileName, d, wcs, sizeArcmin, "none")
    if p is not None:
        #if catalogPath != "none":
            #tab=atpy.Table().read(catalogPath)
            #objRAs=tab['RADeg'].data
            #objDecs=tab['decDeg'].data
            #tag='cat-objects'
            #color='cyan'
            #p.addPlotObjects(objRAs, objDecs, tag, symbol='circle', size=8.0, width=0.5, color=color,
                            #objLabels=None, objLabelSize=12.0)
        plt.savefig("rzK"+os.path.sep+"%s.png" % (img[0].header['OBJNAME'].replace(" ", "_")), dpi = 192)
        plt.close()
