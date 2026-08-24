# Avizo Script
remove -all
remove volrenRed.col typical_brain_female.am Voltex GlobalAxis

# Create viewers
viewer setVertical 0

viewer 0 setBackgroundMode 1
viewer 0 setBackgroundColor 0.72 0.72 0.78
viewer 0 setBackgroundColor2 0.76 0.73 0.74
viewer 0 setTransparencyType 5
viewer 0 setAutoRedraw 0
viewer 0 show
mainWindow show

set hideNewModules 0
[ load ${SCRIPTDIR}/standard_model_f-files/volrenRed.col ] setLabel volrenRed.col
volrenRed.col setIconPosition 20 40
volrenRed.col setNoRemoveAll 1
volrenRed.col fire
{volrenRed.col} setMinMax 10 180
volrenRed.col flags setValue 1
volrenRed.col shift setMinMax -1 1
volrenRed.col shift setButtons 0
volrenRed.col shift setIncrement 0.133333
volrenRed.col shift setValue 0
volrenRed.col shift setSubMinMax -1 1
volrenRed.col scale setMinMax 0 1
volrenRed.col scale setButtons 0
volrenRed.col scale setIncrement 0.1
volrenRed.col scale setValue 1
volrenRed.col scale setSubMinMax 0 1
volrenRed.col fire
volrenRed.col setViewerMask 65535

set hideNewModules 0
[ load ${SCRIPTDIR}/standard_model_f-files/typical_brain_female.am ] setLabel typical_brain_female.am
typical_brain_female.am setIconPosition 20 10
typical_brain_female.am sharedColormap setDefaultColor 0.8 0.8 0.8
typical_brain_female.am sharedColormap setDefaultAlpha 0.500000
typical_brain_female.am fire
typical_brain_female.am setViewerMask 65535

set hideNewModules 0
create HxVoltex {Voltex}
Voltex setIconPosition 318 10
Voltex data connect typical_brain_female.am
Voltex colormap setDefaultColor 1 0.8 0.5
Voltex colormap setDefaultAlpha 0.500000
Voltex colormap connect volrenRed.col
Voltex fire
Voltex options setValue 0 0
Voltex options setToggleVisible 0 1
Voltex options setValue 1 1
Voltex options setToggleVisible 1 1
Voltex range setMinMax 0 -3.40282346638529e+38 3.40282346638529e+38
Voltex range setValue 0 0
Voltex range setMinMax 1 -3.40282346638529e+38 3.40282346638529e+38
Voltex range setValue 1 255
Voltex lookup setValue 2
Voltex gamma setMinMax 1 8
Voltex gamma setButtons 0
Voltex gamma setIncrement 0.466667
Voltex gamma setValue 3
Voltex gamma setSubMinMax 1 8
Voltex alphaScale setMinMax 0 1
Voltex alphaScale setButtons 0
Voltex alphaScale setIncrement 0.1
Voltex alphaScale setValue 1
Voltex alphaScale setSubMinMax 0 1
Voltex texture2D3D setValue 0
Voltex slices setMinMax 0 512
Voltex slices setButtons 1
Voltex slices setIncrement 1
Voltex slices setValue 25
Voltex slices setSubMinMax 0 512
Voltex downsample setMinMax 0 1 100
Voltex downsample setValue 0 1
Voltex downsample setMinMax 1 1 100
Voltex downsample setValue 1 1
Voltex downsample setMinMax 2 1 100
Voltex downsample setValue 2 1
Voltex doIt hit
Voltex fire
Voltex setViewerMask 65535

set hideNewModules 0
create HxAxis {GlobalAxis}
GlobalAxis setIconPosition 305 40
GlobalAxis fire
GlobalAxis axis setValue 0 1
GlobalAxis axis setToggleVisible 0 1
GlobalAxis axis setValue 1 1
GlobalAxis axis setToggleVisible 1 1
GlobalAxis axis setValue 2 1
GlobalAxis axis setToggleVisible 2 1
GlobalAxis options setValue 0 1
GlobalAxis options setToggleVisible 0 1
GlobalAxis options setValue 1 0
GlobalAxis options setToggleVisible 1 1
GlobalAxis options setValue 2 0
GlobalAxis options setToggleVisible 2 1
GlobalAxis thickness setMinMax 1 15
GlobalAxis thickness setButtons 0
GlobalAxis thickness setIncrement 0.933333
GlobalAxis thickness setValue 5
GlobalAxis thickness setSubMinMax 1 15
GlobalAxis color setColor 0 1 0 0
GlobalAxis color setColor 1 0 1 0
GlobalAxis color setColor 2 0 0 1
GlobalAxis color setColor 3 1 0.8 0.5
GlobalAxis axisNames setState text 0 x text 1 y text 2 z 
GlobalAxis font setState name: {Helvetica} size: 10 bold: 0 italic: 0 color: 0.8 0.8 0.8
GlobalAxis fire
GlobalAxis setBoundingBox 0 1064.66 0 1064.66 0 1064.66
{GlobalAxis} setDelta 0 
{GlobalAxis} setLocalMode 0 
{GlobalAxis} setFlip 0 0 
{GlobalAxis} setFlip 1 0 
{GlobalAxis} setFlip 2 0 
GlobalAxis fire
GlobalAxis setViewerMask 65534

set hideNewModules 0


viewer 0 setCameraOrientation -0.997307 0.0630274 0.0374941 0.304835
viewer 0 setCameraPosition 15.7108 151.253 738.587
viewer 0 setCameraFocalDistance 827.812
viewer 0 setCameraNearDistance 136.33
viewer 0 setCameraFarDistance 1439.59
viewer 0 setCameraType perspective
viewer 0 setCameraHeightAngle 44.9023
viewer 0 setAutoRedraw 1
viewer 0 redraw

