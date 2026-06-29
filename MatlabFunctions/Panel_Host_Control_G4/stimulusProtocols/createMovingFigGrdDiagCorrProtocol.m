function protocolStruct = createMovingFigGrdDiagCorrProtocol(inputStruct)

% function createMovingFigGrdDiagCorrProtocol(inputStruct)
%
% This function uses the inputStruct and createProtocol function to
% generate single bar moving through the window . Bar has two components:
% figure bar and ground bar (which will be presented leading and lagging in the desired orientations)
% It has certain assumptions and therefore requires
% less inputs.
% If function is called with no inputStruct, it prompt the user for
% relevant input and allows to change the default
%
% ASSUMPTIONS
% Randomize -       everything is randomized (so that if several gratings appear
%                   they appear in all positions and in all oreintations.
% Masks positions - [1,1] Grid is assumed.
% gratingFuncHand - Function uses the generateBarFrameByInd to make sure
%                   even in diagonal the coverage is complete
% maskType -        Due to the way the bar is generated, mask type is
%                   always rectangle.
% grtMaskInt -      one to one grating and mask. Set to 1. Since mask need to change orientation also
% gridSize -        limited to a [1,1] grid
%
%   NOTE!!! to correct small problems with diagonal line orientation is
%   implemented here and not in createProtocol
%
% INPUT
% Defaults are dilimited with {} and are optional inputs.
% Only 2 obligatory fields are width and gridCenter
%
% inputStruct -     Should have the following fields
% .figBarWid -      single number. Width of figure bar in pixels
% .grdBarWid -      same as above for ground bar
%
%             Note!! function uses generate2BarsFrameByInds and ground bar
%             will be defined as first bar (i.e figBar is plotted on top)
%
%             Note!! fig bar is plotted with every possible overlap with
%             ground bar. e.g if grdBarW==2 and figBarW==1, resulting bar
%             will be with 1 column of figVal and 1 col of grdVal, with
%             figVal either leading or lagging.
%
% .fig/grdBarVal-   1XF/1XG vectors of normalized luminance values between
%                   0-1. { 1 / 0 }
% .figGrdInt -      logical. if TRUE creates all combinations of fig and
%                   grd values. if FALSE fig and grd vals should be the same length
% .barHeight -      single number. height of bar in pixels.
% .barSpan -        single number. span along which bar will move (in
%                   pixels). Will be converted into the width of the
%                   rectangular mask
% .orientation -    Vector (0:7).
%                   Orientations for the gratings. Applied on all inputs.
% .stepDur -        1XT vector. duration in seconds in which the bar will appear
% .gsLevel -        gray scale level ( fixed at 4 )
% .gratingMidVal -  value of the rest of the window (0.49 - bkgd level)
% .maskPositions -  User can specify these directly as an NX2 matrix, or
%                   use the other parameters to generate them (if this is
%                   given other parameters are disregarded
% .gridCenter -     1X2 vector specifying the center of the grid in X and Y
%                   (sptial coordinates in pixels <for an 8X4 arena its 96X32). If one dimension of
%                   grid is even, grid will be presented around center but
%                   will not have a position in the actual center.
% .intFrames -      number of empty intervening frames. If not given half a
%                   second worth (based on generalFrequency)
% .repeats -        scalar. number of times the whole protocol repeats (passed into createProtocol) {3}
% .generalFrequency-Frequency with which frames from the generated protocol
%                   will be dumped (passed on to runDumpProtocol) in position function units
%                   (frames per second on the controller). fixed at 500 for gsLevel 4
% .freqCorrFlag -   Also passed on to runDumpProtocol. Logical flag to
%                   indicate whether different stimuli should be run with temporal frequency
%                   correction { 1 }.
%
% OUTPUT
% protocolStruct with all the required fields from createProtocl.
%
%
%                   NOTE! masks and gratings need not be of the same length
%                   NOTE! grid mask positions will use only the first
%                   maskRadius value to interpert overlap.

%% GENERAL AND DEFAULT PARAMETERS

baseSiz = 445; % size of single frame or mask
arenaSize = [192,48];
gratingFuncHand = @generate2BarsFrameByInds;

default.figBarWid = 1;
default.grdBarWid = 3;
default.figBarVal = 1;
default.grdBarVal = 0;
default.figGrdInt = 0;
default.barHeight = 9;
default.barSpan = 11;
default.gridCenter = 'UI';
default.contrast = 1;
default.gratingMidVal = 0.49;
default.orientations = 'UI';
default.stepDur = 0.04;
default.gridSize = [1,1];
default.gridOverlap = 0;
default.intFrames = nan;
default.repeats = 3;
default.randomize = 1;

fixed.gsLevel = 4;
fixed.generalFrequency = 500;
fixed.maskType = {'rectangle'};
fixed.freqCorrFlag = 0;
fixed.grtMaskInt = 1;
fixed.gridSize = [1,1];
fixed.gridOverlap = 0;

% combining default and input structures
if nargin == 0
    default = modifyDefaultStruct(default);
else
    default = modifyDefaultStruct(default, inputStruct);
end


 %% ORIENTATIONS

 ort = default.orientations;
 assert(isvector(ort), 'Orientation should be 1XM vector')
 assert(all(ismember(ort, 0:7)), 'Orientation values should be between 0 and 7')

 newOrt = ort;


 % orientation is implemented internally
 protocolStruct.orientations = 0;

  %% MASK (masks created with grating)

 maskHW = floor(default.barSpan/2); % rectangle mask input is half width
 assert(isvector(maskHW) && length(maskHW) == 1, 'barSpan should be a single number')
 assert(maskHW > 0, 'barSpan should be positive')


 minMaskR = maskHW;

 maskHH = floor(default.barHeight/2);
 assert(isvector(maskHH) && length(maskHH) == 1, 'barHeight should be a single number');
 assert(maskHH > 0, 'barHeight should be a positive number')

 maskT = fixed.maskType;

 relRegR = maskHW;
 relDiagR = round((2*relRegR+1)/sqrt(2)); % was +1 (with -1 overlap with non-rotated square is too small);
 radCell = {relRegR, relDiagR};


%% GRATING PARAMETERS

% needed to determine number of frames to appear
protocolStruct.generalFrequency = fixed.generalFrequency;

stepLen = default.stepDur;
assert(isvector(stepLen), 'stepDur should be one a vector')

stepFrames = unique(round(stepLen * fixed.generalFrequency));
if length(stepFrames) < length(stepLen)
    warning('%d step durations omitted since were the same after rounding', length(stepLen) - length(stepFrames))
end

assert(min(stepFrames) > 0, 'stimulus can not be presented for such a short duration. Minimal duration is 20ms')

gsLev = fixed.gsLevel;

bkgdVal = default.gratingMidVal;
assert(length(bkgdVal) == 1, 'gratingMidVal should be a single number');
assert(bkgdVal >=0 && bkgdVal <= 1, 'gratingMidVal should be between 0 and 1')

figV = default.figBarVal;
assert(min(figV) >=0 && max(figV) <= 1, 'figBarVal should be between 0 and 1');
assert(isvector(figV), 'figBarVal should be a 1XF vector')

grdV = default.grdBarVal;
assert(min(grdV) >=0 && max(grdV) <= 1, 'grdBarVal should be between 0 and 1');
assert(isvector(grdV), 'grdBarVal should be a 1XG vector')

fgInt = default.figGrdInt;
assert(ismember(fgInt, [0,1]), 'figGrdInt should be logical')

fgC = 0;
if fgInt
    for ii=1:length(figV)
        for jj=1:length(grdV)
            fgC = fgC+1;
            tempFV(fgC) = figV(ii);
            tempGV(fgC) = grdV(jj);
        end
    end
    figV = tempFV;
    grdV = tempGV;
else
    assert(length(figV) == length(grdV), 'when figGrdInt is FALSE, fig and grd values should be the same length')
end



figW = default.figBarWid;
assert(isvector(figW) && length(figW) ==1 , 'figBarW should be a single number')
assert(figW > 0, 'figBarW should be a positive number')

grdW = default.grdBarWid;
assert(isvector(grdW) && length(grdW) ==1 , 'grdBarW should be a single number')
assert(grdW > 0, 'grdBarW should be a positive number')

assert(figW < grdW, 'grdBarWid should be bigger than figBarWid - since they are overlaid')
widDiff = 1+grdW-figW;

count = 0;
gratingArray = [];

for vv=1:length(figV)

    for ww=1:widDiff

        for tt=1:length(stepFrames)

            for oo = 1:length(newOrt)

                count = count+1;

                ortPosInd = rem(newOrt(oo),2) +1;
                relRad = radCell{ortPosInd};

                relPos = -relRad:relRad+grdW-1;
                corrPos = reshape(repmat(relPos, stepFrames(tt), 1), 1, []);

                gtStruct(count).fWid = grdW;
                gtStruct(count).sWid = figW;
                gtStruct(count).fVal = grdV(vv);
                gtStruct(count).sVal = figV(vv);
                gtStruct(count).ori = newOrt(oo);
                %gtStruct(count).sqDim = max(2*maskHW+1, 2*maskHH+1); % generateBarFrameByInds corrects for diagonal internally
                gtStruct(count).sqDim = 2*maskHW+1; % since when using divideTotSquareToCols height is not taken into account

                gtStruct(count).fPos = corrPos;
                gtStruct(count).sPos = corrPos - (ww-1);
                gtStruct(count).gsLevel = gsLev;
                gtStruct(count).bkgdVal = bkgdVal;
                gtStruct(count).matSize = baseSiz;

                maskSt(count).type = maskT{1};
                maskSt(count).radius = [relRad, maskHH];
                maskSt(count).ori = newOrt(oo);

                gratingArray = vertcat(gratingArray, ...
                                       [count, grdV(vv), figV(vv), stepFrames(tt)/fixed.generalFrequency, ...
                                        ww-1, newOrt(oo)]);

            end
        end
    end
end


 tabVarNames =  {'index', 'groundVal', 'figureVal', 'stepDur', 'posDiff', 'orient'};
 gratTable = array2table(gratingArray, 'variablenames', tabVarNames);
 gratTable.Properties.Description = ['span:', num2str(2*maskHW+1), ' ', 'grdWid:', num2str(grdW), ...
                                     ' ', 'figWid:', num2str(figW), ' ', 'Height:', num2str(2*maskHH+1)];

 protocolStruct.gratingTable = gratTable;
 protocolStruct.gratingStruct = gtStruct;
 protocolStruct.masksStruct = maskSt;

 %% GRID

 gridSt.gridSize = fixed.gridSize;
 ovlp = fixed.gridOverlap;

 maskS = 2*minMaskR+1;
 space = maskS - maskS*ovlp;
 gridSt.spacing = [space, space];

 gdCen = default.gridCenter;
 assert(isvector(gdCen), 'gridCenter should be a 1X2 vector')
 assert(length(gdCen)==2, 'gridCenter should be a 1X2 vector')

 stCrds = gdCen - space*(gridSt.gridSize-1)/2;
 if stCrds(1) < 1
    warning('Grid start position in X is out of range - changed to 1')
    stCrds(1) = 1;
 end
 if stCrds(2) < 1
    warning('Grid start position in Y is out of range - changed to 1')
    stCrds(2) = 1;
 end

 gridSt.startPos = stCrds;

 maskPos = makeGrid(gridSt);

 protocolStruct.maskPositions = maskPos;

 %% Misc parameters

 protocolStruct.freqCorrFlag = fixed.freqCorrFlag;

 protocolStruct.funcHand = gratingFuncHand;
 protocolStruct.interleave = fixed.grtMaskInt;

 intF = default.intFrames;
 if isnan(intF)
     protocolStruct.intFrames = floor(fixed.generalFrequency/2.5);
 else % if user gave a number
    assert(intF >= 0, 'intFrames should be a non-negative number')
    protocolStruct.intFrames = intF;
 end

 protocolStruct.repeats = default.repeats;

 protocolStruct.randomize.gratingSeq = default.randomize;
 protocolStruct.randomize.masks = default.randomize;
 protocolStruct.randomize.orientations = default.randomize;
 protocolStruct.randomize.maskPositions = default.randomize;

 %% Creating protocl


 protocolStruct = createProtocol(protocolStruct);

 protocolStruct.inputParams = default;


end
