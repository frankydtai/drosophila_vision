function protocolStruct = createGratingDiagCorrDiffPhaseProtocol(inputStruct)

% function createGratingDiagCorrDiffPhaseProtocol(inputStruct)
%
% This function uses the inputStruct and createProtocol function to
% generate a stationary grating that will appear in all the phases. It has certain assumptions and therefore requires
% less inputs. 
% If function is called with no inputStruct, it prompt the user for
% relevant input and allows to change the default
% 
% ASSUMPTIONS
% Randomize -       everything is randomized (so that if several gratings appear
%                   they appear in all positions and in all oreintations. 
% Masks positions - [1,1] Grid is assumed.
% gratingFuncHand - Function uses the generateGrstingFrameByInds to make sure
%                   even in diagonal the coverage is complete
% maskType -        Due to the way the bar is generated, mask type is
%                   always rectangle. 
% grtMaskInt -      one to one grating and mask. Set to 1. Since mask need to change orientation also        
%
%   NOTE!!! to correct small problems with diagonal line orientation is
%   implemented here and not in createProtocol 
%
% INPUT 
% Defaults are dilimited with {} and are optional inputs. 
% Only 2 obligatory fields are width and gridCenter
%
% inputStruct -     Should have the following fields
% .f/sBarV -        1XN vector. normalized luminance value for first and second bars in the grating. { 1 / 0  }
%                   Note!! if length for fBar and sBar should be identical
% .winHeight -      1XM vector. height of grating window in pixels.
% .winWidth -       single number. width of grating window (in
%                   pixels). Will be converted into the width of the
%                   rectangular mask
% .grtHWidth -      1XW vector(in pixels). Width of a bar in the grating,
%                   effectively half the cycle { 4 }.
% .orientation -    single number (0-3). Since the bar isn't moving 4-7 are redundant. 
%                   Orientations for the gratings. Applied on all inputs.
% .stimDur -        1XN vector. Duration in seconds in which the grating will appear 
% .gsLevel -        gray scale level ( 3 ) 
% .gratingMidVal -  value of the rest of the window (0.49 - bkgd level)
% .gridCenter -     1X2 vector specifying the center of the grid in X and Y
%                   (sptial coordinates in pixels <for an 8X4 arena its 96X32). If one dimension of
%                   grid is even, grid will be presented around center but
%                   will not have a position in the actual center.
% .intFrames -      number of empty intervening frames. If not given half a
%                   second worth (based on generalFrequency)
% .repeats -        scalar. number of times the whole protocol repeats (passed into createProtocol) {3}
% .generalFrequency-Frequency with which frames from the generated protocol
%                   will be dumped (passed on to runDumpProtocol) in position function units 
%                   (frames per second on the controller). 
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

baseSiz = 225; % size of single frame or mask
arenaSize = [96,32];
gratingFuncHand = @generateGratingFrameByInds;

default.fBarV = [0.49, 1, 1];
default.sBarV = [0, 0, 0.49];
default.winHeight = 9;
default.winWidth = 9;
default.grtHWidth = 4;
default.gridCenter = 'UI';
default.gsLevel = 3;
default.gratingMidVal = 0.49;
default.orientations = 'UI';
default.stimDur = [0.02, 0.08]; % [0.02, 0.04, 0.08, 0.16]; 
default.intFrames = nan;
default.repeats = 3;
default.randomize = 1;

fixed.gridSize = [1,1];
fixed.gridOverlap = 0;
fixed.generalFrequency = 50;
fixed.maskType = {'rectangle'};
fixed.freqCorrFlag = 0;
fixed.grtMaskInt = 1;  
 
% combining default and input structures
if nargin == 0
    default = modifyDefaultStruct(default);
else
    default = modifyDefaultStruct(default, inputStruct);
end


 %% ORIENTATIONS
 
 ort = default.orientations;
 assert(length(ort) == 1, 'Orientation should be single number')
 assert(ismember(ort, 0:3), 'Orientation values should be between 0 and 3')
 
 newOrt = ort; %(ismember(ort, 0:3));
 
 
 % orientation is implemented internally
 protocolStruct.orientations = 0;
 
 


  %% MASK (masks created with grating)
 
 maskHW = floor(default.winWidth/2); % rectangle mask input is half width
 assert(isvector(maskHW), 'winWidth should be a single number')
 assert(length(maskHW) == 1, 'winWidth should be a single number')
 assert(maskHW > 1, 'winWidth should be a positive number')
 
 maskHH = floor(default.winHeight/2);
 assert(isvector(maskHH), 'winHeight should be a 1XM vector');
 assert(min(maskHH) > 0, 'winHeight minimum should be a positive number')
 
 maskT = fixed.maskType;

%% GRATING PARAMETERS

% needed to determine number of frames to appear
protocolStruct.generalFrequency = fixed.generalFrequency;

stimLen = sort(default.stimDur); 
assert(isvector(stimLen), 'stimDur should be one a vector')

stimFrames = unique(round(stimLen * fixed.generalFrequency));
if length(stimFrames) < length(stimLen)
    warning('%d step durations omitted since were the same after rounding', length(stimLen) - length(stimFrames))
end
 
assert(min(stimFrames) > 0, 'stimulus can not be presented for such a short duration. Minimal duration is 20ms')

gsLev = default.gsLevel;
assert(ismember(gsLev, 1:4), 'gsLevel should be an integer between 1 and 4')

bkgdVal = default.gratingMidVal;
assert(length(bkgdVal) == 1, 'gratingMidVal should be a single number');
assert(bkgdVal >=0 && bkgdVal <= 1, 'gratingMidVal should be between 0 and 1')

stimFB = default.fBarV;
assert(isvector(stimFB), 'fBarV should be a 1XN vector')
assert(min(stimFB) >=0 && max(stimFB) <= 1, 'fBarV should be between 0 and 1');

stimSB = default.sBarV;
assert(isvector(stimSB), 'sBarV should be a 1XN vector')
assert(min(stimSB) >=0 && max(stimSB) <= 1, 'sBarV should be between 0 and 1');

assert(length(stimSB) == length(stimFB), 'fBarV and sBarV should have the same length')

barW = default.grtHWidth;
assert(isvector(barW), 'barWidth should be a vector');
assert(all(bsxfun(@eq, barW, round(barW))), 'barWidth should use only integers');

relRegR = maskHW;
relDiagR = round((2*relRegR+1)/sqrt(2)); % was +1 (with -1 overlap with non-rotated square is too small);

radVec = [relRegR, relDiagR];

ortPosInd = rem(newOrt,2) +1;
relRad = radVec(ortPosInd);

sqDim = 2*maskHW+1;

count = 0;
gratingArray = [];

for ss=1:length(stimFB)
        
    for hh=1:length(maskHH)

        for kk=1:length(stimFrames)

            for ww=1:length(barW)

                relPhase = 1:2*barW(ww);

                for pp=1:length(relPhase)


                    count = count+1;
                    gtStruct(count).wid = barW(ww);
                    gtStruct(count).ori = newOrt;
                    gtStruct(count).fVal = stimFB(ss);
                    gtStruct(count).sVal = stimSB(ss);
                    gtStruct(count).sqDim = sqDim; % span is a bit unique in its claculation here
                    gtStruct(count).phase = ones(1, stimFrames(kk))* relPhase(pp);
                    gtStruct(count).gsLevel = gsLev; 
                    gtStruct(count).bkgdVal = bkgdVal;
                    gtStruct(count).matSize = baseSiz;

                    maskSt(count).type = maskT{1};
                    maskSt(count).radius = [relRad, maskHH(hh)];
                    maskSt(count).ori = newOrt;

                    gratingArray = vertcat(gratingArray, ...
                                           [count, stimFB(ss), stimSB(ss), barW(ww), newOrt, sqDim, ...
                                            stimFrames(kk)/fixed.generalFrequency, 2*maskHH(hh)+1, relPhase(pp)]); 

                end
            end
        end  
    end
end


 tabVarNames =  {'index','FBval', 'SBVal', 'width', 'orient', 'span', 'stimDur', 'height','phase'};
 
 protocolStruct.gratingTable = array2table(gratingArray, 'variablenames', tabVarNames);
 protocolStruct.gratingStruct = gtStruct;
 protocolStruct.masksStruct = maskSt;
 
 %% GRID 

% since the grid is just one position 

gdCen = default.gridCenter;
assert(isvector(gdCen), 'gridCenter should be a 1X2 vector')
assert(length(gdCen)==2, 'gridCenter should be a 1X2 vector')
     
gridSt.gridSize = fixed.gridSize;
fixed.gridOverlap;
gridSt.spacing = [maskHW(1), maskHW(1)];
gridSt.startPos = gdCen;

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










