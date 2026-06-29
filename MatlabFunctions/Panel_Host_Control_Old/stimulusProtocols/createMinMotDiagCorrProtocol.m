function protocolStruct = createMinMotDiagCorrProtocol(inputStruct)

% function createMinMotDiagCorrProtocol(inputStruct)
%
% This function uses the inputStruct and createProtocol function to
% generate minimal motion stimulus. It has certain assumptions and therefore requires
% less inputs. 
% If function is called with no inputStruct, it prompt the user for
% relevant input and allows to change the default
% 
% ASSUMPTIONS
% Randomize -       everything is randomized (so that if several gratings appear
%                   they appear in all positions and in all oreintations. 
% Masks positions - [1,1] Grid is assumed.
% gratingFuncHand - Function uses the generate2BarsFrameByInd to make sure
%                   even in diagonal the coverage is complete
% maskType -        Due to the way the bar is generated, mask type is
%                   always rectangle. 
% grtMaskInt -      one mask to all grating . Set to 2. Since mask is
%                   constant
% gridSize -        protocol is only presented in a [1,1] grid
% freqCorrFlag -    meaningless in this protocol. { 0 }
%
%   NOTE!!! to correct small problems with diagonal line orientation is
%   implemented here and not in createProtocol 
%
% INPUT 
% Defaults are dilimited with {} and are optional inputs. 
%
% inputStruct -         Should have the following fields
% bar parameters
% .f/sBarWid -          single number for the width of each bar { 1 }
% .f/sBarVal -          1XN vector (0-1) of normalized luminance values for
%                       each bar. { 1 }
% .f/sBarPos -          1XP vector of positions for bars to be presented in. 0 is center, negative 
%                       andpositive are according to the convention in divideSquareToCols. { 'UI' }   
%
% general parameters
% .barsHeight -         Single number. height of bars in pixels. { 9 }
% .span -               Single number. span (in pixels) to define ref. positions 
%                       Will be converted into the width of the
%                       rectangular mask { 11 }
% .timeDiff -           1Xt vector of differenace between bars in secs
% .orientation -        Single number between 0 and 3. Since the bar isn't moving 4-7 are redundant. 
% .firstBarStat -       state of the first bar (0-2).
%                       0 - first bar disappear {defualt}
%                       1 - first bar remains
%                       2 - both (protocol with both options)
% .stepDur -            timing for bar presentation for when timeDiff == 0
%                       (o/w timeDiff is used as stepDur)
% .speedCor -           logical. If TRUE corrects for speed (dt is
%                       multiplied by dx) { 1 }
% .distCutoff -         Single number {defualt inf }. If distance in pixels between first and second bar 
%                       is euqal or bigger than this number, stimuli are
%                       not generated. If inf, all stim are generated
% .gsLevel -            gray scale level ( 3 ) 
% .gratingMidVal -      value of the rest of the window (0.49 - bkgd level)
% .gridCenter -         1X2 vector specifying the center of the grid in X and Y
%                       (sptial coordinates in pixels <for an 8X4 arena its 96X32). If one dimension of
%                       grid is even, grid will be presented around center but
%                       will not have a position in the actual center.
% .intFrames -          number of empty intervening frames. If not given half a
%                       second worth (based on generalFrequency)
% .repeats -            scalar. number of times the whole protocol repeats (passed into createProtocol) {3}
% .generalFrequency-    Frequency with which frames from the generated protocol
%                       will be dumped (passed on to runDumpProtocol) in position function units 
%                       (frames per second on the controller). 
%
% OUTPUT 
%
% protocolStruct with all the required fields from createProtocl. 
%
%
%       NOTE!!! when timeDiff is zero, the duration of the sim flash will
%       be the stepDur

%% GENERAL AND DEFAULT PARAMETERS

baseSiz = 225; % size of single frame or mask
gratingFuncHand = @generate2BarsFrameByInds;


default.fBarWid = 4;
default.sBarWid = 4;
default.fBarVal = 0;
default.sBarVal = 1;
default.fBarPos = 'UI';
default.sBarPos = 'UI';
default.barsHeight = 9;
default.span = 13;
default.stepDur = 0.16; 
default.timeDiff = [0.04, 0.16];
default.firstBarStat = 0;
default.speedCor = 0;
default.distCutoff = 5; 
default.gridCenter = 'UI';
default.gsLevel = 3;
default.gratingMidVal = 0.49;
default.orientations = 'UI';
default.intFrames = nan;
default.repeats = 3;
default.randomize = 1;


fixed.generalFrequency = 50;
fixed.maskType = {'rectangle'};
fixed.freqCorrFlag = 0;
fixed.grtMaskInt = 2;  
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
 assert(isvector(ort) && length(ort) == 1, 'Orientation should be a single number')
 assert(ismember(ort, 0:3), 'Orientation values should be between 0 and 7')
 
 newOrt = ort; 
 
 
 % orientation is implemented internally
 protocolStruct.orientations = 0;
 

  %% MASK (masks created with grating)
 
 maskHW = floor(default.span/2); % rectangle mask input is half width
 assert(isvector(maskHW) && length(maskHW) == 1, 'span should a single number')
 assert(maskHW > 1, 'span should be more than 1')
 
 relRegR = maskHW;
 relDiagR = round((2*relRegR+1)/sqrt(2)); % was +1 (with -1 overlap with non-rotated square is too small);
 
 if rem(newOrt,2) % if it is a diagonal orientation
     relMaxPos = relDiagR;
 else
     relMaxPos = relRegR;
 end
 
 minMaskR = min(maskHW);
 
 maskHH = floor(default.barsHeight/2);
 assert(isvector(maskHH) && length(maskHH) == 1, 'barHeight should be a single number');
 assert(maskHH > 1, 'barsHeight minimum should be more than 1')
 
 maskT = fixed.maskType;
 
 maskSt(1).type = maskT{1};
 maskSt(1).radius = [relMaxPos, maskHH];
 maskSt(1).ori = newOrt;

 protocolStruct.masksStruct = maskSt;

%% GRATING PARAMETERS


protocolStruct.generalFrequency = fixed.generalFrequency;

stepLen = default.stepDur; 
assert(isvector(stepLen) && length(stepLen)==1, 'stepDur should be one a single number')

stepFrames = round(stepLen * fixed.generalFrequency);
assert(stepFrames > 0, 'stimulus can not be presented for such a short duration. Minimal duration is 20ms')

gsLev = default.gsLevel;
assert(ismember(gsLev, 1:4), 'gsLevel should be an integer between 1 and 4')

bkgdVal = default.gratingMidVal;
assert(length(bkgdVal) == 1, 'gratingMidVal should be a single number');
assert(bkgdVal >=0 && bkgdVal <= 1, 'gratingMidVal should be between 0 and 1')

fbStat = default.firstBarStat;
assert(ismember(fbStat, 0:2), 'firstBarStat should be a number between 0 and 2')

spCorr = default.speedCor; 
assert(ismember(spCorr, [0,1]), 'speedCor should be logical')

distCO = default.distCutoff; 
assert(distCO > 1, 'distCutoff should be bigger than 1') %otherwise there are no double bars

fBarW = default.fBarWid;
assert(isvector(fBarW) && length(fBarW) == 1, 'fBarW should be a single number')
assert(fBarW > 0, 'fBarW should be a positive number')

sBarW = default.sBarWid;
assert(isvector(sBarW) && length(sBarW) == 1, 'sBarW should be a single number')
assert(sBarW > 0, 'sBarW should be a positive number')

if sBarW ~= fBarW
    warning('diagonal not corrected properly for different widths')
end

fBarV = default.fBarVal;
assert(isvector(fBarV), 'fBarVal should be a 1XV1 vector')
assert(min(fBarV) >=0 && max(fBarV) <= 1, 'fBarVal should be between 0 and 1');

sBarV = default.sBarVal;
assert(isvector(sBarV), 'sBarVal should be a 1XV2 vector')
assert(min(sBarV) >=0 && max(sBarV) <= 1, 'sBarVal should be between 0 and 1');

fBarPos = sort(default.fBarPos);
assert(isvector(fBarPos), 'fBarVal should be a 1XP1 vector')

sBarPos = sort(default.sBarPos);
assert(isvector(sBarPos), 'sBarPos should be a 1XP2 vector')


assert(all(ismember(abs(union(fBarPos, sBarPos)), 0:relMaxPos)), ...
       'position values are out of range: should be between %d to %d', -relMaxPos, relMaxPos)

timeDiff = sort(default.timeDiff);
assert(isvector(timeDiff), 'timeDiff should be a vector')

stepDiffFrames = unique(round(timeDiff * fixed.generalFrequency));

if length(stepDiffFrames) < length(timeDiff)
    warning('%d timeDiffs omitted since were the same after rounding', length(timeDiff) - length(stepDiffFrames))
end


relSpan = 2*maskHW+1; %max(2*maskHW+1, 2*maskHH+1); since when using divideTotSquareToCols height is not taken into account
count = 0;
cutoff = 0;
gratingArray = [];

    
for v1=1:length(fBarV)
    
    switch fbStat
        case 0 
            postFBVal = bkgdVal;
            pvFlag = 0;
        case 1
            postFBVal = fBarV(v1);
            pvFlag = 1;
        case 2
            postFBVal = [bkgdVal, fBarV(v1)];
            pvFlag = [0, 1];
    end
    
    for v2=1:length(sBarV)
    
        for tt=1:length(stepDiffFrames)
            
            if stepDiffFrames(tt) == 0
                stimFrames = stepFrames;
                combPos = [nan, nan];
            else
                stimFrames = stepDiffFrames(tt);
            end
        
            for pos1=1:length(fBarPos)
                
                for pos2=1:length(sBarPos)
                    
                    corrFac = abs(fBarPos(pos1) - sBarPos(pos2)) - 1;
                    
                    if abs(fBarPos(pos1) - sBarPos(pos2)) >= distCO
                        cutoff = cutoff+1;
                        continue
                    end
                    
                    if stepDiffFrames(tt) == 0 && fBarV(v1) == sBarV(v2)% getting rid of duplicates in sim presntation 
                        tempPos = [fBarPos(pos1), sBarPos(pos2)];
                        
                        if ismember(fliplr(tempPos), combPos, 'rows')
                            continue
                        else
                            combPos = vertcat(combPos, tempPos);
                        end
                    end
                    
                    for pv = 1:length(postFBVal)
                        
                        if pv == 2 % so as to not generate the diagonal same stim twice
                            if fBarPos(pos1) == sBarPos(pos2) || stepDiffFrames(tt)==0
                                continue
                            end
                        end
                        
                        count = count+1;
                        tempFV = [ones(1,stimFrames)*fBarV(v1), ...
                                  ones(1,stepDiffFrames(tt) * corrFac * spCorr) * postFBVal(pv), ...
                                  ones(1,stepDiffFrames(tt))*postFBVal(pv)];    
                        tempSV = [ones(1,stepDiffFrames(tt))*bkgdVal, ...
                                  ones(1,stepDiffFrames(tt) * corrFac * spCorr) * bkgdVal, ...
                                  ones(1,stimFrames)*sBarV(v2)];
                        
                        % to avoid 2 times the duration on the diagonal (if
                        % values are the same (if they differ it flickers
                        % once
                        if fBarPos(pos1) == sBarPos(pos2) && fBarV(v1) == sBarV(v2)
                            tempFV = tempSV; 
                        end
                        
                        
                        gtStruct(count).sqDim = relSpan; % generateBarFrameByInds corrects for diagonal internally
                        gtStruct(count).gsLevel = gsLev; 
                        gtStruct(count).bkgdVal = bkgdVal;
                        gtStruct(count).matSize = baseSiz;
                        gtStruct(count).ori = newOrt;
                        gtStruct(count).fWid = fBarW;
                        gtStruct(count).sWid = sBarW;
                        gtStruct(count).fVal = tempFV;
                        gtStruct(count).sVal = tempSV;
                        gtStruct(count).fPos = fBarPos(pos1);
                        gtStruct(count).sPos = sBarPos(pos2);
                
                        gratingArray = vertcat(gratingArray, ...
                                                        [count, fBarV(v1), sBarV(v2), fBarPos(pos1), sBarPos(pos2), ...
                                                        timeDiff(tt), pvFlag(pv)]); 
                                                    
                    end
                                                
                end
            end
        end
    end  
end


 tabVarNames =  {'index', 'FBVal', 'SBVal', 'FBPos', 'SBPos', 'timeDiff', 'FBStat'};
 gratTable = array2table(gratingArray, 'variablenames', tabVarNames);
 gratTable.Properties.Description = ['span:', num2str(relSpan), '; FBWid:', num2str(fBarW), '; SBWid:', num2str(sBarW), '; speedCorr:', num2str(spCorr)];
 
 protocolStruct.gratingTable = gratTable;
 protocolStruct.gratingStruct = gtStruct;

 if cutoff > 0
     fprintf('%d stimuli have been skipped since distance was larger than %d \n', cutoff, distCO)
 end
 
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










