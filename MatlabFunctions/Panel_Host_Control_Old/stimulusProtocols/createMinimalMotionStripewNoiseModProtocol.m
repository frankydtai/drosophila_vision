function protocolStruct = createMinimalMotionStripewNoiseModProtocol(inputStruct)

% function createMinimalMotionStripeNewProtocol(inputStruct)
%
% This function uses the inputStruct and createProtocol function to
% generate single bar that will appear in every position in the window. 
% after a specified period of time a second bar appear (in every other
% position). It has certain assumptions and therefore requires
% less inputs. 
% The difference between this and the previous version is that this
% protocol generates all the possible pairs in the window and not just
% pairs with first bar in center and a certain distance. Also this function
% does not control step size, it produces all possible combinations within
% the window
% If function is called with no inputStruct, it prompt the user for
% relevant input and allows to change the default
% 
% ASSUMPTIONS
% Since this is a grating certain parameters are assumed to be symmetrical.
% Width -           1 for all stim
% Randomize -       everything is randomized (so that if several gratings appear
%                   they appear in all positions and in all oreintations. 
% Masks positions - For this function only one position is allowed
%                   (otherwise duplications will occur)
% maskType -       'rectangle'. Again to avoid duplications
% gratingFuncHand - Function uses the generateGratingFrame
% stimLength -      previous versions allowed to play with this. Now it is
%                   set to 1 frame and changed with a combination of
%                   generalFrequency (max 50). 
% generalFrequency- set to 50 
%
%
% INPUT 
% Defaults are dilimited with {} and are optional inputs. 
% Only 2 obligatory fields are width and gridCenter
%
% inputStruct -     Should have the following fields
% .firstBar -       1XN vector (0-1) Brightness of the first bar { 1 }
% .secondBar -      1XN vector (0-1) Brightness of second bar { 1 }.
% 
%   NOTE: first and second bars can be either of same length or either can
%   be of length 1. In that case the single value will be applied to all
%   values of the other parameter. 
%
% .windowHalfWidth- This parameter will determine 
%                   the half width of the rectangular window {3}. Max
%                   distace between bars will be 2XwindowHalfWidth.
%                   Number is rounded to correspond to pixels
% .barHeight -      height of bar in pixels (second dimension of the
%                   rectangular mask). {9}  
%                   Would be used
% .orientation -    integer (0-3). Since the bar isn't moving 4-7 are redundant. 
%                   For this protocol only one orientation is allowed since
%                   it also determines the mask position. 
%                   Note! diagonals with bars of width one generate
%                   distortions
%
% .stimDur -        duration in seconds in which a single bar will appear 
% .gsLevel -        gray scale level ( 3 ) 
% .gratingMidVal -  value of the rest of the window (0.49 - bkgd level)
% .maskPositions -  For this version of the function it should be just one
%                   position (otherwise duplications will occur).
%
%                   Note! for this protocol the regular grid paramters
%                   aren't used since the stimulus is presented in windows 
%
% .winCenter -      1X2 vector specifying the center of the window in X and Y
%                   (sptial coordinates in pixels <for an 8X4 arena its 96X32). 
%                   This function is not presented in a grid but in a
%                   single window
% .grtMaskInt -     logical. interleave of grating and masks given (would
%                   be handed into createProtocol. { 1 }
% .intFrames -      number of empty intervening frames. If not given quarter a
%                   second worth (based on generalFrequency)
% .repeats -        scalar. number of times the whole protocol repeats (passed into createProtocol) {3}
% .presentSimFlag - logical. Whether to present both bar flashing together
%                   also { 1 } 
% addNoise -        logical. If true will add a frame of noise for each
%                   stim (based on a noise structure that the user will
%                   feed in) { 1 }
%
%       NOTE: frequency is meanningless in the protocol since duration of bar
%       presentation is corrected for it
%
% .freqCorrFlag -   Also passed on to runDumpProtocol. Logical flag to
%                   indicate whether different stimuli should be run with temporal frequency
%                   correction { 1 }.  
%
% OUTPUT 
% protocolStruct with all the required fields from createProtocl. 
%
% protocolStruct in this protocol has an additional field call gratingInds
% which specify several relevant parameters for each grating in the grating
% structure. Each enetry contains 5 elements: 
%       (1) First bar brightness value
%       (2) Second bar brightness value
%       (3) First bar position in window reference frame (from -R to R,
%       zero being the middle)
%       (4) Second bar position in window reference frame. 
%       (5) Delay between bars in frames
%
%
%                   NOTE! masks and gratings need not be of the same length
%                   NOTE! grid mask positions will use only the first
%                   maskRadius value to interpert overlap. 

%% GENERAL AND DEFAULT PARAMETERS

arenaSize = [96,32];
gratingFuncHand = @generate4BarsGratingFrame;

default.firstBar = 1;
default.secondBar = 1;
default.windowHalfWidth = 3;
default.barHeight = 7;
default.winCenter = 'UI';
default.gsLevel = 3;
default.gratingMidVal = 0.49;
default.orientations = 'UI';
default.stimDur = 0.04; 
default.grtMaskInt = 0;
default.intFrames = nan;
default.repeats = 3;
default.randomize = 1;
default.presentSimFlag = 0;
default.addNoise = 0;

fixed.generalFrequency = 50;
fixed.maskType = {'rectangle'};
fixed.freqCorrFlag = 0;


% combining default and input structures
if nargin == 0
    default = modifyDefaultStruct(default);
else
    default = modifyDefaultStruct(default, inputStruct);
end

%% MASK
 
maskT = fixed.maskType{1};
 
winHW = round(default.windowHalfWidth);
assert(length(winHW) == 1, 'window half width can only be a single number')
assert(winHW > 0, 'window half width should be a positive number')

barH = default.barHeight;
assert(length(barH) == 1, 'bar height can only be a single number')
assert(barH > 0, 'bar height should be a positive number')
barhalfH = floor(barH/2);
 
maskSt(1).type = maskT;
maskSt(1).radius = [winHW, barhalfH];
  
 
protocolStruct.masksStruct = maskSt;
 
numMasks = length(maskSt);

%% GRATING PARAMETERS

% needed to determine number of frames to appear
protocolStruct.generalFrequency = fixed.generalFrequency;

stimDur = default.stimDur; 
assert(length(stimDur) == 1, 'StimLength should be one number')
assert(stimDur > 0, 'stimLength should be a positive number')

stimFrames = round(stimDur * fixed.generalFrequency);
assert(stimFrames > 0, 'stimulus can not be presented for such a short duration. Minimal duration is 20ms')

gsLev = default.gsLevel;
assert(ismember(gsLev, 1:4), 'gsLevel should be an integer between 1 and 4')

bkgdVal = default.gratingMidVal;
assert(length(bkgdVal) == 1, 'gratingMidVal should be a single number');
assert(bkgdVal >=0 && bkgdVal <= 1, 'gratingMidVal should be between 0 and 1')

stimF = default.firstBar;
assert(min(stimF) >=0 && max(stimF) <= 1, 'firstBar should be between 0 and 1');
assert(isvector(stimF), 'firstBar should be a 1XN vector')

stimS = default.secondBar;
assert(min(stimS) >=0 && max(stimS) <= 1, 'secondBar should be between 0 and 1');
assert(isvector(stimS), 'secondBar should be a 1XN vector')

preSim = default.presentSimFlag;
assert(ismember(preSim, [0,1]), 'presentSimFlag should be logical')

if length(stimF) ~= length(stimS)
    if      length(stimF) == 1
        stimF = ones(1, length(stimS)) * stimF;
    elseif  length(stimS) == 1
        stimS = ones(1, length(stimF)) * stimS;
    else
        error('first and second bar should be either of length 1 or identical lengths')
    end
end


% Changes second bar values so that it will differ from first and easier to
% adjust in fixImAfterRot
% Only deals with 0 & 1 in first bar (other values will not get fixed
stimS(stimS == 0) = 0.0005;
stimS(stimS == 1) = 0.9995;

count = 0;

relMaskRHW = maskSt(1).radius(1);
relPos = -relMaskRHW:relMaskRHW;
for ii=1:length(stimF)
        
    ccVals = [bkgdVal, stimF(ii)]; % for use in the cc loop
    
    vals1BSim = ones(1, stimFrames) * stimF(ii);
    vals2BSim = ones(1, stimFrames) * bkgdVal;
    vals3BSim = ones(1, stimFrames) * stimS(ii); % generates a flash of both bars sim 
    vals4BSim = ones(1, stimFrames) * bkgdVal;
    
    for jj=1:length(relPos)
        for kk=0:length(relPos)-1
            
            if kk > 0
                widB2 = kk-1;
                widB3 = 1;
                widB4 = 2*relMaskRHW + 1 -2 -widB2; % 2R+1 si window size and 2 is width of other 2 bars
                secPos =  1 + widB2 + relPos(jj);
                if secPos > relPos(end)
                    secPos = secPos -2*relPos(end) -1;
                end
            else % kk==0 (present bars in the same position - no memvement)
                widB2 = 0;
                widB3 = 0;
                widB4 = 2*relMaskRHW + 1 -1 -widB2; % 2R+1 si window size and 1 is width of other bar
                secPos = relPos(jj);
            end
            
            posDiff = abs(relPos(jj) - secPos);
            
            
            for cc=1:2 % for sequential (first bar disappears) and Consequtive (first bar stays on) 
                
                if kk==0 && cc==2 % so that sim stim wont be generated twice
                    continue
                end
                
                count = count+1;
                
                %vals1B = [ones(1, stimFrames)*stimF(ii), ones(1, (posDiff-1)*stimFrames)*bkgdVal, ones(1, stimFrames)*ccVals(cc)];
                vals1B = [ones(1, stimFrames)*stimF(ii), ones(1, posDiff*stimFrames)*ccVals(cc)];
                vals2B = ones(1, (posDiff+1)*stimFrames) * bkgdVal;
                vals3B = [ones(1, posDiff*stimFrames)*bkgdVal, ones(1,stimFrames)*stimS(ii)];
                vals4B = ones(1, (posDiff+1)*stimFrames) * bkgdVal;
                vals1BSame = vals1B;
                vals1BSame(stimFrames+1:2*stimFrames) = vals3BSim; %in case the first and second bar are of different contrasts
                
                if kk>0
                    relVal1B = vals1B;
                    relVal2B = vals2B;
                    relVal3B = vals3B;
                    relVal4B = vals4B;
                else
                    relVal1B = vals1BSame;
                    relVal2B = ones(1, 2*stimFrames) * bkgdVal;
                    relVal3B = ones(1, 2*stimFrames) * bkgdVal;
                    relVal4B = ones(1, 2*stimFrames) * bkgdVal;
                end
                
                gtStruct(count).width1 = 1;
                gtStruct(count).width2 = widB2;
                gtStruct(count).width3 = widB3;
                gtStruct(count).width4 = widB4;
                gtStruct(count).vals1St = relVal1B;
                gtStruct(count).vals1End = relVal1B;
                gtStruct(count).vals2St = relVal2B;
                gtStruct(count).vals2End = relVal2B;
                gtStruct(count).vals3St = relVal3B;
                gtStruct(count).vals3End = relVal3B;
                gtStruct(count).vals4St = relVal4B;
                gtStruct(count).vals4End = relVal4B;
                gtStruct(count).barAtPos = 1;
                gtStruct(count).gsLevel = gsLev; 
                gtStruct(count).position = relPos(jj);
                gtStruct(count).roundFlag = 0;
                newMaskSt(count) = maskSt(1);
                grtInds(count, :) = [stimF(ii), stimS(ii), relPos(jj), secPos, cc];
                
            end
                
            if preSim
                if secPos >= relPos(jj) % adds sim flash of bars (in same position would present just the first bar)
                    count = count+1;
                    gtStruct(count).width1 = 1;
                    gtStruct(count).width2 = widB2;
                    gtStruct(count).width3 = widB3;
                    gtStruct(count).width4 = widB4;
                    gtStruct(count).vals1St = vals1BSim;
                    gtStruct(count).vals1End = vals1BSim;
                    gtStruct(count).vals2St = vals2BSim;
                    gtStruct(count).vals2End = vals2BSim;
                    gtStruct(count).vals3St = vals3BSim;
                    gtStruct(count).vals3End = vals3BSim;
                    gtStruct(count).vals4St = vals4BSim;
                    gtStruct(count).vals4End = vals4BSim;
                    gtStruct(count).barAtPos = 1;
                    gtStruct(count).gsLevel = gsLev; 
                    gtStruct(count).position = relPos(jj);
                    gtStruct(count).roundFlag = 0;
                    newMaskSt(count) = maskSt(1);
                
                    grtInds(count, :) = [stimF(ii), stimS(ii), relPos(jj), secPos, 0];
                end
            end
            
            
        end
    end
end
 
 
 protocolStruct.gratingStruct = gtStruct;
 protocolStruct.gratingInds = grtInds;
 protocolStruct.masksStruct = newMaskSt;
 
%% ORIENTATIONS
 
 ort = default.orientations;
 assert(isvector(ort) && length(ort) == 1, 'Orientation should be vector of length 1')
 assert(prod(ismember(ort, 0:3)) == 1, 'Orientation values should be between 0 and 3')
 
 if ismember(ort, [1,3])
     beep
     fprintf('\nDiagonal orientation produces lines of different widths and lengths \n')
 end
     
 
 protocolStruct.orientations = ort;
 
 
 %% GRID 
 % No grid in this function - just a single position
 maskPos = default.winCenter;
 assert(size(maskPos, 2) == 2, 'mask positions should be an 1X2 matrix')
 assert(size(maskPos, 2) == 2, 'mask positions should be an 1X2 matrix')
 assert(max(maskPos(:,1)) <= arenaSize(1), 'mask position X values should not exceed %d', arenaSize(1))
 assert(max(maskPos(:,2)) <= arenaSize(2), 'mask position Y values should not exceed %d', arenaSize(2))
 assert(min(maskPos(:)) > 0, 'mask position values should be positive')
 protocolStruct.maskPositions = maskPos;
     
 
 %% Misc parameters
 
 protocolStruct.freqCorrFlag = fixed.freqCorrFlag;
 
 protocolStruct.funcHand = gratingFuncHand;
 protocolStruct.interleave = default.grtMaskInt;
 
 intF = default.intFrames;
 if isnan(intF)
     protocolStruct.intFrames = floor(fixed.generalFrequency/4);
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
 
 if ismember(ort, [1,3])
     protocolStruct = createProtocolMod(protocolStruct);
 else
     protocolStruct = createProtocol(protocolStruct);
 end
 
 protocolStruct.inputParams = default;
 
 
 %% adding random noise in the end of each stim
 
 addN = default.addNoise;
 assert(ismember(addN, [0,1]), 'addNoise should be a logical')
 
 if addN
    noiseSt.repeats = default.repeats;
    noiseSt.totNumStim = length(gtStruct);
    noiseSt.maskRadius = winHW + 2; % so that it is a bit bigger than the stimulated region
    noiseSt.gridCenter = maskPos;
 
    noiseProtSt = createRandomFrameProtocol(noiseSt);
 
    for ii=1:length(noiseProtSt.stim)
         tempNoise = noiseProtSt.stim(ii).matCell;
        protocolStruct.stim(ii).matCell = cat(3, protocolStruct.stim(ii).matCell, tempNoise);
    end
 
    protocolStruct.noiseStruct = noiseProtSt;
    
 end
 
end










