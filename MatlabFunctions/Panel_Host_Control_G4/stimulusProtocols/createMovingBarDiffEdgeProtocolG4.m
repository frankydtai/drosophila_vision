function protocolStruct = createMovingBarDiffEdgeProtocolG4(inputStruct)

% function createMovingBarDiffEdgeProtocolG4(inputStruct)
%
% This function uses the inputStruct and createProtocol function to
% generate a bar that moves across the positions of the window. It is a
% modification of the old createMovingBarProtocol function and as such uses
% generateGratingFrame and not generateBarFrameByInds. 
%
% Therefore it is best to use it only for cardinal directions
%
% The stimulus is designed to to generate a moving object with edges of the same sign.
% (and their corresponding control objects)
% It has certain assumptions and therefore requires less inputs. 
% If function is called with no inputStruct, it prompt the user for
% relevant input and allows to change the default. 
% Added reverse Phi motion (bar flip contrast with every movement) 
% 
% ASSUMPTIONS
% Since this is a grating certain parameters are assumed to be symmetrical.
% 
% position -        Bar will move across in all window positions
% Contrast -        each bar is uniform (no on/off gradients), and both are same
%                   distance from mid level GS (background). 
% Randomize -       everything is randomized (so that if several gratings appear
%                   they appear in all positions and in all oreintations. 
% maskType -        becuase of the way movement is generated it is fixed at
%                   rectangle
% grtMaskInt -      since grating width is completely dependent only on mask
%                   width, they will be read independently. {interleave 2}
% gratingFuncHand - Function uses the generateGratingFrame
%
%
% INPUT 
% Defaults are dilimited with {} and are optional inputs. 
% Only 2 obligatory fields are maskRadius and gridCenter
%
% inputStruct -     Should have the following fields
% .barWidth  -      1XN vector. Width of the moving bar. { 2 }
% .barHeight -      1XM vector. Height of moving bar. practically
%                   determines the height of the rectangle window. If more
%                   than 1 is given would be interleaved with barWidth. { 5 }
% .barDist -        For now single value. distance in pixels tranveled by the bar. Practically width of the rectangular window { 9 }
% .stepDur -        duration in sec for each step of the bar. (min is 20ms, since max freq is 50) 
% .gsLevel -        gray scale level for grating frames
% .contrast -       Difference between bright and dark bars relative to
%                   mid. For now length should be 1.
% .maskPositions -  User can specify these directly as an NX2 matrix, or
%                   use the other parameters to generate them (if this is
%                   given other parameters are disregarded
% .maskType -       'circle' or {'square'}. Can also be a 1XM cell array 
% .maskInt -        logical. If TRUE function will generate all the combinations
%                   between type and radius. { 17 } 
% .gridSize  -      1X2 vector specifying size of grid in X and Y (spatial
%                   coordinates). { [1,1] }
% .gridPosOverlap - Overlap between different positions on mask position grid. 
%                   Units s are normalized maskSizes so that 0 means no overlap and no gap, 
%                   1 means complete overlap (meaningless), and -1 means a gap of one mask
%                   between positions. { 0 }
% .orientations -   bar movement direction { 0:2:6 - 4 cardinal directions}
% .gridCenter -     1X2 vector specifying the center of the grid in X and Y
%                   (sptial coordinates in pixels <for an 8X4 arena its 96X32). If one dimension of
%                   grid is even, grid will be presented around center but
%                   will not have a position in the actual center. {[48, 16]}
% .intFrames -      number of empty intervening frames. If not given fifth a
%                   second worth (based on generalFrequency)
% .repeats -        scalar. number of times the whole protocol repeats (passed into createProtocol) {3} 
% .generalFrequency-Frequency with which frames from the generated protocol
%                   will be dumped (passed on to runDumpProtocol) in Hz.
%                   { 15 }
% .freqCorrFlag -   Also passed on to runDumpProtocol. Logical flag to
%                   indicate whether different stimuli should be run with temporal frequency
%                   correction { 0 }.  
%
% OUTPUT 
% protocolStruct with all the required fields from createProtocl. 
%
%
%                   NOTE! masks and gratings need not be of the same length
%                   NOTE! grid mask positions will use only the first
%                   maskRadius value to interpert overlap. 
%                   Note! since mask is bigger than arena's height, first
%                   and last position of horizontal bars are not displayed 

%% GENERAL AND DEFAULT PARAMETERS

arenaSize = [192,48];
gratingFuncHand = @generateGratingFrame;

default.gridCenter = 'UI';
default.barWidth = 7; 
default.barHeight = 13;
default.barDist = 21;
default.stepDur = 0.04;
default.contrast = 1;
default.orientations = [0,4];
default.maskInt = 1;
default.gridOverlap = 0;
default.gratingMidVal = 0.49;
default.intFrames = nan;
default.repeats = 3;
default.randomize = 1;


% fixed parameters for this protocol that are not user modified

fixed.gsLevel = 4;
fixed.barCont = [00, 01, 10, 11]; % DD D to B, B to D BB
fixed.maskType = {'rectangle'};
fixed.gridSize = [1,1];
fixed.grtMaskInt = 1;
fixed.generalFrequency = 500;
fixed.freqCorrFlag = 0;

% combining default and input structures
if nargin ==0
    default = modifyDefaultStruct(default);
else
    default = modifyDefaultStruct(default, inputStruct);
end


 %% ORIENTATIONS
 
 ort = default.orientations;
 assert(isvector(ort), 'Orientation should be 1XM vector')
 assert(prod(ismember(ort, 0:2:6)) == 1, 'Orientation values should be between 0,2,4, or 6')
 
 protocolStruct.orientations = ort;
 
 %% MASK
 
 maskHW = floor(default.barDist/2); % rectangle mask input is half width
 assert(isvector(maskHW), 'barDist should be a 1XN vector')
 assert(min(maskHW) > 1, 'barDist should be a positive number')
 
 
 maskHH = floor(default.barHeight/2);
 assert(isvector(maskHH), 'barHeight should be a 1XM vector');
 assert(min(maskHH) > 0, 'barHeight minimum should be a positive number')
 
 maskT = fixed.maskType;
 
 maskCount = 0;
 for ii=1:length(maskHW)
     for jj=1:length(maskHH)
         maskCount = maskCount+1;
         maskSt(maskCount).type = maskT{1};
         maskSt(maskCount).radius = [maskHW(ii), maskHH(jj)];
     end
 end
 
 
 % might change after startBar is read in 
 %numMasks = length(maskSt);

 %% GRATING PARAMETERS
 
 stepDur = default.stepDur; 
 assert(isvector(stepDur), 'stepDur should be a vector')
 assert(min(stepDur) > 0, 'stepDur should be a positive number (in secs)')

 stepFrames = unique(round(stepDur * fixed.generalFrequency));
 if length(stepFrames) < length(stepDur)
     warning('%d step durations omitted since were the same after rounding', length(stepDur) - length(stepFrames))
 end
 
 assert(min(stepFrames) > 0, 'stimulus can not be presented for such a short duration. Minimal duration is 20ms')

 
 barW = default.barWidth;
 assert(isvector(barW), 'barWidth should be a vector')
 assert(min(barW) >= 3, 'barWidth should be at least 3')
 
 barC = fixed.barCont;
 
 cont = default.contrast;
 assert(isvector(cont) && length(cont) == 1, 'Contrast should be vector of length 1')
 assert(cont > 0 && cont <= 1, 'Contrast should be  0 < cont <= 1');
 
 bkgdVal = default.gratingMidVal;
 assert(length(bkgdVal) == 1, 'gratingMidVal should be a single number');
 assert(bkgdVal >=0 && bkgdVal <= 1, 'gratingMidVal should be between 0 and 1')
 
 onVal = bkgdVal + cont/2; % 0.49 is for the middle value to be rounded down (in GS3 it is 3 and not 4)
 offVal = bkgdVal - cont/2.041; % so that it wont go negative
 
 
 gsLev = fixed.gsLevel;
 
 count = 0;
 gratingArray = [];
 
for mm=1:length(maskSt)
    
    relMaskR = maskSt(mm).radius(1); 
    
    for ii=1:length(barW)
       
        for jj=1:length(barC)
            
            for kk=1:length(stepFrames)
        
                count = count+1;
                
                switch barC(jj)
                    case 0
                        gtStruct(count).valsONSt = bkgdVal;
                        gtStruct(count).valsONEnd = bkgdVal;
                        gtStruct(count).valsOFFSt = offVal;
                        gtStruct(count).valsOFFEnd = offVal;
                        gtStruct(count).widthON = 2*relMaskR+1; % to make sure bar gradually appears and disappears
                        gtStruct(count).widthOFF = barW(ii);
                        gtStruct(count).barAtPos = 0;
                    case 1
                        gtStruct(count).valsONSt = bkgdVal;
                        gtStruct(count).valsONEnd = bkgdVal;
                        gtStruct(count).valsOFFSt = offVal;
                        gtStruct(count).valsOFFEnd = onVal;
                        gtStruct(count).widthON = 2*relMaskR+1; 
                        gtStruct(count).widthOFF = barW(ii);
                        gtStruct(count).barAtPos = 0;
                    case 10
                        gtStruct(count).valsONSt = onVal;
                        gtStruct(count).valsONEnd = offVal;
                        gtStruct(count).valsOFFSt = bkgdVal;
                        gtStruct(count).valsOFFEnd = bkgdVal;
                        gtStruct(count).widthOFF = 2*relMaskR+1; 
                        gtStruct(count).widthON = barW(ii);
                        gtStruct(count).barAtPos = 1;
                    case 11
                        gtStruct(count).valsONSt = onVal;
                        gtStruct(count).valsONEnd = onVal;
                        gtStruct(count).valsOFFSt = bkgdVal;
                        gtStruct(count).valsOFFEnd = bkgdVal;
                        gtStruct(count).widthOFF = 2*relMaskR+1; 
                        gtStruct(count).widthON = barW(ii);
                        gtStruct(count).barAtPos = 1;
                end
                
                
            
                basePos = -(relMaskR+barW(ii)):relMaskR;
                
                gtStruct(count).pos = basePos;
                gtStruct(count).gsLevel = gsLev; 
                gtStruct(count).stepFrames = stepFrames(kk);
                
                newMaskSt(count) = maskSt(mm); 
                span = 2*maskSt(mm).radius(1)+1;
                height = 2*maskSt(mm).radius(2)+1;

                gratingArray = vertcat(gratingArray, ...
                                       [count, barC(jj), barW(ii), span, ...
                                        stepFrames(kk)/fixed.generalFrequency, height]);
                
            end
        end
        
    end
    
end
        
 
 tabVarNames =  {'index', 'barCont', 'width', 'span', 'stepDur', 'height'};

 protocolStruct.gratingTable = array2table(gratingArray, 'variablenames', tabVarNames);

 protocolStruct.gratingStruct = gtStruct;
 protocolStruct.masksStruct = newMaskSt;
 protocolStruct.relGtStName = 'pos'; 
 
 %% GRID 
 
 % if maskPositions exist then the rest of the parameters are unnecessary
 if nargin  ==1 && isfield(inputStruct, 'maskPositions')
     maskPos = inputStruct.maskPositions;
     assert(size(maskPos, 2) == 2, 'mask positions should be an NX2 matrix')
     assert(max(maskPos(:,1)) <= arenaSize(1), 'mask position X values should not exceed %d', arenaSize(1))
     assert(max(maskPos(:,2)) <= arenaSize(2), 'mask position Y values should not exceed %d', arenaSize(2))
     assert(min(maskPos(:)) > 0, 'mask position values should be positive')
     protocolStruct.maskPositions = maskPos;
     
 else %if maskPositions isn't given use default paramters (or grid input)
     
     gridSt.gridSize = fixed.gridSize;
     
     ovlp = default.gridOverlap;
     assert(isscalar(ovlp), 'Overlap should be a single number')
     assert(ovlp < 1, 'Overlap value above 1 are not accepted')
     assert(ovlp ~=1, 'What are you stupid? overlap 1 means no grid')
 
     maskS = 2*max(maskHH)+1;
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
    
 end
 
 %% Misc parameters
 
 protocolStruct.freqCorrFlag = fixed.freqCorrFlag;
 protocolStruct.generalFrequency = fixed.generalFrequency;
 
 protocolStruct.funcHand = gratingFuncHand;
 protocolStruct.interleave = fixed.grtMaskInt;
 
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
 
 protocolStruct = createProtocolG4(protocolStruct);
 
 protocolStruct.inputParams = default;
 protocolStruct.inputParams.gsLevel = gsLev;  

 
end










