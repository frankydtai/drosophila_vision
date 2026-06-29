function iniPosStruct = generateIniPositions(peakX, peakY)

% function iniPosStruct = generateIniPositions(peakX, peakY)
%
% This function generates initial positions for the Short protocols based on
% the results from the Base protocols. 
% It uses the following assumaptions: 
% Short for HBars is 16 positions long,
% Short for VBars is 32 positions long, 
% Short for 16Sq is 3X2 (in jumps of 2) positions long using this matrix (1-40X 1-24Y)
%   5 85 - 105
% 	4 64 - 84
% 	3 43 - 63
% 	2 22 - 42
% 	1 1  - 21
%
% Short for 08Sq is 2.5X2 (in jumps of 4) positions long using this matrix (45x X 11y in actual arena)
%
% INPUT
% peakX, peakY - position for peak response along both arena axis (96
% positions in 1 and 32 in the other). Based on the results from base
% protocol. 
%
% OUTPUT
% iniPosStruct - structure with the following fields:
% iniposHbars - for all horizontal patterns
% iniposVbars - for all vertical patterns
% inipos16Sq - for 16X16 patterns (looming)
% inipos08sq - for 08X08 patterns (fading and edges)
% iniposCrossSt - for the short protocol of crossRF
% iniposCrossVSt - for the very short protocol of crossRF (random flashing)
% Should be used in createProtocolScript

% Vbars
VBars = [peakX-16, 1];
if VBars(1)+32 > 96 % so that section will not loop around unnecessarily
    VBars(1) = 96-32;
elseif VBars(1) < 1
    VBars(1) = 1;
end

iniPosStruct.VBars = VBars;

%Hbars
HBars = [peakY-8, 1];
if HBars(1)+16 > 32 % so that section will not loop around unnecessarily
    HBars(1) = 32-16;
elseif HBars(1) < 1
    HBars(1) = 1;
end

iniPosStruct.HBars = HBars;

%16Sq
baseXsq16 = floor(linspace(17,86, 17));
if peakX < 17
    sq16baseX = 1;
elseif peakX > 86
    sq16baseX = 18;
else
    sq16baseX = find(baseXsq16 - peakX < 0, 1, 'last')+1;
end

if peakY < 12
    sq16Y = 1;
elseif peakY < 18
    sq16Y = 2;
else
    sq16Y = 3;
end

SQ16 = [(sq16Y-1)*21 + sq16baseX, 1];
iniPosStruct.SQ16 = SQ16;

%08Sq
baseXsq08 = floor(linspace(10,86, 38));
if peakX < 10
    sq08baseX = 1;
elseif peakX >= 86
    sq08baseX = 39;
else
    sq08baseX = find(baseXsq08 - peakX <= 0, 1, 'last')+1;
end


baseYsq08 = floor(linspace(9,22, 6));
if peakY < 9
    sq08baseY = 1;
elseif peakY >= 22
    sq08baseY = 7;
else
    sq08baseY = find(baseYsq08 - peakY >= 0, 1)+1;
end
    
SQ08 = [(sq08baseY-1)*45 + sq08baseX, 1];
iniPosStruct.SQ08 = SQ08;


% CrossRF
% Short
CrossX = ceil(peakX/2)-6;
if CrossX+13 > 48 
    CrossX = 48-13;
elseif CrossX < 1
    CrossX = 1;
end

CrossY = ceil(peakY/2)-6;
if CrossY+13 > 16 
    CrossY = 16-13;
elseif CrossY < 1
    CrossY = 1;
end

iniPosStruct.CrossSt = [CrossX, CrossY];

% Very Short

CrossXV = ceil(peakX/2)-3;
if CrossXV+8 > 48 
    CrossXV = 48-8;
elseif CrossXV < 1
    CrossXV = 1;
end

CrossYV = ceil(peakY/2)-3;
if CrossYV+8 > 16 
    CrossYV = 16-8;
elseif CrossYV < 1
    CrossYV = 1;
end


iniPosStruct.CrossVSt = [CrossXV, CrossYV];

% diagonal

crossDiag = [CrossX-6 , CrossY];
if crossDiag(1) < 1
    crossDiag(1) = 49 + crossDiag(1);
end

iniPosStruct.CrossDiag = crossDiag;


end



