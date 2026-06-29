function iniPos = generateIniPositions(peakX)

% function iniPos = generateIniPositions(peakX)
%
% This function is designed to work with 8X8 and 16X16 window patterns that 
% share a common X positioning (e.g change in X moves window while change 
% in Y moves/changes pattern). The X position is 1(empty) + 23X6 
% (xx and yy on the actual arena moving in half panel jumps) 
%
% INPUT
%
% peakX -   x position in which maximal response was achieved (window
% position)
%
% OUTPUT
% iniPos -  2 element vector, with first being 3X3 initial position (QF) 
%           and second being 2X2 initial position (HR)  

tempX = rem(peakX, 23);
tempY = ceil(peakX/23);

modXQF = tempX-2;
if modXQF < 1
    modXQF = 1;
end

modYQF = tempY-2;
if modYQF < 1
    modYQF = 1;
end

iniPos(1) = modXQF + (modYQF-1)*23;

modXHR = tempX-1;
if modXHR < 1
    modXHR = 1;
end

modYHR = tempY-1;
if modYHR < 1
    modYHR = 1;
end

iniPos(2) = modXHR + (modYHR-1)*23;


end



