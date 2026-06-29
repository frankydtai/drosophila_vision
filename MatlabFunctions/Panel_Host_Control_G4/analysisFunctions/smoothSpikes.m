function smVec = smoothSpikes(vVec, smWin)

% function smVec = smoothSpikes(vVec)
%
% This function uses a median filter to smooth spikes out so that the
% baseline activity can be seen 
%
% INPUT 
% vVec -    repXN vector of voltage signal
% smWin -   (optional) integer. size of the window to be used in median
%           filter. Defualt is 101
%
% OUTPUT
% smVec -   repXN smooth vector

if nargin < 2
    smWin = 101;
end

smVec = movmedian(vVec, smWin);







end