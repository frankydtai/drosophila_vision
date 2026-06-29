function maskPos = makeLinearTrajectory(paramStruct)

% function maskPos = makeLinearTrajectory(paramStruct)
%
% This function generates a linear tracetory by interpolating between start
% and stop with the relevant number of steps. Intermediate values are
% rounded
%
% INPUT
%
% LINEARTRAJECTORY paramStruct should have the fields: 
% .startPos         1X2 vector (x,y position on arena)
% .endPos           1x2 vector (same)
% .numSteps         integer (number of steps between start and stop - linearly
%                   interpolated and rounded to fit arena)
% .arenaSize        (optional). If not given the default is [96, 32]
%
% parameters should be given in spatial coordinates (X horizontal).The function does
% not transforms spatial to controller coordinates (done in
% getArenaMaskTransform)
%
% OUTPUT
% maskPos - NX2 matrix of positions in arena coordinates

% if isfield(paramStruct, 'arenaSize')
%     arenaSiz = paramStruct.arenaSize;
% else
%     arenaSiz  = [96, 32];
% end

startPos = paramStruct.startPos;
endPos = paramStruct.endPos;
steps = paramStruct.numSteps;

% Error checking moved to makeLinearTrajectoriesGrid 
%
% if min([startPos, endPos]) < 1
%     error('position outside arena range: lees than 1')
% elseif max(startPos(1), endPos(1)) > arenaSiz(1)
%     error('position outside arena range: exceed in X')
% elseif max(startPos(2), endPos(2)) > arenaSiz(2)
%     error('position outside arena range: exceed in Y')
% end





% indexing is off on purpose to convert spatial to matlab arena coordinates
xCrds = round(linspace(startPos(1), endPos(1), steps));
yCrds = round(linspace(startPos(2), endPos(2), steps));

maskPos = [xCrds', yCrds'];



end





