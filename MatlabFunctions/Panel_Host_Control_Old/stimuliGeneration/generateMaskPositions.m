function maskPos = generateMaskPositions(funHand, paramStruct)

% function maskPos = generateMaskPositions(funHand, paramStruct)
%
% This function generates an Nx2 matrix of mask position (relative to
% arena) using a set of subordinate functions. These include "grid",
% "linearTrajectory" etc., each with their correcsponding set of
% parameters. All input should be in arena coordinates (bottom left pixel is <1,1>) and given in pixles 
% GRID paramStruct should have the fields: 
%   .gridSize (1X2 vector) 
%   .startPos (bottom-left corner of the grid - 1X2 vector) 
%   .spacing (spacing between adject points, on both X and Y dimensions 1X2 vector)
%
% LINEARTRAJECTORY paramStruct should have the fields: 
% .startPos 1X2 vector (x,y position on arena)
% .endPos   1x2 vector (same)
% .numSteps integer (number of steps between start and stop - linearly
%                   interpolated and rounded to fit arena)
%
% NOTE! arena coordinates in paramStruct should be given based on spatial
% coordinates (i.e. X has 96 pixels and Y 32)

% checking inputs are proper
relFuncHand = {'makeGrid'; 'makeLinearTrajectory'};
relFields{1} = {'gridSize', 'startPos', 'spacing'};
relFields{2} = {'startPos', 'endPos', 'numSteps'};

%function handle is legit
relFuncTag = find(cellfun(@(x) strcmp(x, func2str(funHand)), relFuncHand));

if isempty(relFuncTag)
    error('Function handle to irrelevant function')
end

% parameter structure has all the right fields
fnames = fieldnames(paramStruct);
fieldsSum = sum(ismember(fnames, relFields{relFuncTag}));

if fieldsSum < length(relFields{relFuncTag})
    error('some fields are missing from paramStruct')
end



maskPos = funHand(paramStruct);

end