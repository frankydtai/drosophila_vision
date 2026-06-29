function aoVec = makeAOVecForLED(pulseStruct)

% function aoVec = makeAOVecForLED(pulseStruct)
% 
% This function create a vector that can be use as an input to
% makeVSDAOfunction_image. It is designed specifically to generate TTL
% pulses for an LED controller and therefore requires only timing input. 
% Since the AO functionality in the panel controller is limited to 1KHz,
% this function also assumes it. If Structure is not given a default one is
% use
%
% INPUT
% 
% pulseStruct -     structure with the following fields
%   .numPulse -     number of pulses in the vector
%   .pulseWid -     pulse width. Should be given in ms
%   .ipi      -     inter pulse interval (also in ms)
%   .firstPulse -   timing of first pulse (so that it could be sync w/
%                   posFunc and the pattern display). Also in ms
%                   (practically means #0 before the starting pulse)
%   .padEndLen -    Number of zeros to add in the end of the pulse train
%                   (since the panel controller plays the AO file in a cyclic manner)
%   .amp -          pulse amplitude (to be used as non-TTL input)
%
% OUTPUT
% 
% aoVec -           1XN vector of 0s and 5s to generate the desired pulse
%                   train (should be fed into makeVSDAOfunction_image to generate file)

if nargin <1 
    pulseStruct = generateDefaultPulseStruct;
end

% verifying input is correct
assert(isfield(pulseStruct, 'numPulse'), 'structure missing numPulse field')
numP = pulseStruct.numPulse;
assert(length(numP) == 1 && numP > 0, 'numPulse should be a single positive number')
assert(isfield(pulseStruct, 'pulseWid'), 'structure missing pulseWidfield')
pWid = pulseStruct.pulseWid;
assert(length(pWid) == 1 && pWid > 0, 'pulseWid should be a single positive number')
assert(isfield(pulseStruct, 'ipi'), 'structure missing ipi field')
ipi = pulseStruct.ipi;
assert(length(ipi) == 1 && ipi > 0, 'ipi should be a single positive number')
assert(isfield(pulseStruct, 'firstPulse'), 'structure missing firstPulse field')
firP = pulseStruct.firstPulse;
assert(length(firP) == 1 && firP > 0, 'firstPulse should be a single positive number')
assert(isfield(pulseStruct, 'padEndLen'), 'structure missing padEndLen field')
padE = pulseStruct.padEndLen;
assert(length(padE) == 1 && padE > 0, 'padEndLen should be a single positive number')
assert(isfield(pulseStruct, 'amp'), 'structure missing padEndLen field')
amp = pulseStruct.amp;
assert(length(amp) == 1 && amp > 0, 'amplitude should be a single positive number')

%constructing the vector
vecSt = zeros(1, firP-1);

pTrain = repmat([ones(1, pWid)*amp, zeros(1, ipi)], 1, numP);
pTrain = pTrain(1:end-ipi); %get rid of last ipi that is not followed by pulse

vecEnd = zeros(1, padE);

aoVec = [vecSt, pTrain, vecEnd];


end