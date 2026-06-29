function gtFrame = generateRandomDotFrame(gtStruct)

% This function uses the fields from gtStruct to generate the required
% frame (uneven due to rotation symmetry). 
%
% INPUT
% gtSeqStruct - structure with the following fields (all fields should be
% single values).
% .dotSize  -           	single number. size of each random dot in pixel 
% .propON(OFF) -            fraction. Proportion of dots that will be
%                           ON(OFF). Rest would be background level
% .valON(OFF) -             fraction (optional). Values to use for ON(OFF)
%                           dots. If not given 1(0) is used. 
% .gsLevel -                (optional) number of bit on the gray scale level (default 3)
% .bkgdVal -                (optional) value for the rest of the frame.
%                           Defualt is 0.49
% .relFrameSize -           (optional) size of matrix in which to generate
%                           the random pattern (can speed up performance).
%                           Shoud be a single odd number (odd is important for matrix rotation. 
%                           if not given the entire matSize is taken
% .rngSeed -                To be used when reproducible frames are desired
%                           (if not given a time stamp would be assigned)
%
% OUTPUT 
% gtFrame -     a 225X225 matrix to be used with the relevant masks
%
% Note! to reach the middle value (3 fro gs3 and 7 for 4) values should be
%       slightly below 0.5 (since it is rounded up). 
%
% NOte! if only one type of dot is wanted, prop should be set to 0 (of val
%       is equal to bkgd it will generate an error)


%% INPUT managment

% size of the total image
matSiz = 225;

if isfield(gtStruct, 'gsLevel')
    gsLevel = gtStruct.gsLevel;
    assert(isvector(gsLevel), 'gsLevel should be a single number')
    assert(length(gsLevel) == 1, 'gsLevel should be a single number')
else
    gsLevel = 3;
end

if isfield(gtStruct, 'valON')
    valON = gtStruct.valON;
    assert(isvector(valON), 'valON should be a single number')
    assert(length(valON) == 1, 'valON should be a single number')
else
    valON = 1;
end

if isfield(gtStruct, 'valOFF')
    valOFF = gtStruct.valOFF;
    assert(isvector(valOFF), 'valOFF should be a single number')
    assert(length(valOFF) == 1, 'valOFF should be a single number')
else
    valOFF = 0;
end

if isfield(gtStruct, 'bkgdVal')
    bkgdVal = gtStruct.bkgdVal;
    assert(isvector(bkgdVal), 'bkgdVal should be a single number')
    assert(length(bkgdVal) == 1, 'bkgdVal should be a single number')
else
    bkgdVal = 0.49;
end

if isfield(gtStruct, 'relFrameSize')
    relSiz = gtStruct.relFrameSize;
    assert(relSiz <= matSiz, 'relSiz can not be bigger than matSiz')
    assert(rem(relSiz, 2) == 1, 'relSiz should be an odd number')
else
    relSiz = matSiz;
end

if isfield(gtStruct, 'rngSeed')
    seed = gtStruct.rngSeed;
    assert(isvector(seed), 'seed should be a single number')
    assert(length(seed) == 1, 'seed should be a single number')
else
    tt = clock;
    seed = tt(5) * tt(6);
end


assert(valON <= 1, 'ON value can not exceed 1')
assert(valOFF >= 0, 'OFF value can not be smaller than 0')
assert(valOFF < bkgdVal && bkgdVal < valON, 'values must conform to OFF < bkgd < ON')

propON = gtStruct.propON;
propOFF = gtStruct.propOFF;
dotSize = gtStruct.dotSize-1;
assert(dotSize >= 0, 'dotsize should be a positive number') % zero here would mean a dot of 1 pixel

assert(propON >=0 && propON <= 1, 'ON proportioo should be between 0 and 1')
assert(propOFF >=0 && propOFF <= 1, 'OFF proportioo should be between 0 and 1')
assert(propON+propOFF <= 1, 'sum of propON and OFF can not exceed 1')


if dotSize > 10
    warning('ATTENTION: dot size very large')
end

%%

maxVal = 2^gsLevel - 1;


onNum = round(valON*maxVal);
offNum = round(valOFF*maxVal);
bkgdNum = round(bkgdVal*maxVal);

propNumON = round(relSiz^2 * propON);
propNumOFF = round(relSiz^2 * propOFF);


tempFrame = ones(relSiz, relSiz) * bkgdNum;

rng(seed)

onInds = randsample(relSiz^2, propNumON);
[onSub1, onSub2] = ind2sub([relSiz, relSiz], onInds);
allOnSubs = horzcat(onSub1, onSub2, ones(length(onSub1),1));
offInds = randsample(relSiz^2, propNumOFF);
[offSub1, offSub2] = ind2sub([relSiz, relSiz], offInds);
allOffSubs = horzcat(offSub1, offSub2, zeros(length(offSub1),1));

allSubs = vertcat(allOnSubs, allOffSubs);
permInds = randperm(size(allSubs, 1));
allSubs = allSubs(permInds, :);

for ii=1:length(allSubs)
    tempSub1 = intersect(allSubs(ii,1):allSubs(ii,1)+dotSize, 1:relSiz);
    tempSub2 = intersect(allSubs(ii,2):allSubs(ii,2)+dotSize, 1:relSiz);
    [tempComb1, tempComb2] = meshgrid(tempSub1, tempSub2);
    if allSubs(ii,3) == 0
        relNum = offNum;
    else
        relNum = onNum;
    end
    tempFrame(tempComb1(:), tempComb2(:)) = relNum;
end


midPoint = ceil(matSiz/2);
halfWin = floor(relSiz/2);

gtFrame = ones(matSiz, matSiz) * bkgdNum;
gtFrame(midPoint-halfWin:midPoint+halfWin, midPoint-halfWin:midPoint+halfWin) = tempFrame;

% This allows createProtocol to distinguish between 0 create from mask and rotation
% and 0 from the pattern. Should be dealt with in createProtocol
gtFrame(gtFrame == 0) = 0.001;







end