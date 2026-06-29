function newArenaIm = placeBarInArena(arenaIm, barSt)

% function newArenaIm = placeBarInArena(arenaIm, barSt)
%
% This function adds a bar to a background image, based on the specs in the
% bar structure. 
%
% INPUT
%
% arenaIm -         2D current image of the background with no bar (background could
%                   be simply gray, noise, other bars, etc.)
% barSt -           (optional) structure that specifies bar characteristics. Should have the
%                   following fields:
%   center -        1X2 vector that is within the range set by size(arenaIm)
%   size -          1X2 vector specifying x and y dimensions of the bar (in arena pixels). 
%   proportion -    1X3 vector for the proportion of min, mean, and max pixels
%                   within the bar. Numbers in vector are normalized to the sum (i.e. 1,1,2 would be 1/4, 1/4 and 1/2).
%   range -         1X2 vector specifying min and max values
%   arrangement -   string. Optional values are 'random', 'verticalLH',
%                   'horizontalLH', 'verticalHL', 'horizontalHL'. LH and HL
%                   mean Low then High or vice versa
%
% NOTE! if certain fields are not given they will be complemented with
% default values (see below)
%
% OUTPUT
% newArenaIm -      same dimensions as arenaIm only with bar place in it
%                   (overwriting previous values at that position)




arenaSize = size(arenaIm);

defaultBarSt = generateDefaultBarSt(arenaSize);

namesCell = fieldnames(defaultBarSt);

% populate and check the bar structure
if nargin == 2
    cellCount = 0;
    for ii=1:length(namesCell)
        if isfield(barSt, namesCell{ii})
            cellCount = cellCount+1;
            tempVal = getfield(barSt, namesCell{ii});
            relevantCheck(ii, tempVal, arenaSize, defaultBarSt)
            defaultBarSt = setfield(defaultBarSt, namesCell{ii}, tempVal);
        end
    end
    if cellCount ~= length(fieldnames(barSt))
        warning('fields skipped in barSt since they did not match default structure')
    end
end
    
meanVal = floor(mean(arenaIm(:)));
barTemplate = ones(defaultBarSt.size)*meanVal;
minMaxVals = defaultBarSt.range;

% generates the bar
switch defaultBarSt.arrangement
    case 'random'
        if ~isempty(defaultBarSt.seed)
            rng(defaultBarSt.seed)
        end
        totNum = prod(defaultBarSt.size);
        propV = prop2Vals(defaultBarSt.proportion, totNum);
        minInd = randsample(totNum, propV(1));
        restInd = setdiff(1:totNum, minInd);
        maxInd = randsample(restInd, propV(3));
        barTemplate(minInd) = minMaxVals(1);
        barTemplate(maxInd) = minMaxVals(2);
    case 'verticalLH'
        totNum = defaultBarSt.size(1);
        propV = prop2Vals(defaultBarSt.proportion, totNum);
        barTemplate(1:propV(1), :) = minMaxVals(1);
        barTemplate(end - propV(3)+1:end, :) = minMaxVals(2);
    case 'verticalHL'
        totNum = defaultBarSt.size(1);
        propV = prop2Vals(defaultBarSt.proportion, totNum);
        barTemplate(1:propV(1), :) = minMaxVals(2);
        barTemplate(end - propV(3)+1:end, :) = minMaxVals(1);
    case 'horizontalLH'
        totNum = defaultBarSt.size(2);
        propV = prop2Vals(defaultBarSt.proportion, totNum);
        barTemplate(:, 1:propV(1)) = minMaxVals(1);
        barTemplate(:, end - propV(3)+1:end) = minMaxVals(2);
    case 'horizontalHL'
        totNum = defaultBarSt.size(2);
        propV = prop2Vals(defaultBarSt.proportion, totNum);
        barTemplate(:, 1:propV(1)) = minMaxVals(2);
        barTemplate(:, end - propV(3)+1:end) = minMaxVals(1);
end

% position it in the arena

leftBotCor = defaultBarSt.center - floor(defaultBarSt.size/2);
arenaIm(leftBotCor(1):leftBotCor(1)+defaultBarSt.size(1)-1, ...
        leftBotCor(2):leftBotCor(2)+defaultBarSt.size(2)-1) = barTemplate;
    
newArenaIm = arenaIm;

end

%%

function relevantCheck(fieldNum, relVal, arenaSize, defaultSt)

% This subfunction performs the relevant QC on the input structure
% based on the order center, size, proportion, range, arrangement


switch fieldNum
    case 1
        assert(isvector(relVal), 'center should be a 1X2 vector')
        assert(length(relVal) == 2, 'center should be a 1X2 vector')   
        assert(relVal(1) < arenaSize(1), 'X center is out of range')
        assert(relVal(2) < arenaSize(2), 'X center is out of range')
    case 2
        assert(isvector(relVal), 'size should be a 1X2 vector')
        assert(length(relVal) == 2, 'size should be a 1X2 vector')
        assert(defaultSt.center(1) + relVal(1)/2 < arenaSize(1), 'bar x dim is out of arena')
        assert(defaultSt.center(1) - relVal(1)/2 > 0, 'bar x dim is out of arena')
        assert(defaultSt.center(2) + relVal(2)/2 < arenaSize(2), 'bar y dim is out of arena')
        assert(defaultSt.center(2) - relVal(2)/2 > 0, 'bar y dim is out of arena')
    case 3
        assert(isvector(relVal), 'proportion should be a 1X3 vector')
        assert(length(relVal) == 3, 'proportion should be a 1X3 vector')
        assert(prod(relVal >= 0) ==1, 'proportion specs should be non-negative')
        assert(sum(relVal > 0) > 0, 'proportion specs should have at least one positive number')
    case 4
        assert(isvector(relVal), 'range should be a 1X2 vector')
        assert(length(relVal) == 2, 'range should be a 1X2 vector')
        assert(prod(ismember(relVal, 0:7)) == 1, 'range values should be between 0-7')
    case 5
        assert(ischar(relVal), 'arrangement should be a string with values: random, vertical or horizontal')
        assert(ismember(relVal, {'random', 'verticalLH', 'horizontalLH', 'verticalHL', 'horizontalHL'}), ...
               'arrangement should be a string with values: random, verticalLH or horizontalLH or HL')
end


end



%% converts the proportion to actual pixels/rows/columns

function pixVals = prop2Vals(propInput, divRange)

% divRange is the number to divide based on the proportions (total #
% pixels,  rows or columns)

propBase = (propInput/sum(propInput)) * divRange;
propF = floor(propBase);
propDiff = propBase - propF;
[~, propDiffOrd] = sort(propDiff, 'descend');

vals2Add = divRange - sum(propF);

for ii=1:vals2Add
    tempInd = propDiffOrd(ii);
    propF(tempInd) = propF(tempInd)+1;
end

pixVals = propF;


end






