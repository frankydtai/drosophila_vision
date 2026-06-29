function displayAllStimFromProtocol(pStruct, relRelInd, frameToShow)

% function displayAllStimFromProtocol(pStruct, relRelInd)
%
% This function plots all the stimuli from the given protocol strucure
% given the relevant relInds dimension. 
%
% INPUT
% pStruct -     protocol structure with .stim field
% relRelInd -   single value (between 1 and 4) that designates the relInds dimension along which to
%               plot.  All othere dimensions will kept at their first value.
%               dimensions are: (1)Grating (2)Mask (3)Orientation
%               (4)Position
% frameToShow = (optional) which frame of matCell of each stimulus to show.
%               if not given function will go for middle.

if nargin < 3
    frameToShow = 0;
end


assert(isfield(pStruct, 'stim'), 'stricture is missing .stim field')
assert(ismember(relRelInd, 1:4), 'relRelInd is out of range')

if isfield(pStruct.stim, 'relInds')
    allUInds = unique(vertcat(pStruct.stim.relInds), 'rows');
elseif isfield(pStruct.stim, 'combInds')
    error('function not designed for pStructComb');
else
    error('pStruct does not contain relevant inds field')
end

if isfield(pStruct, 'gsLevel') % for AO and comb protocols 
    maxVal = 2^pStruct.gsLevel-1;
    if isfield(pStruct, 'aoVec') % gets rid of the empty stim in AO protocols
        allUInds = allUInds(2:end, :);
    end
else
    maxVal = 2^pStruct.inputParams.gsLevel-1;
end


baseInds = allUInds(1,:);

% checks if relInds change together (i.e. grating w/ masks)
otherInds = setdiff(1:4, relRelInd);
addDimFlag = zeros(1,length(otherInds));
for ii=1:length(otherInds)
    if isequal(allUInds(:,relRelInd), allUInds(:, otherInds(ii)))
        addDimFlag(ii) = 1;
    end
end

relUVals = unique(allUInds(:, relRelInd));
relStimInds = zeros(length(relUVals), 4);
addDimFlagInds = find(addDimFlag);

for ii=1:length(relUVals) 
    tempBase = baseInds;
    tempBase(relRelInd) = relUVals(ii);
    
    for jj=1:length(addDimFlagInds)
        tempBase(otherInds(addDimFlagInds(jj))) = relUVals(ii); % since the only case is that they are identical
    end
    relStimInds(ii, :) = tempBase;
    
end

% checks that all the combinations created actually exist
assert(prod(ismember(relStimInds, allUInds, 'rows')) == 1, 'Some of the indices created do not exist in original protocol - function aborted');

exmpIm = cell(1, size(relStimInds,1));
for ii=1:size(relStimInds, 1)
    tempI = getStimInds(pStruct, relStimInds(ii,:));
    tempIm = pStruct.stim(tempI(1).inds(1)).matCell;
    if frameToShow
        exmpIm{ii} = tempIm(:,:,frameToShow);
    else
        numFrames = size(tempIm, 3);
        relFr = ceil(numFrames/2);
        exmpIm{ii} = tempIm(:,:,relFr);
    end
end

if length(exmpIm) > 10
    posCell = generatePositionCell(0.05, 0.95, 0.01, 0.975, -1, 0.01, 10);
    numFigs = ceil(length(exmpIm)/10);
    numAxes = 10;
else
    posCell = generatePositionCell(0.05, 0.95, 0.01, 0.975, -1, 0.01, length(exmpIm));
    numFigs = 1;
    numAxes = length(exmpIm);
end

for ii=1:numFigs
    figure('units', 'normalized', 'position', [0.4, 0.05, 0.175, 0.9])
    for jj=1:numAxes
        relInd =(ii-1)*10 + jj;
        if relInd > length(exmpIm)
            break
        end
        axes('position', posCell{jj})
        plotMidFrame2(exmpIm{relInd}, maxVal)
        th = title(num2str(relStimInds(relInd, :)));
        th.VerticalAlignment = 'middle';
    end
end
    
    
    


end
