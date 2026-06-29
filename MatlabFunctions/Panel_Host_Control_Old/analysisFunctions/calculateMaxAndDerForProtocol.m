function meanProtDat = calculateMaxAndDerForProtocol(pStruct, pStTable)

% function calculateMaxAndDerForProtocol(pStruct, pStTable)
%
% This function is designed to work with protocols that have all 8
% orientations and several different gratings. It will align the data in 
% protocol to the 'center' or 'move' positions given by the table, 
% calculate mean response traces, and from that (based on appear and disappear frames)
% will calculate max response and max derivative. 
%
% INPUT
%
% pStruct -         protocol structure from a protocol that presented
%                   several different grating stimuli in 8 orientations
%
% pStTable -        protocol table as described in expandStimTable. in the vairable description of the table one var should be 
%                   defined as 'zeroVar' and this will be used to zero to . Also must
%                   have an 'appear' and 'disappear' columns to mark the edges of where the
%                   response will be calculated in. 
%                   Must also have a stimInd variable as str (can have nans
%                   in it)
%
% OUTPUT
%
% meanProtDat -     numGrt X numOrt structure with 2 fields in each 
%   .mean -         field that is a result from getAlignedStimData
%   .result -       field that is a result from calculateMaxRespAndDer
%   .table -        row from the full table that corresponds to that stim
%                   (in table format)
%
% Both have additional subfields within them


assert(isfield(pStruct.stim, 'data'), 'pStruct is missing data field')
assert(length(pStruct.orientations) ==8, 'pStruct must have 8 orientations') % not sure this is necessary


assert(istable(pStTable), 'pStTable must be a table')

varNames = pStTable.Properties.VariableNames;

assert(ismember('appear', varNames), 'pStTable is missing appear variable')
assert(ismember('disappear', varNames), 'pStTable is missing disappear variable')

assert(ismember('zeroVar', pStTable.Properties.VariableDescriptions), 'table is missing zeroVar')
cenInd = strcmpi(pStTable.Properties.VariableDescriptions, 'zeroVar');

%assert(sum(ismember({'center', 'move'}, varNames)) > 0, 'pStTable is missing move or center')

fullTab = expandStimTable(pStruct, pStTable);
numStim = height(fullTab);

cenPosInd = fullTab{:, cenInd};

% if ismember('center', varNames)
%     cenPosInd = fullTab.center;
% elseif ismember('move', varNames)
%     cenPosInd = fullTab.move;
% else
%     error('pStTable is missing either center or move variable')
% end



for ii=1:numStim
    
    relInd = str2num(fullTab.stimInd{ii});
    
    tempMeanSt = getAlignedStimData(pStruct, relInd, cenPosInd(ii));
    appFrame = fullTab.appear(ii);
    disFrame = fullTab.disappear(ii);
    stInd = tempMeanSt.meanPos(tempMeanSt.meanPos(:, 2) == appFrame, 1);
    spInd = tempMeanSt.meanPos(tempMeanSt.meanPos(:, 2) == disFrame, 1);
    
    maxRes = calculateMaxRespAndDer(tempMeanSt.mean, stInd, spInd);
    
    meanProtDat(relInd(1), relInd(3)).mean = tempMeanSt;
    meanProtDat(relInd(1), relInd(3)).result = maxRes;
    meanProtDat(relInd(1), relInd(3)).table = fullTab(ii, :);
    
end

% In case not all the protocol was used as input and relInds do not
% correspond with order (there are gaps)

fullInd = ~cellfun(@isempty, {meanProtDat(:, 1).mean});
meanProtDat = meanProtDat(fullInd, :);


end
