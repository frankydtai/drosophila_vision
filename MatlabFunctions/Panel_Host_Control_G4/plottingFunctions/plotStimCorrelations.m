
function varargout = plotStimCorrelations(pStruct, tableVar)

% function plotStimCorrelations(pStruct)
%
% This function plots the correlations between the individual frames of
% each unique stimulus in the structure pStruct. The idea is to used it for
% a quick visual way of generating aoVec that will correspond with the
% stimuli. 
%
% INPUT
%
% pStruct -         structure after it contains the .stim fields (after it has
%                   passed through the relevant createXXXProtocol function
%
% tableVar-         (optional). variable name from gratingTable that will be displayed
%                   on the correlation image

if nargin<2
    tableVar = {}; 
else
    assert(isfield(pStruct, 'gratingTable'), 'structure is missing gratingTable field')
    assert(ismember(tableVar, pStruct.gratingTable.Properties.VariableNames), 'No %s variable in gratingTable')
    tableVals = pStruct.gratingTable{:, tableVar};
end


assert(isfield(pStruct, 'stim'), 'Structure is missing .stim field') 

% geting all the unique stimuli (getting rid of repeats)
if isfield(pStruct.stim, 'relInds')
    tempInds = vertcat(pStruct.stim.relInds);
elseif isfield(pStruct.stim, 'combInds')
    tempInds = vertcat(pStruct.stim.combInds);
else
    error('structure does not contain recognized inds field')
end

[~, stimInd, ~] = unique(tempInds, 'rows');
numStim = length(stimInd);
corrRes = cell(1,numStim); % used cell since stim might be of different length

for ii=1:numStim
    tempStim = pStruct.stim(stimInd(ii)).matCell;
    corrRes{ii} = zeros(1, size(tempStim,3));
    for jj=2:size(tempStim, 3)
        corrRes{ii}(jj) = corr2(tempStim(:,:,jj-1), tempStim(:,:,jj));
    end
end
        
allLen = cellfun(@length, corrRes);
corrResMat = nan(numStim, max(allLen));

for ii=1:numStim
    corrResMat(ii, 1:allLen(ii)) = corrRes{ii};
end

figure

imagesc(corrResMat, [-1, 1])
if isfield(pStruct, 'generalFrequency')
    protFreq = num2str(pStruct.generalFrequency);
elseif isfield(pStruct, 'inputStruct')
    protFreq = num2str(pStruct.inputStruct.generalFrequency);
else
    protFreq = 'Not found';
end

title(['genFreq - ', protFreq])

if ~isempty(tableVar)
    for ii=1:length(tableVals)
        text(tableVals(ii)+1, ii, '|', 'color', 'r', 'fontsize', 20) % added one to account for starting in 1 in matlab
    end
    
end


axh = gca;

if nargout ==1
    varargout{1} = axh;
end


end