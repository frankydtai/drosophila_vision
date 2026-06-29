function checkCrudeSpikeDetction(pStruct, relInds)


% function checkCrudeSpikeDetction(pStruct, stimNum)
%
% This function is designed to plot raster and voltage trace from the same
% stimulus response to validate the spike identification procedure.
% It uses relInds and not stimNum to plot all the repeats from a single
% stimulus
%
% INPUT
% pStruct -     protocolStruct with results in the stim field (in .data)
% relInds -     1X4 vector of relInds from .stim in the structure

allInds = vertcat(pStruct.stim.relInds);

relStimI = ismember(allInds, relInds, 'rows');

assert(sum(relStimI) > 0, 'no relInds in the structure match the given input')

relStim = pStruct.stim(relStimI);

posCell = generatePositionCell(0.05, 0.95, 0.05, 0.95, 0.02, 0.04, [length(relStim), 2]);
axh = gobjects(size(posCell));


for ii=1:length(relStim)
    
    relDat = relStim(ii).data{1}(:,3)*10;
    relTime = relStim(ii).data{1}(:,1);
    
    spikeRas = findSpikesCrude(relDat);
    
    axh(ii,1) = axes('position', posCell{ii, 1});
    plot(relTime, spikeRas)
    
    axh(ii,2) = axes('position', posCell{ii, 2});
    plot(relTime, relDat)
    
    
end


end