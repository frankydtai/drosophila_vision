function subStruct = parseStructByTable(protStruct, stringCond)

% function subStruct = parseStructByTable(protStruct, stringCond)
%
% This function uses the grating talbe to prase protStruct into a smaller
% structure, based in the given logical condition. 
% To be used mainly with protocols in which the different stimuli are
% generateed via changes in gratings parameters.
% Uses the gratingTable index varialbe to reference gratings in stim.relInds
%
% INPUT 
% protStruct -      structure with gratingTable, stim and stim.relInds fields
% stringCond -      logical condition given as a string. For details see
%                   'logicalTableParsing'
% OUTPUT
% subStruct -       sub-structure with the desired gratings


assert(isfield(protStruct, 'gratingTable'), 'protStruct is missing gratingTable field')
assert(isfield(protStruct, 'stim'), 'protStruct is missing stim field')
assert(isfield(protStruct.stim, 'relInds'), 'stim is missing relInds field')

allInds = vertcat(protStruct.stim.relInds);

subStruct = protStruct;

subGratTab = logicalTableParsing(protStruct.gratingTable, stringCond); %stringCond is checked within this function

relGratInds = subGratTab.index;

subStruct.gratingTable = subGratTab;

relStimInds = ismember(allInds(:,1), relGratInds);

subStruct.stim = protStruct.stim(relStimInds);


end
