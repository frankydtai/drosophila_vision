function playByStimInds(protStruct, desiredStim)

% function playByStimInds(protStruct, desiredStim)
%
% function feeds the output of the first inds in getStimInds into playStimwImplay
% 

stimInds = getStimInds(protStruct, desiredStim);

for ii=1:length(stimInds)
    
    relI = stimInds(ii).inds(1);
    
    playStimwImplay(protStruct.stim(relI).matCell)
    
end


end