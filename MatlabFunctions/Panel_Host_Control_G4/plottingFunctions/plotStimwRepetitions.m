function plotStimwRepetitions(pStruct, getStimIndsInput)

% function plotStimwRepatitions(pStruct, getStimIndsInput)
%
% This function plots a overlaid plot for each stimulus (all repetitions on
% one plot). It uses getStimInds for the input, and generates a figure for
% each separate stim. 
%
% INPUT
% pStruct -             protocolStruct w/ stim field and data in it
% getStimIndsInput -    input for getStimInds. 1X4 vector describing the
%                       desired relInds (for more see getStimInds)

indsSt = getStimInds(pStruct, getStimIndsInput);

close all
linCol = [1,1,1]*0.6;


for ii=1:length(indsSt)
    
    figure
    set(gcf, 'NumberTitle', 'off','Name', num2str(indsSt(ii).val))
    hold on
    for jj=1:length(indsSt(ii).inds)
        plot(pStruct.stim(indsSt(ii).inds(jj)).data{1}(:,3), 'color', linCol);
    end
end


end
