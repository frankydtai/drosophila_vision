function fullStimTable = expandStimTable(pStruct, stimTable)

% function fullStimTable = expandStimTable(pStruct, stimTable)
%
% This function is meant to expand a stimTable into a fullStimTable in
% which every stimulus has an explicit description. Practically, it takes
% rows in the table that were defined as 'nan' in one of their indices and
% expands to all the relevant values based on pStruct
% 
%
% INPUT 
%
% pStruct -     protocolStruct with all the required fields (output of
%               runPosFuncProtocol). For this function only .stim.relInds
%               is necessary (or if combProtocol then .stim.combInds
%
% stimTable -   stimulus table with a description for each stim. 
%               Table must have column name stimInd corresponding to stimInds and
%               column with relevant frames for analysis
% For example:
%
%       stimInd         indDisc      cenAppear    cenDisappear    surrAppear    surrDisappear
%     _____________    ___________    _________    ____________    __________    _____________
% 
%     '1 0 1 0'        'justCen'        5            6             NaN           NaN          
%     '2 0 0 nan'      'justSurr'     NaN          NaN               5             6          
%     '3 1 nan nan'    'surrFirst'      6            7               5             6          
%     '3 2 nan nan'    'sim'            5            5               5             5          
%     '3 3 nan nan'    'cenFirst'       5            6               6             7        
%  
% All nans in the table will be exanded to appropriate full relInds. 
% Table should be generated manually for each protocolStruct
% (plotStimCorrelations can help)
%
%
% OUTPUT
%
% fullStimTable -   same as stimTable only expanded for all the nans

allNames = stimTable.Properties.VariableNames;

assert(sum(strcmp('stimInd', allNames)) == 1, 'stimTable is missing stimInd column (case sensitive)')

fullStimTable = stimTable;
fullStimTable(1:end, :) = [];

for ii=1:height(stimTable)
    
    currRow = stimTable(ii, :);
    currInd = str2num(currRow.stimInd{1});
    
    if sum(isnan(currInd))
        
        relInds = getStimInds(pStruct, currInd);
        
        for jj=1:length(relInds)
            
            currRow.stimInd = num2str(relInds(jj).val);
            
            fullStimTable = [fullStimTable; currRow];
        end
        
    else
        
        fullStimTable = [fullStimTable; currRow];
        
    end
    
end




end