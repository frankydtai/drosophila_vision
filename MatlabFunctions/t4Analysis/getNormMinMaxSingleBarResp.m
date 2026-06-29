function maxMinCell = getNormMinMaxSingleBarResp(allSingleBarTable)

% function getNormMinMaxSingleBarResp(allSingleBarTable)
%
% this function uses allSingleBarTable to generate a normalized max and min
% response for all cells by positions (normalized within a duration and
% summed between durations)
%
% relDurInd is used internally to exclude the short durations
%
% INPUT
% allSingleBarTable -       generated using extractFitParameterFromSingleBarFit
%
% OUTPUT
% maxMinCell -              cell array with PosX2 array in for each T4 cell. column 1 is maxResp
%                           and column 2 is abs value of minREsp

allNum = unique(allSingleBarTable.cellNum);
allDur = unique(allSingleBarTable.duration);

relDurInd = allDur > 40;

ort = nan(1,length(allNum));

normMaxResp = cell(1,length(allNum));
normMinResp = cell(1,length(allNum));

%allNMinResp = cell(1,length(allNum));
%allNMaxResp = cell(1,length(allNum));

for ii=1:length(allNum)
    
    relTab = allSingleBarTable(allSingleBarTable.cellNum == allNum(ii), :);
    ort(ii) = relTab.orient(1); %since they are identical
    allPos = unique(relTab.position);
    
    maxResp = zeros(length(allPos),length(allDur));
    minResp = zeros(length(allPos),length(allDur));
    
    for jj=1:length(allDur)
    
        relInds  = relTab.duration == allDur(jj);
        
        tempMaxVal = relTab.maxVal(relInds);
        tempMinVal = relTab.minVal(relInds);
        
        maxResp(:, jj) = tempMaxVal / max(tempMaxVal); % put more weight to the longer stim
        minResp(:, jj) = tempMinVal / min(tempMinVal); 
        
    end
    
    %normResp{ii} = intResp;
    normMaxResp{ii} = nansum(maxResp(:,relDurInd), 2);
    normMinResp{ii} = nansum(abs(minResp(:,relDurInd)), 2);
    
    %allNMaxResp{ii} = nansum(maxResp, 2)/2; % seems like looking at the last 2 is preferable
    
end


maxMinCell = cellfun(@(x,y) [x,y], normMaxResp, normMinResp, 'uniformoutput', 0);

end
