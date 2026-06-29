function protStruct = addRelVarsToOldMinMovTable(protStruct)

% adds pairInd and direction variables to old minMovingBar protocols
% (generated before 20160510

relTable = protStruct.gratingTable;

allPairs = [relTable.startPos, relTable.stopPos];
uniPairs = unique(allPairs, 'rows');


uniRecPairs = [nan, nan];

for ii=1:size(uniPairs,1)
    
    if ~ismember(uniPairs(ii, :), uniRecPairs, 'rows') && ~ismember(fliplr(uniPairs(ii, :)), uniRecPairs, 'rows')
        
        uniRecPairs = vertcat(uniRecPairs, uniPairs(ii, :));
        
    end
    
end

uniRecPairs = sortrows(uniRecPairs(~isnan(uniRecPairs(:, 1)), :));

pairIndVec = zeros(height(relTable), 1);

for ii=1:size(uniRecPairs, 1)
    
    tempPair = uniRecPairs(ii, :);
    fwdInd = ismember(allPairs, tempPair, 'rows');
    bkwdInd = ismember(allPairs, fliplr(tempPair), 'rows');
    pairIndVec(logical(fwdInd + bkwdInd)) = ii;
    
end

dirVec = sign(diff(allPairs, [], 2));

relTable.pairInd = pairIndVec;
relTable.direction = dirVec;

protStruct.gratingTable = relTable;



    



end