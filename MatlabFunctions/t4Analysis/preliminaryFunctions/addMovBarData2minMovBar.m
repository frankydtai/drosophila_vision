function combProtSt = addMovBarData2minMovBar(movBarProtSt, minMovBarProtSt)

% function addMovBarData2minMovBar(movBarProtSt, minMovBarProtSt)
% 
% this function is designed to extract the relevant direction from movbar
% protocol and treat it as an additional minNov trajectory in the minimal
% moving protocol. 
%
% INPUT 
% mavBar/minMOvBarProtSt -      protocol structure from mov/minMovBar
%                               protocols 
%
% OUTPUT
% similar in strucutre to a minMovBar protocol. To mark the original moving
% bar stim, it is given a pairInd value that is +2 of the last one (in
% gratingTable)


combProtSt = minMovBarProtSt;

mmTab = minMovBarProtSt.gratingTable; 

relOrt = minMovBarProtSt.inputParams.orientations; 

mvSubProt = parseStructByTable(movBarProtSt, ['ismember(xTab.orient, [ ' num2str([relOrt, relOrt+4]), ' ] )']);
mvRelTab = mvSubProt.gratingTable; 

lastIdx = mmTab.index(end);
lastPI = max(mmTab.pairInd);

relSp = mvRelTab.span(1);
if rem(relOrt,2) % to correct for diagonal movement
    relSp = 2*round(relSp/sqrt(2))+1;
end

relPos = floor(relSp/2);

subTabH = height(mvRelTab);

pairInd = ones(subTabH, 1) * (lastPI+2);
direction = zeros(subTabH, 1);
startPos = direction; stopPos = direction; index = direction; oldIdx = direction;


% creating new table
count = lastIdx;
for ii=1:subTabH 
    count=count+1;
    index(ii) = count;
    
    if mvRelTab.orient(ii) < 4
        direction(ii) = 1;
        startPos(ii) = -relPos;
        stopPos(ii) = relPos; 
    else
        direction(ii) = -1;
        startPos(ii) = relPos;
        stopPos(ii) = -relPos; 
    end
    
end
    
addTab = [mvRelTab(:, {'stepDur'; 'appear'; 'disappear'; 'framePerStep'}), table(index, direction, pairInd, startPos, stopPos)];
finTab = vertcat(mmTab, addTab);

addRelInds = ones(subTabH, 4);
addRelInds(:,1) = addTab.index;

combProtSt.gratingTable = finTab;


% creating new relInds (different than usual since only subset of stim was
% chosen)

numStimMM = length(minMovBarProtSt.stim);
numStimMV = length(mvSubProt.stim);

for ii=1:numStimMV 
    currInd = mvSubProt.stim(ii).relInds(1);
    tempInd = find(mvRelTab.index == currInd);
    mvSubProt.stim(ii).relInds = [tempInd+lastIdx, 1,1,1];
end


combProtSt.stim(numStimMM+1:numStimMM+numStimMV) = mvSubProt.stim;



end