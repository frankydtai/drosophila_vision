function convProtSt = convMovBarData2minMovBar(movBarProtSt, relOrt)

% function convMovBarData2minMovBar(movBarProtSt, relOrt)
% 
% this function is designed to extract the relevant direction from movbar
% protocol and treat it as an additional minNov trajectory in the minimal
% moving protocol. For cells that don't have minMovbar 
%
% INPUT 
% mavBar -                      protocol structure from mov/minMovBar
%                               protocols 
% relOrt -                      relevant orientation to extract from
%                               movBar. Should be single number between 0-3
%
% OUTPUT
% similar in strucutre to a minMovBar protocol. To mark the original moving
% bar stim, it is given a pairInd value that is +2 of the last one (in
% gratingTable)


convProtSt = movBarProtSt; 

assert(ismember(relOrt, 0:3), 'relOrt is out of range')

mvSubProt = parseStructByTable(movBarProtSt, ['ismember(xTab.orient, [ ' num2str([relOrt, relOrt+4]), ' ] )']);
mvRelTab = mvSubProt.gratingTable; 

lastIdx = 0;
lastPI = 0;

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
finTab = addTab;

addRelInds = ones(subTabH, 4);
addRelInds(:,1) = addTab.index;

convProtSt.gratingTable = finTab;


% creating new relInds (different than usual since only subset of stim was
% chosen)

numStimMM = 0;
numStimMV = length(mvSubProt.stim);

for ii=1:numStimMV 
    currInd = mvSubProt.stim(ii).relInds(1);
    tempInd = find(mvRelTab.index == currInd);
    mvSubProt.stim(ii).relInds = [tempInd+lastIdx, 1,1,1];
end


convProtSt.stim(numStimMM+1:numStimMM+numStimMV) = mvSubProt.stim;
convProtSt.stim(numStimMM+numStimMV+1:end) = [];



end