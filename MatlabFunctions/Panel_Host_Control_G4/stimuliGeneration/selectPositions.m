
function [selectedPos, selPosInd] = selectPositions(posRange, recFlag)

% function selectedPos = selectPositions(posRange)
%
% This function creates a gui for clicking on all the possible pairwise
% combinations of position from posRange. 
%
% INPUT
%
% posRange -            1XP vector of relevant positions
% recFlag -             (optional) logical. If TRUE output the reciprocal positions
%                       also
% OUTPUT
% selectedPosition -    NX2 matrix of clicked-on combinations

assert(isvector(posRange), 'posRange should be a 1XP vector')
posRange = sort(posRange);

if nargin < 2
    recFlag = 0;
else
    assert(ismember(recFlag, [0,1]), 'recFlag should be logical')
end

posCell = generatePositionCell(0.05, 0.975, 0.025, 0.95, 0.02, 0.02, [length(posRange)-1,length(posRange)-1]);

fH = figure('position', [600, 150, 900, 800]);
pnlH = uipanel(fH,'Title','Select Positions','FontSize',14);
               


count = 0;
for ii=1:length(posRange)
    
    for jj=ii+1:length(posRange)
        
        count= count+1;
        
        relStr = num2str([posRange(ii), posRange(jj)]);
        
        ckH(count) = uicontrol(pnlH, 'Style', 'togglebutton', 'String', relStr,...
                               'units', 'normalized', 'Position', posCell{ii, jj-1}, ...
                                'Callback', '', 'fontSize', 14, 'Tag', 'chkBox');  
    
    end
    
end

bH = uicontrol(pnlH, 'Style', 'pushbutton', 'String', 'DONE',...
               'units', 'normalized', 'Position', posCell{end, 1}, ...
               'fontSize', 10, 'Callback', @getPos);  


fH.Visible = 'on';

uiwait

outPos = bH.UserData;
outPos = sortrows(outPos);
outPosInd = (1:size(outPos,1))';

if recFlag
    tempPos = fliplr(outPos);
    outPos = vertcat(outPos, tempPos);
    outPosInd  = vertcat(outPosInd, outPosInd);
end

% % orginaizing the positions nicely
% outPos(:,3) = outPos(:,2) - outPos(:,1);
% sortedOP = sortrows(outPos, 3);
% selectedPos = sortedOP(:,1:2);

selectedPos = outPos;
selPosInd  = outPosInd;

close(fH)

end


function getPos(hObject,eventdata)
        
        intChkH = findobj('Tag', 'chkBox');
        relChkH = findobj(intChkH, 'Value', 1);
        
        uDat = zeros(length(relChkH), 2);
        
        for ii=1:length(relChkH)
            tempStr = relChkH(ii).String;
            uDat(ii, :) = str2num(tempStr);
        end
        
        hObject.UserData = uDat;
        uiresume
end



