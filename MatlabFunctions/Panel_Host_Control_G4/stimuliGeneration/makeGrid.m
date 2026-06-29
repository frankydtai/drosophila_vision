function maskPos = makeGrid(paramStruct)

% function maskPos = makeGrid(paramStruct)
%
% This function generates  a NX2 matrix of x,y positions for a grid in
% arena spatial coordinates (bottom-left pixel is <1,1>). input checking is done
% already in the higher function (generateMaskPositions). The function does
% not transforms spatial to controller coordinates (done in
% getArenaMaskTransform)
%
% INPUT
% paramStruct - structure. should contain the fields:
%   .gridSize       (1X2 vector)
%   .startPos       (bottom-left corner of the grid - 1X2 vector)
%   .spacing        (spacing between adject points, on both X and Y dimensions 1X2 vector)
%   .arenaSize      (optional). If not given the default is [96,32]
%
% OUTPUT
% maskPos - NX2 matrix of positions in arena coordinates


if isfield(paramStruct, 'arenaSize')
    arenaSiz = paramStruct.arenaSize;
else
    arenaSiz  = [192, 48];
end

relSpaceX = paramStruct.spacing(1);
relSpaceY = paramStruct.spacing(2);
unbreakFlag = 1;

while unbreakFlag

    % indexing is off on purpose to convert spatial to matlab arena coordinates
    xCrds = paramStruct.startPos(1):relSpaceX:(paramStruct.startPos(1)+relSpaceX*paramStruct.gridSize(1)-1);
    yCrds = paramStruct.startPos(2):relSpaceY:(paramStruct.startPos(2)+relSpaceY*paramStruct.gridSize(2)-1);

    if min(xCrds) < 1
        fprintf('X position lees than 1 \n')
        [xCrds, relSpaceX, unbreakFlag] = fixCrdsORSpacing(xCrds, relSpaceX, 1);
    elseif max(xCrds) > arenaSiz(1)
        fprintf('Y position exceeds arena size\n')
        [xCrds, relSpaceX, unbreakFlag] = fixCrdsORSpacing(xCrds, relSpaceX, arenaSiz(1));
    elseif min(yCrds) < 1
        fprintf('Y position lees than 1 \n')
        [yCrds, relSpaceY, unbreakFlag] = fixCrdsORSpacing(yCrds, relSpaceY, 1);
    elseif max(yCrds) > arenaSiz(2)
        fprintf('Y position exceeds arena size\n')
        [yCrds, relSpaceY, unbreakFlag] = fixCrdsORSpacing(yCrds, relSpaceY, arenaSiz(2));
    else
        unbreakFlag = 0;
    end

end

[tempX, tempY] = meshgrid(xCrds, yCrds);

maskPos = [round(tempX(:)), round(tempY(:))];



end


%%

function [crdsVecNew, relSpaceNew, ubFlag] = fixCrdsORSpacing(crdsVec, relSpace, extVal)

% This subfunction allows the user to change grid setting to conform with
% arena. Either by changing the spacing between positions, cropping the
% positions that exceed the dimensions of the arena, or changing the values
% that exceed to max/min values

choiceFlag = 1;

while choiceFlag
    inp = input('Please choose one of the following:\n (1) Change spacing\n (2) Crop exterme values\n (3) Pin to arena edge\n');
    if ismember(inp, 1:3)
        choiceFlag = 0;
    else
        fprintf('input should be between 1 and 3 only');
    end
end

switch inp
    case 1
        fprintf('current spacing is relevant dimension is %d \n', relSpace)
        inp2 = input('Please enter new value for spacing: \n');
        relSpaceNew = round(inp2);
        crdsVecNew = crdsVec;
        ubFlag = 1;
    case 2
        if extVal == 1 %crops values lees than 1 but bigger than any other ext val
            crdsVecNew = crdsVec(crdsVec >= extVal);
        else
            crdsVecNew = crdsVec(crdsVec <= extVal);
        end
        relSpaceNew = relSpace;
        ubFlag = 0;
    case 3
        if extVal == 1 % same assumption as above
            crdsVecNew = crdsVec;
            crdsVecNew(crdsVecNew < extVal) = extVal;
        else
            crdsVecNew = crdsVec;
            crdsVecNew(crdsVecNew > extVal) = extVal;
        end
        relSpaceNew = relSpace;
        ubFlag = 0;
end



end
