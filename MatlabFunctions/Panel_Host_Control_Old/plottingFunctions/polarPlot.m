function  varargout = polarPlot(radMat, plotOptionsSt)

% This function plot data in a polar coordinate plot. Assmues
% pi/4 steps (therefore radVec shoud have 8 points). 
%
% INPUT
%
% radMat -          NX8(max) matrix of Radi lengths (response magnitude) ordered from
%                   Orientation 0 to 7. Radi from orientation zero will be 
%                   displayed on the right, and from Ori 2 on the bottom (CW manner).
%                   This will match an outward direction of motion.   
% plotOptionSt -    optional. plotting option structure. Can include the
%                   following fields (defaults in curly brackets): 
% .type -           string. can be either 
%                   {'line'}: displays line around the circle, 
%                   'mean': arrow where the vector mean is, or 
%                   'both'.
% .normalize -      logical. Whether data will be normalized to max per row
%                   or not.
% .color -          MX3 color matrix. If not given cbrewer('qual', 'Set1') will be used.
%                   If given and M < N than will cycle through the matrix.
% .legend -         logical. if true adds a legend to the plot
% .axHand -         Optional. axes handle. If given plot rendered within
%                   the given axes
% .thetaVec -       If radMat input is not the full 8 orientations, the
%                   relevant theta vector could be supplied here.
% .maxRange -       single value. If given will determine the plot x/ylim
%                   [ -maxRange, +maxRange]
%
% OUTPUT
%
% if one output is asked it is the axes handle. if 2 it is also the legend
% handle

if nargin < 2
    plotOptionsSt = makeDefaultPolarPlotOptionsStruct;
end


matSiz = size(radMat); 


%% checking input plotting parameters

if isfield(plotOptionsSt, 'thetaVec')
    thetaVec = plotOptionsSt.thetaVec;
    assert(matSiz(2) == length(thetaVec), 'radMat and thetaVec do not have the same dimension')
    thetaVec = [thetaVec, thetaVec(1)]; % to close the circle
else
    thetaVec = [0:-pi/4:-3*pi/4, pi:-pi/4:0]; 
    assert(matSiz(2) == 8, 'if no thetaVec is given, second dimension in radMat should be equal to 8')
end



if isfield(plotOptionsSt, 'color')
    inpCol = plotOptionsSt.color;
    assert(size(inpCol, 2) == 3, 'input color field should be a MX3 matrix')
else
    inpCol = cbrewer('qual', 'Set1', matSiz(1));
end

if size(inpCol,1) < matSiz(1)
    multFac = ceil(matSiz(1)/size(inpCol,1));
    colInd = repmat(1:size(inpCol, 1), 1, multFac);
    relCol = inpCol(colInd(1:matSiz(1)), :);
else
    relCol = inpCol;
end

if isfield(plotOptionsSt, 'normalize')
    normFlag = plotOptionsSt.normalize;
    assert(ismember(normFlag, [0,1]), 'normalize should be a logical')
else 
    normFlag = 0;
end

if isfield(plotOptionsSt, 'legend')
    legFlag = plotOptionsSt.legend;
    assert(ismember(legFlag, [0,1]), 'legend should be a logical')
else
    legFlag  = 0;
end

if isfield(plotOptionsSt, 'type')
    plotType = lower(plotOptionsSt.type);
    assert(ismember(plotType, {'line', 'mean', 'both'}), 'type should be line, mean or both')
else
    plotType  = 'line';
end

if isfield(plotOptionsSt, 'axHand')
    axh = plotOptionsSt.axHand;
    axes(axh)
else
    axh = axes();
end

if isfield(plotOptionsSt, 'maxRange')
    maxVal = plotOptionsSt.maxRange;
    noMaxValFlag = 0;
else
    noMaxValFlag = 1;
end


cla

pH = get(axh, 'parent');
set(pH, 'color', [1,1,1])

%% calculating mean and max vectors

maxMat = max(radMat, [], 2);

if normFlag
    radMat = radMat./repmat(maxMat, 1, matSiz(2));
    maxMat = ones(size(maxMat));
end

radMat = [radMat, radMat(:,1)]; % to close the circle
xx = zeros(matSiz(1), matSiz(2) +1);
yy = xx;
for ii=1:matSiz(1)
    [xx(ii, :), yy(ii, :)] = pol2cart(thetaVec, radMat(ii, :));
end

meanMat = (radMat(:,1:end-1) * exp(1i*thetaVec(1:end-1))')./sum(radMat(:, 1:end-1),2);
meanMat = conj(meanMat); % since the thetas are flipped in the plot

lineMSiz = 5;
meanMSiz = 3;


hold on 

plot(axh, 0, 0, 'o', 'markersize', meanMSiz, 'markerfacecolor', 'k', 'markeredgecolor', 'k') %just to make origin black

for ii=1:matSiz(1)
    switch plotType
        case 'line'
            plot(axh, xx(ii, :), yy(ii, :), '-o', 'markersize', lineMSiz, ...
                 'color', relCol(ii, :), 'linewidth', 2, ...
                 'markerfacecolor', relCol(ii, :), 'Tag', 'legendTag');
        case 'mean'
            plot(axh, [0, meanMat(ii)*maxMat(ii)], '-s', 'markersize', meanMSiz, ...
                 'color', relCol(ii, :), 'markerfacecolor', relCol(ii, :), 'linewidth', 4, 'Tag', 'legendTag');
        case 'both'
            plot(axh, xx(ii, :), yy(ii, :), '-o', 'markersize', lineMSiz, ...
                 'color', relCol(ii, :), 'markerfacecolor', relCol(ii, :), 'linewidth', 2); 
            plot(axh, [0, meanMat(ii)*maxMat(ii)], '-o', 'markersize', meanMSiz, ...
                'color', relCol(ii, :), 'markerfacecolor', relCol(ii, :), 'linewidth', 4, 'Tag', 'legendTag');
    end
end


if noMaxValFlag
    maxVal = max(abs([axh.XLim, axh.YLim]));
end

axis square
box off

axh.XLim = [-maxVal, maxVal];
axh.YLim = [-maxVal, maxVal];

axh.XAxisLocation = 'origin';
axh.YAxisLocation = 'origin';

xxTick = axh.XTick;
axh.YTick = xxTick;
axh.XTick = xxTick;
axh.XTickLabel = arrayfun(@num2str, abs(xxTick), 'uniformoutput', 0);
axh.YTickLabel = arrayfun(@num2str, abs(xxTick), 'uniformoutput', 0);
xxTick = xxTick(xxTick > 0);

hold on 
cirDat = linspace(0, 2*pi, 100);

cirCol = [1,1,1]* 0.85;

line([-xxTick(end), xxTick(end)] * 1/sqrt(2), [-xxTick(end), xxTick(end)] * 1/sqrt(2), 'color', cirCol, 'linewidth', 1);
line([xxTick(end), -xxTick(end)] * 1/sqrt(2), [-xxTick(end), xxTick(end)] * 1/sqrt(2), 'color', cirCol, 'linewidth', 1);

for ii=1:length(xxTick)
    line(sin(cirDat) * xxTick(ii), cos(cirDat) * xxTick(ii), 'color', cirCol)
end


hold off

axh.Children = flipud(axh.Children); % to put the data on top 

if legFlag
    legHand = findobj(axh, 'Tag', 'legendTag');
    legH = legend(legHand, 'location', 'southwestoutside');
    legH.Box = 'off';
    legH.FontSize = 12;
end


if nargout == 1 
    varargout{1} = axh;
elseif nargout == 2
    varargout{1} = axh;
    varargout{2} = legH;
end





end
