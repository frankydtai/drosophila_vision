function plotStepData(stepDat)

% This functions plots the the mean and each repeat of the current steps in
% step data. Its input is an NX2 matrix (N number of smaples of Current and
% Voltage channels). 


% getting steps out of the noisy voltage data
Vdata = stepDat(:,1);
 Cdata = stepDat(:,2)*10; % to convert to mV
[histV, histP] = hist(Vdata, 250);
indsCell = SplitVec(find(histV), 'consecutive');

volRange = zeros(2, length(indsCell));

for ii=1:length(indsCell)
    temp = histP(indsCell{ii});
    tmin = min(temp);
    tmax = max(temp);
    fudge = (tmax-tmin)/2;
    volRange(:,ii) = [tmin-fudge, tmax+fudge];
end
    
numStep = size(volRange,2);
axh = zeros(1, numStep);
maxLen = 0;
for ii=1:numStep % skips over zero current injection
    
    axh(ii) = subplot(1, numStep, ii);
    
    vBySamp = Vdata > volRange(1,ii) & Vdata < volRange(2,ii);
    [tempVal, tempBrack] = SplitVec(vBySamp, 'equal', 'firstval', 'bracket');
    relBrack = find(tempVal);
    
    if ii==1
        addSamp = 0;
    else
        addSamp = 750;
    end
    
    for jj=1:length(relBrack)
        rangeToPlot = (tempBrack(relBrack(jj), 1)-addSamp):(tempBrack(relBrack(jj), 2)+addSamp);
        if length(rangeToPlot) > maxLen
            maxLen = length(rangeToPlot);
        end
        hold on
        plot(Cdata(rangeToPlot))
    end
    hold off
    
end


yyall = get(axh(:), 'ylim');
totmin = min(cellfun(@min, yyall));
totmax = max(cellfun(@max, yyall));

set(axh(:), 'ylim', [totmin, totmax], 'xlim', [0, maxLen]);









end