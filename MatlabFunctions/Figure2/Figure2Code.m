

addpath ../supportingFunctions/

% this might take a while

load ./singleBarStT4.mat
load ./singleBarStT5.mat




%% organizing the T4 dataset in a cell array so that all positions are aligned

relLen = 45000; % measure manually (increased for T5 since ISI is bigger)
newZero = floor(relLen/2.5);

allDur = [0.02, 0.04, 0.08, 0.16, 0.32];
allWid = [1,2,4];
allVal = [0,1];

% 31 for between -15 and +15 (empty will be cleared)
% 5 since one cell has 320
% 3 widths (1,2,4) and 2 vals (0,1)
allSigDatCellT4 = cell(length(singleBarStT4), 31, length(allDur), length(allWid), length(allVal)); % 


for ii=1:length(singleBarStT4)
    
    if ii==5
        continue
    end
    
    relParams = singleBarStT4(ii).result(end,end,end,end);
    
    origPos = singleBarStT4(ii).positions;
    relMaxExt = relParams.maxExtPosVal;
    relPD = sign(relParams.minInhPosVal - relMaxExt);
    
    preRelPos = (origPos - relMaxExt); % * relPD +15; % +15 since allSigDatMat is 31 positions long              
        
    tempDur = singleBarStT4(ii).durations; 
    tempWid = singleBarStT4(ii).widths; 
    tempVal = singleBarStT4(ii).vals; 
    
    for pp=1:length(preRelPos)
    
        for dd=1:length(tempDur)

            durInd = find(allDur == tempDur(dd)); 

            for ww=1:length(tempWid)

                widInd = find(allWid == tempWid(ww));
                
                % have to take width into account when flipping PD (since
                % the leading edge which signifies position is flipped)
                if relPD == 1
                    relPos = preRelPos * relPD + 16;
                else
                    relPos = (preRelPos - tempWid(ww) + 1) * relPD + 16;  % +16 since allSigDatMat is 31 positions long  
                end

                for vv=1:length(tempVal)

                    valInd = find(allVal == tempVal(vv)); 

                    relDat = singleBarStT4(ii).result(pp,dd,ww,vv);
                    
                    if relDat.empty
                        continue
                    end

                    inputSt.zeroInd = relDat.subData.zeroInd;
                    inputSt.data = relDat.subData.baseSub(:,2);
                    shiftVec = padRespVecGen(inputSt, newZero, relLen);
                    allSigDatCellT4{ii,relPos(pp),durInd,widInd,valInd} = shiftVec;

                end

            end        

        end
        
    end
    
    
end

clear inputSt relPos dSiz ii pp dd ww vv shiftVec relPD orig* relMaxExt relParams relDat 
clear tempVal tempDur tempWid valInd durInd widInd 

% Taking the data for the 2 most used step durations

relSigDatCellT4 = allSigDatCellT4(:,:,[2,4], :,:);
relTimeVec = (((1:relLen) - newZero)/20)';


%% Same for T5


% 31 for between -15 and +15 (empty will be cleared)
% 5 since one cell has 320
% 3 widths (1,2,4) and 2 vals (0,1)
allSigDatCellT5 = cell(length(singleBarStT5), 31, 5, 3, 2); 


for cn=1:length(singleBarStT5)
    
    relParams = singleBarStT5(cn).result(end,end);
    
    origPos = singleBarStT5(cn).positions;
    relMaxExt = relParams.maxExtPosVal;
    relPD = sign(relParams.minInhPosVal - relMaxExt);
    
    preRelPos = (origPos - relMaxExt); % * relPD +16; % +15 since allSigDatMat is 31 positions long  -  Should be 16           
        
    tempDur = singleBarStT5(cn).durations; 
    tempWid = singleBarStT5(cn).widths; 
    tempVal = singleBarStT5(cn).vals; 
    
    for pp=1:length(preRelPos)
    
        for dd=1:length(tempDur)

            durInd = allDur == tempDur(dd); 

            for ww=1:length(tempWid)

                widInd = allWid == tempWid(ww);
                
                % have to take width into account when flipping PD (since
                % the leading edge which signifies position is flipped)
                if relPD == 1
                    relPos = preRelPos * relPD + 16;
                else
                    relPos = (preRelPos - tempWid(ww) + 1) * relPD + 16;  % +16 since allSigDatMat is 31 positions long  
                end

                for vv=1:length(tempVal)

                    valInd = allVal == tempVal(vv); 

                    relDat = singleBarStT5(cn).result(pp,dd,ww,vv);
                    preRelMax = singleBarStT5(cn).result(end,end,ww,vv).max; 
                    relMax = preRelMax(pp,dd); 
                    
                    preRelMin = singleBarStT5(cn).result(end,end,ww,vv).min; 
                    relMin = preRelMin(pp,dd); 
                    
                    if relDat.empty
                        continue
                    end

                    inputSt.zeroInd = relDat.subData.zeroInd;
                    inputSt.data = relDat.subData.baseSub(:,2);
                    shiftVec = padRespVecGen(inputSt, newZero, relLen);
                    allSigDatCellT5{cn,relPos(pp),durInd,widInd,valInd} = shiftVec;

                end

            end        

        end
        
    end
    
    
end

clear inputSt relPos dSiz ii pp dd ww vv shiftVec relPD orig* relMaxExt relParams relDat 
clear tempVal tempDur tempWid valInd durInd widInd preRelM* relMin relMax cn 

%
relSigDatCellT5 = allSigDatCellT5(:,:,[2,4], :,:);
relTimeVec = (((1:relLen) - newZero)/20)';


%% plotting the single bar responses by position

close all


w2hFac = 1.25;
figW = 18.3; %183mm 

fh = figure('units', 'centimeters', 'position', [5, 5, figW, figW*w2hFac], 'color', [1,1,1]);


stimDur = [20, 40, 80, 160, 320]; % times 20 to turn into samples
barWid = [1,2,4];
relStimDurInd = [2,4];
relStimDur = stimDur(relStimDurInd);
relVal = 1:2;

yyLim = [-7.5, 25];
yyLimSt = [-5, 5];
xxLim = [ -200, 750];

dSiz = size(allSigDatCellT4);
sdSmoothWin = 2000;
downSampFac = 100;

timeVec = ((1:downSampFac:relLen) - newZero)/20;

chopEdges = 9;
relPosInd = (1+chopEdges:dSiz(2)-chopEdges) + 1;

xSt1 = 0.05;
xEnd1 = 0.95;
ySt1 = 0.65;
yEnd1 = 0.95; 
xDif1 = 0.001; 
yDif1 = 0.02; 

% plotting only w1 40ms, and w4 40 and 160ms
plotCond = [3,2];
bothCells = {allSigDatCellT4, allSigDatCellT5};

posCell1 = generatePositionCell(xSt1, xEnd1, ySt1, yEnd1, xDif1, yDif1, [length(relPosInd), 2]); % for T4 and T5


pColB = [0.6784, 0.8667, 0.5569; 0.1922, 0.6392, 0.3294];
pColD = [0.5882, 0.5882, 0.5882; 0.1451, 0.1451, 0.1451];

axh = gobjects(size(posCell1));


for cl = 1:length(bothCells)
    
    wI = plotCond(1);
    durI = plotCond(2);
    
    relAllCell = bothCells{cl};

    for jj=1:length(relPosInd)

        pInd = relPosInd(jj);
        
        axh(jj,cl) = axes('position', posCell1{jj, cl});
        
        hold on 

        for vv=length(relVal):-1:1  % to plot the dark on top 

            if vv==1
                pCol = pColD; 
            else
                pCol = pColB; 
            end

            plotMean = mean([relAllCell{:, pInd, relStimDurInd(durI), wI, vv}], 2);
            plotMeanRed = plotMean(1:downSampFac:end);

            if isempty(plotMean) % extrame positions might be empty
                axh(jj, cl).YColor = 'none'; 
                axh(jj, cl).XColor = 'none'; 
                continue
            end

            %adding stim lines
            line([0, 0], yyLimSt, 'linewidth', 1, 'color', [1,1,1]*0.6)
            line([relStimDur(durI), relStimDur(durI)], yyLimSt, 'linewidth', 1, 'color', [1,1,1]*0.6)

            relDat = [relAllCell{:, pInd, relStimDurInd(durI), wI, vv}]; 
            
            relRedDat = relDat(1:downSampFac:end, :);
            plotSEM = smooth(std(relRedDat, 0, 2)/sqrt(size(relRedDat,2)), sdSmoothWin);

            patchXX = ([1:relLen, relLen:-1:1]' - newZero) / 20 ; % to match time
            patchYY = [plotMeanRed + plotSEM; flipud(plotMeanRed - plotSEM)];

            downSampPatchXX = patchXX(1:downSampFac:end); 
            downSampPatchYY = patchYY; % downsmapled before calculting SD

            plot(downSampPatchXX, downSampPatchYY, 'linewidth', 1, 'color', pCol(2, :))
            plot(timeVec, plotMeanRed, 'linewidth', 3, 'color', pCol(2, :)) % not downsampled - then have to change ranges for everything
            
            if cl ==1
                title({num2str(pInd-16)})
            end
            
            if jj==1
               text(xxLim(1) + 100, yyLim(2) - 10, ['Wid:', num2str(barWid(wI))])  
            end
            
        end

        hold off
        
    end
      
end


dSiz = size(axh); 

for ii=1:dSiz(1)
    for jj=1:dSiz(2)
        axh(ii,jj).YLim = yyLim;
        axh(ii,jj).XLim  = xxLim; 
        axh(ii,jj).YColor = 'none';
        axh(ii,jj).XColor = 'none';
    end
end


% plotting scalebars

vmLength = 10; %in mV
timeLength = 250; %in ms
stVal = 100; 
yStVal = 5; 


fh.CurrentAxes =  axh(1,1);
    
line([xxLim(1)+stVal, xxLim(1)+stVal + timeLength], [yStVal, yStVal], 'linewidth', 3, 'color', 'k')
text(xxLim(1)+stVal, yStVal - 2, [num2str(timeLength), 'ms'])
    
line([xxLim(1)+stVal, xxLim(1)+stVal], [yStVal, yStVal+vmLength], 'linewidth', 3, 'color', 'k')
text(xxLim(1)+stVal - 200, yStVal + 5, [num2str(vmLength), 'mV'])

clear downSampPatc? patchXX patchYY relDat relRedDat cl jj ii
clear plotSEM plotMean? relPosInd pInd 

%% organizing response by position table


bothCellSt = {singleBarStT4, singleBarStT5};
TType = [4,5];

relWid = [1,2,4]; 
relDur = 0.16;
relVal = [0,1];

respTab = table;


for cl = 1:length(bothCellSt)
    
    relSt = bothCellSt{cl};

    for cc=1:length(relSt)

        if isempty(relSt(cc).result) || length(relSt(cc).vals) == 1
            continue
        end

        tempStat = relSt(cc).result(end,end,end,end);
        
        for ww=1:length(relWid)

            dInd = relSt(cc).durations == relDur;
            wInd = relSt(cc).widths == relWid(ww); 
            if all(wInd==0)
                continue
            end
            tempPos = relSt(cc).positions;
            vInds = arrayfun(@(x) find(relSt(cc).vals == x),relVal);

            tempEPos = tempStat.maxExtPosVal;
            tempIPos = tempStat.minInhPosVal;
            normPos = nan(size(tempPos)); 

            relPD = sign(tempIPos - tempEPos);

            if relPD == 1
                normPos = tempPos - tempEPos;
            else
                normPos = (tempPos - tempEPos - relWid(ww) + 1) * relPD; 
            end 
        

            for ii=1:length(vInds)

                tempDat = relSt(cc).result(:, dInd, wInd, vInds(ii)); 

                for jj=1:length(tempDat)

                    if isempty(tempDat(jj).resp)
                        continue
                    end

                    respMax = tempDat(jj).resp.maxVal;
                    respMin = tempDat(jj).resp.minVal;

                    if respMax > 0 || respMin < 0
                        cellType = TType(cl); 
                        cellNum = cc;
                        relPos = normPos(jj);
                        barV = relVal(ii);
                        barW = relWid(ww);

                        respTab = [respTab; table(cellType, cellNum, relPos, barW, barV, respMax, respMin)];

                    end

                end

            end
            
        end

    end
    
end

clear relSt respM* cellNum temp* cc jj *Ind normPos cl cellType TType vInds ii barV

%% checking width effect on NC response magnitude by cell type 

relWidComp = [1,4; 2,4];
maxTab = table;
% minTab = table;

for cc=1:2
    
    relV = cc-1;
    relCT = cc+3;
    
    relTab = respTab(respTab.cellType == relCT & respTab.barV == relV, :);
    
    relCells = unique(relTab.cellNum);
    
    for cn=1:length(relCells)
        
        cNum = relCells(cn);
        relP = unique(relTab.relPos(relTab.cellNum == cNum));
        
        for pp=1:length(relP)
            
            rPos = relP(pp);
            tempTab = relTab(relTab.cellNum == cNum & relTab.relPos == rPos, :);
            temp2Tab = tempTab(tempTab.respMax ~= 0, :);
            
            if all(ismember(relWidComp(1,:), temp2Tab.barW))
                max4 = temp2Tab.respMax(temp2Tab.barW == relWidComp(1,2));
                max1 = temp2Tab.respMax(temp2Tab.barW == relWidComp(1,1));
                maxDiff = max4 - max1;
                widCI = 1;
                maxTab = [maxTab; table(relCT, relV, cNum, rPos, widCI, maxDiff)];
            elseif all(ismember(relWidComp(2,:), temp2Tab.barW))
                max4 = temp2Tab.respMax(temp2Tab.barW == relWidComp(2,2));
                max2 = temp2Tab.respMax(temp2Tab.barW == relWidComp(2,1));
                maxDiff = max4 - max2;
                widCI = 2;
                maxTab = [maxTab; table(relCT, relV, cNum, rPos, widCI, maxDiff)];

            end
            
        end
        
    end
    
end
        

[h,p] = ttest(maxTab.maxDiff(maxTab.widCI == 1 & maxTab.relCT ==4));

% relCT=4 widCI=1 p =4.1053e-08
% relCT=4 widCI=2 p =1.1745e-13
% relCT=5 widCI=1 p =1.9365e-05
% relCT=5 widCI=2 p =2.7596e-08

%% plotting response by position

relWid = [1,2,4]; 
relVal = [0,1];
numCutOff = 3; 

xSt2 = xSt1; 
xEnd2 = xEnd1; 
ySt2 = 0.35; 
yEnd2 = ySt1 - 0.025; 
yDif2 = 0.02;
xDif2 = 0.05;

posCell2 = generatePositionCell(xSt2, xEnd2, ySt2, yEnd2, xDif2, yDif2, [4, length(bothCellSt)]);
axh2 = gobjects(size(posCell2)); 

xxLim = [-5,10];
yyLim = [-15, 30];

relCol = [0, 0, 0; 0.1922, 0.6392, 0.3294]; 
jit = 0.1; 


for cl=1:length(bothCellSt)
    
    ww=3; % plotting only w4 at first

    for vv=1:length(relVal)
        
        axh2(vv,cl) = axes('position', posCell2{vv,cl}); 
        hold on 
        
        relTTab = respTab(respTab.cellType == cl+3 & respTab.barV == relVal(vv) & respTab.barW == relWid(ww), :);

        nonZMaxI = relTTab.respMax > 0;
        nonZMinI = relTTab.respMin < 0;

        [maxGroups, maxGID] = findgroups(relTTab.relPos(nonZMaxI));
        meanMaxByPos = splitapply(@mean, relTTab.respMax(nonZMaxI), maxGroups);
        lengthMaxByPos = splitapply(@length, relTTab.respMax(nonZMaxI), maxGroups);
        meanMaxInd = lengthMaxByPos > numCutOff; 

        [minGroups, minGID] = findgroups(relTTab.relPos(nonZMinI));
        meanMinByPos = splitapply(@mean, relTTab.respMin(nonZMinI), minGroups);
        lengthMinByPos = splitapply(@length, relTTab.respMin(nonZMinI), minGroups);
        meanMinInd = lengthMinByPos > numCutOff; 

        line(xxLim, [0,0], 'linewidth', 1, 'color', 'k')
        plot(relTTab.relPos(nonZMaxI) + rand(sum(nonZMaxI), 1) * jit, relTTab.respMax(nonZMaxI), '.', ...
             'markerfacecolor', 'none', 'markeredgeColor', relCol(vv,:) + [1,1,1]*0.05, 'markersize', 5);
        plot(relTTab.relPos(nonZMinI) + rand(sum(nonZMinI), 1) * jit, relTTab.respMin(nonZMinI), '.', ...
             'markerfacecolor', 'none', 'markeredgeColor', relCol(vv,:) + [1,1,1]*0.05, 'markersize', 5);

        plot(maxGID(meanMaxInd), meanMaxByPos(meanMaxInd), 'linewidth', 2, ...
             'color', relCol(vv,:), 'marker', 'o', 'markerfacecolor', relCol(vv,:), 'markeredgecolor', 'none', 'markersize', 7);
        plot(minGID(meanMinInd), meanMinByPos(meanMinInd), 'linewidth', 2, ...
              'color', relCol(vv,:), 'marker', 'o', 'markerfacecolor', relCol(vv,:), 'markeredgecolor', 'none', 'markersize', 7);
        
        axh2(vv,cl).XTick = (xxLim(1):5:xxLim(2)) + floor(ww/1.5); % +2 since it is W4
        axh2(vv,cl).XTickLabel = arrayfun(@num2str, xxLim(1):5:xxLim(2), 'uniformoutput', 0);

    end
    
    hold off
    
    
    
end




for cl=1:length(bothCellSt)
    
    for ww=1:length(relWid)-1 % plotted above
    
        axh2(2+ww,cl) = axes('position', posCell2{2+ww,cl}); 
        hold on 

        for vv=1:length(relVal)
            
            relTTab = respTab(respTab.cellType == cl+3 & respTab.barV == relVal(vv) & respTab.barW == relWid(ww), :);
            
            nonZMaxI = relTTab.respMax > 0;
            nonZMinI = relTTab.respMin < 0;
            
            [maxGroups, maxGID] = findgroups(relTTab.relPos(nonZMaxI));
            meanMaxByPos = splitapply(@mean, relTTab.respMax(nonZMaxI), maxGroups);
            lengthMaxByPos = splitapply(@length, relTTab.respMax(nonZMaxI), maxGroups);
            meanMaxInd = lengthMaxByPos > numCutOff; 

            [minGroups, minGID] = findgroups(relTTab.relPos(nonZMinI));
            meanMinByPos = splitapply(@mean, relTTab.respMin(nonZMinI), minGroups);
            lengthMinByPos = splitapply(@length, relTTab.respMin(nonZMinI), minGroups);
            meanMinInd = lengthMinByPos > numCutOff; 

            line(xxLim, [0,0], 'linewidth', 1, 'color', 'k')

            plot(maxGID(meanMaxInd), meanMaxByPos(meanMaxInd), 'linewidth', 2, ...
                 'color', relCol(vv,:), 'marker', 'o', 'markerfacecolor', relCol(vv,:), 'markeredgecolor', 'none', 'markersize', 7);
            plot(minGID(meanMinInd), meanMinByPos(meanMinInd), 'linewidth', 2, ...
                  'color', relCol(vv,:), 'marker', 'o', 'markerfacecolor', relCol(vv,:), 'markeredgecolor', 'none', 'markersize', 7);
              
            axh2(2+ww,cl).XTick = (xxLim(1):5:xxLim(2)) + floor(ww/1.5); 
            axh2(2+ww,cl).XTickLabel = arrayfun(@num2str, xxLim(1):5:xxLim(2), 'uniformoutput', 0);

        end

        hold off
        
        
    
    end
    
end

axS = size(axh2);

for ii=1:axS(1)
    for jj=1:axS(2)
        
        if jj==1
            % to put dark resp on top
            axh2(ii,jj).Children = flipud(axh2(ii,jj).Children);
            axh2(ii,jj).XColor= 'none';
        end
        axh2(ii,jj).XLim = xxLim; 
        axh2(ii,jj).YLim = yyLim; 
        axh2(ii,jj).YTick = -10:10:yyLim(2); 
        
        if ii > 1
            axh2(ii,jj).YColor= 'none';
        end
        
    end
end

%% finding start time differences 


relVal = [0,1];
relWid = 4;
relDur = 0.160;

maxCutOff = 2; % responses smaller than this are not included
qRange = [0.1, 0.9]; % in precentage from max 
qRangeD = [0.2, 0.8]; % for decay 
arenaDelay = 7; % measured in arenaDelayScript (for T4 paper)

diffTab = table;
cellType = [4,5];



for cl = 1:length(bothCellSt)
    
    relCellSt = bothCellSt{cl};

    for ii=1:length(relCellSt)

        if isempty(relCellSt(ii).result) % cell 5
            continue
        end

        tempEPos = relCellSt(ii).result(end,end,end,end).maxExtPosVal;
        tempIPos = relCellSt(ii).result(end,end,end,end).minInhPosVal;
        relPos = relCellSt(ii).positions;
        normPos = nan(size(relPos)); 

        relPD = sign(tempIPos - tempEPos);

        if relPD == 1
            normPos = relPos - tempEPos;
        else
            normPos = (relPos - tempEPos - relWid + 1) * relPD; 
        end 

        widI = find(relCellSt(ii).widths == relWid);
        durI = find(relCellSt(ii).durations == relDur);

        for vv=1:length(relVal)

            for pp=1:length(relPos)
                
                % if these are no 2 vals for the cell or the particular
                % stimulus response is empty
                if size(relCellSt(ii).result,4) < 2 || relCellSt(ii).result(pp, durI, widI, vv).empty 
                    continue
                end

                % For opposite polarity takes the hypP into account when
                % comparing to maxCutoff
                if vv==cl && ( relCellSt(ii).result(pp, durI, widI, vv).resp.maxVal == 0 || ...
                       relCellSt(ii).result(pp, durI, widI, vv).resp.maxVal - relCellSt(ii).result(pp, durI, widI, vv).resp.minVal < maxCutOff)
                    continue
                end
                % for same polarity - just the max
                if vv~=cl && relCellSt(ii).result(pp, durI, widI, vv).resp.maxVal < maxCutOff
                    continue
                end


                tempZI = relCellSt(ii).result(pp, durI, widI, vv).subData.zeroInd;
                tempTime = relCellSt(ii).result(pp, durI, widI, vv).subData.baseSub(:, 1);
                tempDat = relCellSt(ii).result(pp, durI, widI, vv).subData.baseSub(:, 2);

                maxInd = relCellSt(ii).result(pp, durI, widI, vv).resp.maxInd;
                minInd = relCellSt(ii).result(pp, durI, widI, vv).resp.minInd;
                minVal = relCellSt(ii).result(pp, durI, widI, vv).resp.minVal;
                maxVal = relCellSt(ii).result(pp, durI, widI, vv).resp.maxVal;


                % for opposite bars if there is a min before the peak 
                if vv==cl && minVal < 0 && minInd < maxInd 
                    totValR = maxVal - minVal;
                    qRangeVR = totValR * qRange + minVal;
                    riseStInd = minInd; 
                else
                    totValR = maxVal;
                    qRangeVR = totValR * qRange;
                    riseStInd = tempZI; 
                end

                % for same polarity bars if there is a min after the peak 
                if vv~=cl && minVal < 0 && maxInd < minInd 
                    totValD = maxVal - minVal;
                    qRangeVD = totValD * fliplr(qRangeD) + minVal;
                    decayEndInd = minInd; 
                else
                    totValD = maxVal;
                    qRangeVD = totValD * fliplr(qRangeD);
                    decayEndInd = length(tempDat); 
                end

                % in case part of the rise is out of range
                preRiseInds = arrayfun(@(x) find(tempDat(riseStInd:maxInd) < x, 1, 'last'), qRangeVR, 'uniformoutput', 0);
                if any(cellfun(@isempty, preRiseInds))
                    continue
                else
                    preRiseInds = [preRiseInds{:}];
                end
                riseInds = preRiseInds+riseStInd-1; % bringing it back to global ind ref
                
                % time difference 
                timeDiffR = tempTime(riseInds(2)) - tempTime(riseInds(1)); 
                riseStTime = tempTime(riseInds(1)) - arenaDelay; 
                riseStInd = riseInds(1);
                riseEndInd = riseInds(2);
                
                preDecayInds = arrayfun(@(x) find(tempDat(maxInd:decayEndInd) < x, 1, 'first'), qRangeVD, 'uniformoutput', 0);
                if any(cellfun(@isempty, preDecayInds))
                    continue
                else
                    preDecayInds = [preDecayInds{:}];
                end
                decayInds = preDecayInds+maxInd-1; % bringing it back to global ind ref
                % time difference 
                timeDiffD = tempTime(decayInds(2)) - tempTime(decayInds(1)); 
                
                decayStInd = decayInds(1);
                decayEndInd = decayInds(2);
                
                nPos = normPos(pp);
                val = relVal(vv);
                cellNum = ii;
                Ttype = cellType(cl);
                
                diffTab = [diffTab; table(Ttype, cellNum, val, nPos, riseStInd, riseEndInd, riseStTime, timeDiffR, ...
                           decayStInd, decayEndInd, timeDiffD)];

            end

            fprintf('Finished cell %d value %d\n', ii, vv)

        end

    end
    
end

clear decay* rise* iter* ii pp vv nPos val cellNum startV targetV cl
clear temp* tot* lin* max* min* normPos widI durI qRange* timeDiff* Ttype


%%

relV = [0,1];
relSP = -1:5;

riseStTCell = cell(2, length(relV)*length(relSP));

for cl = 1:length(bothCellSt)
    for vv=1:length(relV)
        for pp=1:length(relSP)
            riseStTCell{cl, vv, pp} = diffTab(diffTab.Ttype == cellType(cl) & diffTab.val == relV(vv) & diffTab.nPos == relSP(pp), :).riseStTime;
        end 
    end
end


%% plotting results

boxW = 2;

xSt3 = xSt1;
xEnd3 = 0.4;
ySt3 = 0.05; 
yEnd3 = ySt2 - 0.05; 
yDif3 = 2*yDif2;
yyLim3 = [0,300];

posCell3 = generatePositionCell(xSt3, xEnd3, ySt3, yEnd3, -1, yDif3, 2);
axh3 = gobjects(size(posCell3)); 


pCol = [ 0, 0, 0; 0.1922, 0.6392, 0.3294];

for cl=1:length(bothCellSt)

    axh3(cl) = axes('position', posCell3{cl});
    
    for vv=1:length(allVal)
        
        relCell = squeeze(riseStTCell(cl,vv,:));
        
        relCol = pCol(vv,:);
        
        allPos = 6*(1:length(relSP)) + 2*(vv-1);
        
        plotSt.jitRange = boxW / 2; 
        plotSt.boxWid = boxW;
        plotSt.boxEdgeCol = relCol;
        plotSt.boxFillCol = 'w';
        plotSt.dotEdgeCol = 'none';
        plotSt.dotFillCol = relCol + [1, 1, 1]*0.05; % to be able to select 
        plotSt.medLineCol = 'k'; 
        plotSt.dotSize = 3;
        plotSt.boxLineWid = 2; 
        plotSt.positions = allPos; 
        
        plotBoxandJitterDots(axh3(cl), relCell, plotSt)
        
        if vv==1
            line([allPos(1), allPos(end)], [160,160], 'linewidth', 1, 'color', 'k')
            axh3(cl).XTick =6*(1:length(relSP)) + 1;
            axh3(cl).XTickLabel = arrayfun(@num2str, relSP - 2, 'uniformoutput', 0); % -2 since it is w4
        end
        
    end
    
    axh3(cl).YLim = yyLim3;
    
end

axh3(1).XTickLabel = {};





%%

% copied from axh1


allPosSiz = size(allSigDatCellT5,2);
relPosInd = (1+chopEdges:allPosSiz-chopEdges) + 1;

xSt4 = xEnd3 + 0.025;
xEnd4 = xEnd1;
ySt4 = ySt3;
yEnd4 = yEnd3; 
xyDif4 = 0.02; 

posCell4 = generatePositionCell(xSt4, xEnd4, ySt4, yEnd4, xyDif4, xyDif4, [3, 2]); 
axh4 = gobjects(size(posCell4));

relStimDurInd = [2,4];
relStimDur = stimDur(relStimDurInd);

yyLim = [-5, 12.5];
yyLimSt2 = [-5,5];
xxLim = [ -200, 750];

sdSmoothWin = 2000;
downSampFac = 100;

timeVec = ((1:downSampFac:relLen) - newZero)/20;

% plotting only w4 40 and 160ms in pos 1 
expWI = 3;
expPosI = relPosInd(7); % position 1; (for width 4 position -1)

for cl = 1:length(bothCells)
    
    relAllCell = bothCells{cl};
    axh4(2, cl) = axes('position', posCell4{2, cl});
    
    hold on 
    
    for dd = 1:length(relStimDurInd)
    
        durI = relStimDurInd(dd);

        % plot dark for T4 and bright for T5
        if cl==1
            pCol = pColD;
            vv=1;
        else
            pCol = pColB; 
            vv=2;
        end

        plotMean = mean([relAllCell{:, expPosI, durI, expWI, vv}], 2);
%         size([relAllCell{:, expPosI, durI, expWI, vv}], 2)
%         T4 n=15; T5 n=12
        plotMeanRed = plotMean(1:downSampFac:end);

        %adding stim lines
        if dd==1
            line([0, 0], yyLimSt2, 'linewidth', 1, 'color', [1,1,1]*0.6)
        end
        line([relStimDur(dd), relStimDur(dd)], yyLimSt2, 'linewidth', 1, 'color', [1,1,1]*0.6)

        relDat = [relAllCell{:, expPosI, durI, expWI, vv}]; 
        relRedDat = relDat(1:downSampFac:end, :);
        plotSEM = smooth(std(relRedDat, 0, 2)/sqrt(size(relRedDat,2)), sdSmoothWin);

        patchXX = ([1:relLen, relLen:-1:1]' - newZero) / 20 ; % to match time
        patchYY = [plotMeanRed + plotSEM; flipud(plotMeanRed - plotSEM)];

        downSampPatchXX = patchXX(1:downSampFac:end); 
        downSampPatchYY = patchYY; % downsmapled before calculting SD

        patch(downSampPatchXX, downSampPatchYY, pCol(dd, :), ...
             'edgecolor', pCol(dd, :), 'facealpha', 0.4)

        plot(timeVec, plotMeanRed, 'linewidth', 3, 'color', pCol(dd, :)) % not downsampled - then have to change ranges for everything
        
    end
    
    hold off
      
end

dSiz = size(axh4); 

for ii=1:dSiz(2)

    axh4(2,ii).YLim = yyLim;
    axh4(2,ii).XLim  = xxLim; 
    axh4(2,ii).YColor = 'none';
    axh4(2,ii).XColor = 'none';

end


% plotting scalebars

vmLength = 10; %in mV
timeLength = 250; %in ms
stVal = 100; 
yStVal = 5; 

fh.CurrentAxes =  axh4(2,1);
    
line([xxLim(1)+stVal, xxLim(1)+stVal + timeLength], [yStVal, yStVal], 'linewidth', 3, 'color', 'k')
text(xxLim(1)+stVal, yStVal - 2, [num2str(timeLength), 'ms'])
    
line([xxLim(1)+stVal, xxLim(1)+stVal], [yStVal, yStVal+vmLength], 'linewidth', 3, 'color', 'k')
text(xxLim(1)+stVal - 200, yStVal + 5, [num2str(vmLength), 'mV'])
    

%% comparing 40 and 160ms max responses to w4 dark bars
% comapring responses from normPos 0:2 (not corrected for width)
% uses to window zeroInd : respTime to calculate max

relVal = [0,1];
relWid = 4;
relDur = [0.04,0.160];
stTime = 50; % ~40ms for resp and 7ms arena delay
respTime = 200; % in ms (after stim presentation)

singleW4BarTabT4 = table;
singleW4BarTabT5 = table;

for ii=1:length(singleBarStT4)
    if isempty(singleBarStT4(ii).result) % cell 5
        continue
    end
    tempEPosI = singleBarStT4(ii).result(end,end,end,end).maxExtPosInd;
    tempIPosI = singleBarStT4(ii).result(end,end,end,end).minInhPosInd;
    
    valI = find(singleBarStT4(ii).vals == relVal(1));
    widI = find(singleBarStT4(ii).widths == relWid);
    durI = arrayfun(@(x) find(singleBarStT4(ii).durations == x), relDur);
    
    % considering only position [0:2] (or for when refering to width 4: [-2:0])
    if tempEPosI < tempIPosI
        relPos = tempEPosI:tempEPosI+2;
    else
        relPos = tempEPosI-2:tempEPosI;
    end
    
    for dd=1:length(durI)
        
        respInterval = [stTime, stTime+respTime] + relDur(dd) * 1000;
        
        for pp=1:length(relPos)
            
            tempTime = singleBarStT4(ii).result(relPos(pp), durI(dd), widI, valI).subData.baseSub(:, 1);
            tempRespWin = arrayfun(@(x) find(tempTime > x, 1, 'first'), respInterval);
            tempDat = singleBarStT4(ii).result(relPos(pp), durI(dd), widI, valI).subData.baseSub(:, 2);
            
            cellNum = ii;
            dur = singleBarStT4(ii).durations(durI(dd));
            wid = singleBarStT4(ii).widths(widI);
            val = singleBarStT4(ii).vals(valI);
            pos = singleBarStT4(ii).positions(relPos(pp));
            
            respIntSt = tempRespWin(1);
            respIntEnd = tempRespWin(2);
            
            maxVal = max(tempDat(respIntSt:respIntEnd));
            meanVal = mean(tempDat(respIntSt:respIntEnd));
            upQVal = quantile(tempDat(respIntSt:respIntEnd), 0.75);
            
            singleW4BarTabT4 = [singleW4BarTabT4; table(cellNum, dur, wid, val, pos, respIntSt, respIntEnd, maxVal, meanVal, upQVal)];
            
        end
        
    end
    
end

for ii=1:length(singleBarStT5)
    if isempty(singleBarStT5(ii).result) % cell 5
        continue
    end
    tempEPosI = singleBarStT5(ii).result(end,end,end,end).maxExtPosInd;
    tempIPosI = singleBarStT5(ii).result(end,end,end,end).minInhPosInd;
    
    valI = find(singleBarStT5(ii).vals == relVal(2));
    widI = find(singleBarStT5(ii).widths == relWid);
    durI = arrayfun(@(x) find(singleBarStT5(ii).durations == x), relDur);
    
    if tempEPosI < tempIPosI
        relPos = tempEPosI:tempEPosI+2; % actually positions [-2:0] for width 4
    else
        relPos = tempEPosI-2:tempEPosI;
    end
    
    for dd=1:length(durI)
        
        respInterval = [stTime, stTime+respTime] + relDur(dd) * 1000;
        
        for pp=1:length(relPos)
            
            if size(singleBarStT5(ii).result(relPos(pp), durI(dd), widI, valI),4) == 0
                continue
            end
            
            if singleBarStT5(ii).result(relPos(pp), durI(dd), widI, valI).empty
                continue
            end
            
            tempTime = singleBarStT5(ii).result(relPos(pp), durI(dd), widI, valI).subData.baseSub(:, 1);
            tempRespWin = arrayfun(@(x) find(tempTime > x, 1, 'first'), respInterval);
            tempDat = singleBarStT5(ii).result(relPos(pp), durI(dd), widI, valI).subData.baseSub(:, 2);
            
            cellNum = ii;
            dur = singleBarStT5(ii).durations(durI(dd));
            wid = singleBarStT5(ii).widths(widI);
            val = singleBarStT5(ii).vals(valI);
            pos = singleBarStT5(ii).positions(relPos(pp));
            
            respIntSt = tempRespWin(1);
            respIntEnd = tempRespWin(2);
            
            maxVal = max(tempDat(respIntSt:respIntEnd));
            meanVal = mean(tempDat(respIntSt:respIntEnd));
            upQVal = quantile(tempDat(respIntSt:respIntEnd), 0.75);
            
            singleW4BarTabT5 = [singleW4BarTabT5; table(cellNum, dur, wid, val, pos, respIntSt, respIntEnd, maxVal, meanVal, upQVal)];
            
        end
        
    end
    
end


clear dur maxVal val pos wid cellNum tempZI tempDat relPos tempEPosI tempIPosI
clear valI durI widI relWid relVal ii pp dd tempRespEnd tempTime meanVal upQVal



%% plotting upper Q per cell 40 and 160
% 

[relGroupsT4, groupIDT4] = findgroups(singleW4BarTabT4(:, {'cellNum', 'dur'}));
meanSingleW4T4 = splitapply(@mean, singleW4BarTabT4.meanVal, relGroupsT4);

[relGroupsT5, groupIDT5] = findgroups(singleW4BarTabT5(:, {'cellNum', 'dur'}));
meanSingleW4T5 = splitapply(@mean, singleW4BarTabT5.meanVal, relGroupsT5);

for ii=1:length(relDur)
    meanSW4CellT4{ii} = meanSingleW4T4(groupIDT4.dur == relDur(ii));
    meanSW4CellT5{ii} = meanSingleW4T5(groupIDT5.dur == relDur(ii));
end

plotMSW4CellT4 = meanSW4CellT4;
plotMSW4CellT4{3} = plotMSW4CellT4{2};
plotMSW4CellT4{2} = plotMSW4CellT4{1} * 4; 

plotMSW4CellT5 = meanSW4CellT5;
plotMSW4CellT5{3} = plotMSW4CellT5{2};
plotMSW4CellT5{2} = plotMSW4CellT5{1} * 4;




boxW = 1.5;
xSq = 0.0275;

axh4(3,1) = axes('position', posCell4{3,1} + [xSq, 0, -xSq, 0]); 

plotSt.jitRange = boxW / 2; 
plotSt.boxWid = boxW;
plotSt.boxEdgeCol = pColD([1,1,2],:);
plotSt.boxFillCol = 'w'; 
plotSt.dotEdgeCol = 'none';
plotSt.dotFillCol = pColD([1,1,2],:) +[1,1,1]*0.05; 
plotSt.medLineCol = 'k'; 
plotSt.dotSize = [3, 0.1, 3];
plotSt.boxLineWid = 2; 
plotSt.positions = [3, 6, 9]; 

plotBoxandJitterDots(axh4(3,1), plotMSW4CellT4, plotSt)

axh4(3,2) = axes('position', posCell4{3,2} + [xSq, 0, -xSq, 0]); 
plotSt.boxEdgeCol = pColB([1,1,2],:); 
plotSt.dotEdgeCol = pColB([1,1,2],:) + [1,1,1]*0.01;
plotSt.dotFillCol = pColB([1,1,2],:) +[1,1,1]*0.01; 
plotBoxandJitterDots(axh4(3,2), plotMSW4CellT5, plotSt)


for ii=1:dSiz(2)
    axh4(3,ii).XTick = plotSt.positions; 
    axh4(3,ii).XTickLabel = arrayfun(@num2str, relDur([1,1,2]) * 1000, 'uniformoutput', 0); 
    axh4(3,ii).YLim = [-3, 17.5];
    axh4(3,ii).YTick = 0:5:15;
end

%% t test

[h4,p4,ci4,stats4] = ttest(plotMSW4CellT4{3}, plotMSW4CellT4{2});

[h5,p5,ci5,stats5] = ttest(plotMSW4CellT5{3}, plotMSW4CellT5{2});

%     h4        p4        h5       p5    
%     __    __________    __    _________
% 
%     1     1.8934e-05    1     0.0067024
% 

%% cell for decay time

relV = [0,1];
posCutOff = 2; % looking only at trailing side
absCO = 1; % lookin at center

decayTimeCell = cell(2, length(relV));
decayTimeCellID = decayTimeCell;

for cl = 1:length(bothCellSt)
    for vv=1:length(relV)
%         decayTimeCell{cl, vv} = diffTab(diffTab.Ttype == cellType(cl) & diffTab.val == relV(vv) & abs(diffTab.nPos) <= absCO, :).timeDiffD;
        decayTimeCell{cl, vv} = diffTab(diffTab.Ttype == cellType(cl) & diffTab.val == relV(vv) & ismember(diffTab.nPos, 2:4), :).timeDiffD;
        decayTimeCellID{cl, vv} = diffTab(diffTab.Ttype == cellType(cl) & diffTab.val == relV(vv) & ismember(diffTab.nPos, 2:4), :).cellNum;
    end
end

%% plotting decay time

pCol =  [0, 0, 0; 0.1922, 0.6392, 0.3294];

yyLim4a = [0,600];

for cl=1:length(bothCellSt)

    axh4(1,cl) = axes('position', posCell4{1,cl});
    hold on 
    
    for vv=1:length(allVal)
        
        relCell = squeeze(decayTimeCell(cl,vv));
        
        relCol = pCol(vv,:);
        
        allPos = 3 + vv-1;
        
        plotSt.jitRange = boxW / 4; 
        plotSt.boxWid = boxW /2;
        plotSt.boxEdgeCol = relCol;
        plotSt.boxFillCol = 'w';
        plotSt.dotEdgeCol = 'none';
        plotSt.dotFillCol = relCol +[1,1,1]*0.05;
        plotSt.medLineCol = 'k'; 
        plotSt.dotSize = 3;
        plotSt.boxLineWid = 2; 
        plotSt.positions = allPos; 
        
        plotBoxandJitterDots(axh4(1,cl), relCell, plotSt)
        
        if vv==1
%             line([allPos(1), allPos(end)], [160,160], 'linewidth', 1, 'color', 'k')
            axh4(1,cl).XTick =6*(1:length(relSP)) + 1;
            axh4(1,cl).XTickLabel = arrayfun(@num2str, relSP - 2, 'uniformoutput', 0); % -2 since it is w4
        end
        
    end
    
    hold off
    
    axh4(1,cl).YLim = yyLim4a;
    axh4(1,cl).XLim = [2.5, 4.5];
end

axh4(1,1).XColor = 'none';

axh4(1,2).XTick = [3,4];
axh4(1,2).XTickLabel = {'OFF', 'ON'};


