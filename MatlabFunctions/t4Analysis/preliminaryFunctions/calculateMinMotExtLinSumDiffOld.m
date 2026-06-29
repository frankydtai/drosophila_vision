function minMotLinSt = calculateMinMotExtLinSumDiffOld(pStruct)

% function calculateMinMotExtLinSumDiff(pStruct)
%
% This function is designed to use the output of calcMinMotExtLinComp and
% calculate the difference between the original data traces and the linear
% sum traces, and divide them by the bars timing. 
%
% INPUT
%
% pStruct -         protocolStruct from a minimal motion protocol where
%                   gratingTable already has the relevant fields (f/sAppear  and
%                   f/sDisappear). These can be added using the tableGeneratingScript
%
% OUTPUT
% TBD


fudgeTime = 35; %in ms , used as time between appearance of bar and actual resp
sampFac = 20; % 20kHz 
fudgeSamp = fudgeTime * sampFac;
preStimBaseWin = [-250, -50]; % ms before stim to use for baseline calculation
backToBaseFac = 2.5; % factor to multifly 

minMotLinSt = calcMinMotExtLinComp(pStruct);

datSiz = size(minMotLinSt);

%% calculating duration for diff calculation 

allBase = cell(datSiz(1), datSiz(3));

% calculating baseline
for ii=1:datSiz(1)
    for kk=1:datSiz(3)
        relDat = minMotLinSt(ii,ii,kk).subData.baseSub;
        baseInds = arrayfun(@(x) find(relDat(:,1) > x, 1, 'first'), preStimBaseWin); 
        allBase{ii,kk} = relDat(baseInds(1):baseInds(2), 2); 
    end 
end
        
baseSTD = std(vertcat(allBase{:}));

% finding the mean point when the response goes back to 1SD from baseline
stimEndWin = zeros(datSiz(1), datSiz(3));

for ii=1:datSiz(1)
    for kk=1:datSiz(3)
        
        relDat = minMotLinSt(ii,ii,kk).subData.baseSub;
        sbFrames = minMotLinSt(ii,ii,kk).data.table{1,{'sAppear'; 'sDisappear'}};
        sbTimeInd = ismember(minMotLinSt(ii,ii,kk).data.align.meanPos(:,2), sbFrames);
        sbSamp = minMotLinSt(ii,ii,kk).data.align.meanPos(sbTimeInd,1);
        
        backToBaseInd = find(relDat(sbSamp(2)+fudgeSamp:end,2) - mean(relDat(:,2)) < 0.5*baseSTD, 1, 'first');  % subtract mean to avoid effects of minor flactuations
        if isempty(backToBaseInd) || backToBaseInd < backToBaseFac*(sbSamp(2)-sbSamp(1));
            backToBaseInd = backToBaseFac*(sbSamp(2)-sbSamp(1)); % if can't find use length of stim dur
        end
        
        stimEndWin(ii,kk) = backToBaseInd; % since sb will be added later
%         stimEndWin(ii,kk) = relDat(backToBaseInd+sbSamp(2)+fudgeSamp-1, 1); 
        
%         plot(relDat(:,1), relDat(:,2))
%         hold on
%         plot(stimEndWin(ii,kk), 0, 'o', 'markerfacecolor', 'k', 'markersize', 8)
%         hold off 
%         pause
    end 
end


medEndWinSamp = round(median(stimEndWin));

%% calculating difference

for ii=1:datSiz(1)
    
    for jj=1:datSiz(2)
        
        if ii~=jj
        
            for kk=1:datSiz(3)
                
                if ~isempty(minMotLinSt(ii,jj,kk).data) %  condition is for upper side of matrix in timeDiff==0
                    
                    
                    % relTime = minMotLinSt(ii,jj,kk).subData.baseSub(:,1);
                    relDat = minMotLinSt(ii,jj,kk).subData.baseSub(:,2);
                    relSum = minMotLinSt(ii,jj,kk).linSum(:,2);
                    relDiff = relDat - relSum;
                
                    % getting bar appearance indices to calculate difference
                    % just there
                
%                   fbFrames = minMotLinSt(ii,jj,kk).data.table{1,{'fAppear'; 'fDisappear'}};
%                   fbTimeInd = ismember(minMotLinSt(ii,jj,kk).data.align.meanPos(:,2), fbFrames);
%                   fbSamp = minMotLinSt(ii,jj,kk).data.align.meanPos(fbTimeInd,1);
%                   fbInds = fbSamp + fudgeSamp;
                    sbFrames = minMotLinSt(ii,jj,kk).data.table{1,'sAppear'};
                    sbTimeInd = ismember(minMotLinSt(ii,jj,kk).data.align.meanPos(:,2), sbFrames);
                    sbSamp = minMotLinSt(ii,jj,kk).data.align.meanPos(sbTimeInd,1);
                
                
%                   fbInds = [minMotLinSt(ii,jj,kk).subData.zeroInd, sbSamp + fudgeSamp]; %between zero and second bar !! but then length is different
                    fbInds = [sbSamp + fudgeSamp - medEndWinSamp(kk), sbSamp + fudgeSamp]; 
                    sbInds = [sbSamp + fudgeSamp, sbSamp + fudgeSamp + medEndWinSamp(kk)]; % between second bar and baseline
                
                    minMotLinSt(ii,jj,kk).linDiff(1).totDiff = relDiff;
                    minMotLinSt(ii,jj,kk).linDiff(1).inds = fbInds;
                    minMotLinSt(ii,jj,kk).linDiff(2).inds = sbInds;
                
                    fbDiff = relDiff(fbInds(1):fbInds(2));
                    sbDiff = relDiff(sbInds(1):sbInds(2));
                
                    minMotLinSt(ii,jj,kk).linDiff(1).diff = fbDiff;
                    minMotLinSt(ii,jj,kk).linDiff(2).diff = sbDiff;
                    
                end
                
            end
            
        end
        
    end
    
end










end