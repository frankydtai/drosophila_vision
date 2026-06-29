function minMotLinSt = calculateMinMotExtLinSumDiffOld2(pStruct)

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
calcWinTime = 400; % in ms
calWinSampSize = calcWinTime * sampFac;


minMotLinSt = calcMinMotExtLinComp(pStruct);

datSiz = size(minMotLinSt);


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
                
                    bFrames = minMotLinSt(ii,jj,kk).data.table{1,{'fAppear';'sAppear'}};
                    fbTimeInd = ismember(minMotLinSt(ii,jj,kk).data.align.meanPos(:,2), bFrames(1));
                    fbSamp = minMotLinSt(ii,jj,kk).data.align.meanPos(fbTimeInd,1);
                    sbTimeInd = ismember(minMotLinSt(ii,jj,kk).data.align.meanPos(:,2), bFrames(2));
                    sbSamp = minMotLinSt(ii,jj,kk).data.align.meanPos(sbTimeInd,1);
                    
                     if sbSamp-fbSamp < calWinSampSize
                         fbInds = [sbSamp - calWinSampSize + fudgeSamp, sbSamp + fudgeSamp-1]; % before second bar
                     else
                         fbInds = [fbSamp + fudgeSamp, fbSamp + fudgeSamp + calWinSampSize-1]; % after first bar
                     end
                    sbInds = [sbSamp + fudgeSamp, sbSamp + fudgeSamp + calWinSampSize-1]; % between second bar constant duration
                
                    minMotLinSt(ii,jj,kk).linDiff(1).totDiff = relDiff;
                    minMotLinSt(ii,jj,kk).linDiff(1).inds = fbInds;
                    minMotLinSt(ii,jj,kk).linDiff(2).inds = sbInds;
                
                    fbDiff = relDiff(fbInds(1):fbInds(2));
                    sbDiff = relDiff(sbInds(1):sbInds(2));
                    
                    fbRatio = sum(relDat(fbInds(1):fbInds(2))) / sum(relSum(fbInds(1):fbInds(2)));
                    sbRatio = sum(relDat(sbInds(1):sbInds(2))) / sum(relSum(sbInds(1):sbInds(2)));
                    
                    fbMaxResp = quantile(relDat(fbInds(1):fbInds(2)), 0.95);
                    sbMaxResp = quantile(relDat(sbInds(1):sbInds(2)), 0.95);
                    
                    fbMaxLinResp = quantile(relSum(fbInds(1):fbInds(2)), 0.95);
                    sbMaxLinResp = quantile(relSum(sbInds(1):sbInds(2)), 0.95);
                    
                    minMotLinSt(ii,jj,kk).linDiff(1).diff = fbDiff;
                    minMotLinSt(ii,jj,kk).linDiff(2).diff = sbDiff;
                    
                    minMotLinSt(ii,jj,kk).linDiff(1).ratio = fbRatio;
                    minMotLinSt(ii,jj,kk).linDiff(2).ratio = sbRatio;
                    
                    minMotLinSt(ii,jj,kk).linDiff(1).maxResp = [fbMaxResp, fbMaxLinResp];
                    minMotLinSt(ii,jj,kk).linDiff(2).maxResp = [sbMaxResp, sbMaxLinResp];
                    
                end
                
            end
            
        end
        
    end
    
end










end