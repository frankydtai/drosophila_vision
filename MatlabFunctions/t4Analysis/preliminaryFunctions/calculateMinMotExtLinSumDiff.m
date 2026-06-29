function minMotLinSt = calculateMinMotExtLinSumDiff(pStruct)

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
% minMotLinSt -     Uses the output structure from calcMinMotExtLinComp
%                   (which has the sum of linear responses, and adds the following fields
% .linDiff -        a strucure computed for each bar that contain the following fields (appear only in positions where a comparison can be made - e.g. no diagonal)
%   .totDiff -      difference between baseline subtarcted data and linear
%                   sum (appears only for first bar)
%   .inds -         indices upon which the calculations for each bar were
%                   made
%   .diff -         totDiff values between the relevant indices.
%   .maxResp -      index and values for max data response between the
%                   above indices
%   .maxLinResp -   Same but for linear sum of the resposne
%   .linInd -       linearity index: (maxResp - maxLinResp)/ maxLinResp



fudgeTime = 45; %in ms , used as time between appearance of bar and actual resp; a bit bigger than beginning of resp, but intended to catch peak
sampFac = 20; % 20kHz 
fudgeSamp = fudgeTime * sampFac;
relQ = 0.995;



minMotLinSt = calcMinMotExtLinComp(pStruct);

datSiz = size(minMotLinSt);


%% calculating difference

for ii=1:datSiz(1)
    
    for jj=1:datSiz(2)
        
%         if ii~=jj
        
            for kk=1:datSiz(3)
                
                if ~isempty(minMotLinSt(ii,jj,kk).data) %  condition is for upper side of matrix in timeDiff==0
                    
                    % relTime = minMotLinSt(ii,jj,kk).subData.baseSub(:,1);
                    relDat = minMotLinSt(ii,jj,kk).subData.baseSub(:,2);
                    
                    
                    % getting bar appearance indices to calculate difference
                    % just there
                    bFrames = minMotLinSt(ii,jj,kk).data.table{1,{'fAppear';'fDisappear';'sAppear';'sDisappear'}};
                    fbTimeInd = ismember(minMotLinSt(ii,jj,kk).data.align.meanPos(:,2), bFrames([1,2]));
                    fbSamp = minMotLinSt(ii,jj,kk).data.align.meanPos(fbTimeInd,1);
                    sbTimeInd = ismember(minMotLinSt(ii,jj,kk).data.align.meanPos(:,2), bFrames([3,4]));
                    sbSamp = minMotLinSt(ii,jj,kk).data.align.meanPos(sbTimeInd,1);
                    
                    fbInds = [fbSamp(1) + fudgeSamp, fbSamp(2) + fudgeSamp];  
                    sbInds = [sbSamp(1) + fudgeSamp, sbSamp(2) + fudgeSamp]; 
                    
                    minMotLinSt(ii,jj,kk).linDiff(1).inds = fbInds;
                    minMotLinSt(ii,jj,kk).linDiff(2).inds = sbInds;
                    
                    [fbMaxResp, fbMaxInd] = findQandInd(relDat, fbInds(1), fbInds(2), relQ);
                    [sbMaxResp, sbMaxInd] = findQandInd(relDat, sbInds(1), sbInds(2), relQ);
                    
                    minMotLinSt(ii,jj,kk).linDiff(1).maxResp = [fbMaxInd, fbMaxResp];
                    minMotLinSt(ii,jj,kk).linDiff(2).maxResp = [sbMaxInd, sbMaxResp];
                    
                    if ii ~= jj
                        
                        relSum = minMotLinSt(ii,jj,kk).linSum(:,2);
                        relDiff = relDat - relSum;
                        minMotLinSt(ii,jj,kk).linDiff(1).totDiff = relDiff;
                        
                        fbDiff = relDiff(fbInds(1):fbInds(2));
                        sbDiff = relDiff(sbInds(1):sbInds(2));
                        
                        
                        [fbMaxLinResp, fbMaxLinInd] = findQandInd(relSum, fbInds(1), fbInds(2), relQ);
                        [sbMaxLinResp, sbMaxLinInd] = findQandInd(relSum, sbInds(1), sbInds(2), relQ);
                        
                        minMotLinSt(ii,jj,kk).linDiff(1).diff = fbDiff;
                        minMotLinSt(ii,jj,kk).linDiff(2).diff = sbDiff;
                        
                        minMotLinSt(ii,jj,kk).linDiff(1).maxLinResp = [fbMaxLinInd, fbMaxLinResp];
                        minMotLinSt(ii,jj,kk).linDiff(2).maxLinResp = [sbMaxLinInd, sbMaxLinResp];
                
                        minMotLinSt(ii,jj,kk).linDiff(1).linInd = (fbMaxResp - fbMaxLinResp)/fbMaxLinResp;
                        minMotLinSt(ii,jj,kk).linDiff(2).linInd = (sbMaxResp - sbMaxLinResp)/sbMaxLinResp;
                
                        minMotLinSt(ii,jj,kk).linDiff(1).corr = corr(relDat(fbInds(1):fbInds(2)), relSum(fbInds(1):fbInds(2)));
                        minMotLinSt(ii,jj,kk).linDiff(2).corr = corr(relDat(sbInds(1):sbInds(2)), relSum(sbInds(1):sbInds(2)));
                        
                        
                    end
                    
                end
                
            end
            
        
        
    end
    
end










end

%%


function [qVal, qInd] = findQandInd(relVec, startInd, stopInd, quan)

% This sub function get indices and finds the max and its ind within the
% relevant range. If max is within the first 100 samples of the data, the
% function rewrites the max as the values in the middle of the relevant
% range (since the response is going down)

sampThresh = 100;

qVal = quantile(relVec(startInd:stopInd), quan);
preQInd = find(relVec(startInd:stopInd) > qVal, 1, 'first');

if preQInd < sampThresh
    qInd = round(mean([startInd, stopInd]));
    qVal = relVec(qInd);
else
    qInd = startInd - 1 + preQInd;
end




end



