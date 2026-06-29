function minMovStatSt = addCompStatForMinMovLinwSigBarSigmoid(minMovSingleSt)

% function minMovStatSt = addCompStatForMinMovLinwSigBarSigmoid(minMovSingleSt)
%
% This function adds measurements and stats to the linear comparison
% between singlebar and minMov protocols. It is a version of addCompStatForMinMovLinwSigBar
% that simply does not compute rlin arlin and elin
%
% Function is to be used internally within calcMinMovingBarBasedOnSingleBar
% and use its output as the input. 
%
% NOTE! when linear sum is rectified/enhanced the effect is applied only after the stimulus  


winSiz = 500; % smoothing window size (compatible with fastest speed)
maxPercent = 10; % to determine rise time
datSiz = size(minMovSingleSt);
sampToMSConv = 20; % since smapling rate is 20K

minMovStatSt = minMovSingleSt;
% find max for subData and linSum data

for ii=1:datSiz(1)
    
    for jj=1:datSiz(2)
        
        for kk=1:datSiz(3)
            
            maxVal = zeros(3,1);
            maxInd = maxVal; maxTime = maxVal; riseTime = maxVal; riseInd = maxVal;
            
            if isempty(minMovSingleSt(ii,jj,kk).subData)
                continue
            end
            
%             [ii,jj,kk]
            % statistics for DATA
            relDat = minMovSingleSt(ii,jj,kk).subData.baseSub(:,2);
            relTime = minMovSingleSt(ii,jj,kk).subData.baseSub(:,1);
            smDat = smooth(relDat, winSiz);
            [~, maxIndD] = max(smDat);
            
            if maxIndD < 100 % no depolarisation
                maxVal(1) = nan;
                maxInd(1) = 1;
                maxTime(1) = nan;
                riseInd(1) = 1;
                riseTime(1) = nan;
                maxIndD = 2; % so fit function will not error
            else
                maxVal(1) = relDat(maxIndD);
                maxInd(1) = maxIndD;
                maxTime(1) = relTime(maxIndD);
                riseInd(1) = find(relDat(1:maxIndD) < relDat(maxIndD)/maxPercent, 1, 'last'); % walking back from the peak
                riseTime(1) = relTime(riseInd(1));
            end
            
            
            % statistics for LinSum
            linDat = minMovSingleSt(ii,jj,kk).linSum(:,2);
            smLin = smooth(linDat, winSiz);
            [~, maxIndL] = max(smLin);
            
            if maxIndL < 100 % no linear resp
                maxVal(2) = nan;
                maxInd(2) = 2;
                maxTime(2) = nan;
                riseInd(2) = 2;
                riseTime(2) = nan;
                
            else
                maxVal(2) = linDat(maxIndL);
                maxInd(2) = maxIndL;
                maxTime(2) = relTime(maxIndL);
                riseInd(2) = find(linDat(1:maxIndL) < linDat(maxIndL)/maxPercent, 1, 'last'); % walking back from the peak
                riseTime(2) = relTime(riseInd(2));
            end
            
            wSigSumDat = minMovSingleSt(ii,jj,kk).wSigSum(:,1);
            smSigSum = smooth(wSigSumDat, winSiz);
            [~, maxIndWS] = max(smSigSum);
            
            if maxIndWS < 100 % no linear resp
                maxVal(3) = nan;
                maxInd(3) = 2;
                maxTime(3) = nan;
                riseInd(3) = 2;
                riseTime(3) = nan;
                
            else
                maxVal(3) = wSigSumDat(maxIndWS);
                maxInd(3) = maxIndWS;
                maxTime(3) = relTime(maxIndWS);
                riseInd(3) = find(wSigSumDat(1:maxIndWS) < wSigSumDat(maxIndWS)/maxPercent, 1, 'last'); % walking back from the peak
                riseTime(3) = relTime(riseInd(3));
            end
            
            
            minMovStatSt(ii,jj,kk).stat.table = table(maxVal, maxTime, riseTime, maxInd, riseInd, 'rownames', {'dat'; 'lin'; 'wSig'});
            
%% Correlation of only the rise phase - proved to be not so useful            
%             if maxIndD+100 > length(relDat) %in responses where there is only hyperpolarization
%                 regRange = [riseInd(1), maxIndD; maxIndD-100, length(relDat)]; % regressing seperately from rise to max and max to end 
%             else
%                 regRange = [riseInd(1), maxIndD; maxIndD+1, length(relDat)]; % regressing seperately from rise to max and max to end 
%             end
%             
%             % regressing dat on lin
%             maxCorr = zeros(2,1);
%             timeLag = maxCorr;
%             %rSq = slope; intercept = slope;
%             
%             for rr = 1
%                 regDat = relDat(regRange(rr,1):regRange(rr,2));
%                 regLin = linDat(regRange(rr,1):regRange(rr,2));
%                 
% %                 [tempFit, tempGOF] = fit(regLin, regDat, 'poly1');
% %                 slope(rr) = tempFit.p1;
% %                 intercept(rr) = tempFit.p2;
% %                 rSq(rr) = tempGOF.rsquare;
%                 [riseCorrV, riseLag] = xcorr(regLin, regDat, 'coeff');
%                 [maxRCV, maxRCI] = max(riseCorrV);
%                 maxCorr(2) = maxRCV;
%                 timeLag(2) = riseLag(maxRCI)/sampToMSConv;
% 
%             end
%             
             %minMovStatSt(ii,jj,kk).stat.fit = table(slope, intercept, rSq, 'rownames', {'rise'; 'decay'});
%%          
            linRelDat = [linDat, wSigSumDat];
            
            maxCorr = nan(size(linRelDat, 2), 1);
            timeLag = maxCorr;
            regCorr = maxCorr;
            
            for lInd = 1:size(linRelDat, 2)

                [corrV, corrL] = xcorr(relDat, linRelDat(:,lInd), 'coeff');
                [maxCV, maxCI] = max(corrV);
                
                regCorr(lInd) = corr(relDat, linRelDat(:,lInd));
                
                maxCorr(lInd) = maxCV;
                timeLag(lInd) = corrL(maxCI)/sampToMSConv;
            
            end
            
            minMovStatSt(ii,jj,kk).stat.xCorr = table(maxCorr, timeLag, regCorr, 'rownames', {'Lin'; 'wSig'});
            
           
        end
        
    end
    
end






end
