function minMovStatSt = addCompStatForMinMovLinwSigBar(minMovSingleSt)

% function minMovStatSt = addCompStatForMinMovLinwSigBar(minMovSingleSt)
%
% This function adds measurements and stats to the linear comparison
% between singlebar and minMov protocols
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
            
            maxVal = zeros(5,1);
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
                %maxIndL = 2;
            else
                maxVal(2) = linDat(maxIndL);
                maxInd(2) = maxIndL;
                maxTime(2) = relTime(maxIndL);
                riseInd(2) = find(linDat(1:maxIndL) < linDat(maxIndL)/maxPercent, 1, 'last'); % walking back from the peak
                riseTime(2) = relTime(riseInd(2));
            end
            
            recLinDat = minMovSingleSt(ii,jj,kk).recLinSum(:,1);
            smRecLin = smooth(recLinDat, winSiz);
            [~, maxIndRL] = max(smRecLin);
            
            if maxIndRL < 100 % no linear resp
                maxVal(3) = nan;
                maxInd(3) = 2;
                maxTime(3) = nan;
                riseInd(3) = 2;
                riseTime(3) = nan;
                %maxIndL = 2;
            else
                maxVal(3) = recLinDat(maxIndRL);
                maxInd(3) = maxIndRL;
                maxTime(3) = relTime(maxIndRL);
                riseInd(3) = find(recLinDat(1:maxIndRL) < recLinDat(maxIndRL)/maxPercent, 1, 'last'); % walking back from the peak
                riseTime(3) = relTime(riseInd(3));
            end
            
            enhLinDat = minMovSingleSt(ii,jj,kk).enhLinSum(:,1);
            smEnhLin = smooth(enhLinDat, winSiz);
            [~, maxIndEL] = max(smEnhLin);
            
            if maxIndEL < 100 % no linear resp
                maxVal(4) = nan;
                maxInd(4) = 2;
                maxTime(4) = nan;
                riseInd(4) = 2;
                riseTime(4) = nan;
                %maxIndL = 2;
            else
                maxVal(4) = enhLinDat(maxIndEL);
                maxInd(4) = maxIndEL;
                maxTime(4) = relTime(maxIndEL);
                riseInd(4) = find(enhLinDat(1:maxIndEL) < enhLinDat(maxIndEL)/maxPercent, 1, 'last'); % walking back from the peak
                riseTime(4) = relTime(riseInd(4));
            end
            
            arLinDat = minMovSingleSt(ii,jj,kk).altRecLinSum(:,1);
            smARLin = smooth(arLinDat, winSiz);
            [~, maxIndAR] = max(smARLin);
            
            if maxIndAR < 100 % no linear resp
                maxVal(5) = nan;
                maxInd(5) = 2;
                maxTime(5) = nan;
                riseInd(5) = 2;
                riseTime(5) = nan;
                %maxIndL = 2;
            else
                maxVal(5) = arLinDat(maxIndAR);
                maxInd(5) = maxIndAR;
                maxTime(5) = relTime(maxIndAR);
                riseInd(5) = find(arLinDat(1:maxIndAR) < arLinDat(maxIndAR)/maxPercent, 1, 'last'); % walking back from the peak
                riseTime(5) = relTime(riseInd(5));
            end
            
            
            minMovStatSt(ii,jj,kk).stat.table = table(maxVal, maxTime, riseTime, maxInd, riseInd, 'rownames', {'dat'; 'lin'; 'recLin'; 'enhLin'; 'arLin'});
            
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
            linRelDat = [linDat, recLinDat, enhLinDat, arLinDat];
            
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
            
            minMovStatSt(ii,jj,kk).stat.xCorr = table(maxCorr, timeLag, regCorr, 'rownames', {'Lin'; 'recLin'; 'enhLin'; 'arLin'});
            
           
        end
        
    end
    
end






end
