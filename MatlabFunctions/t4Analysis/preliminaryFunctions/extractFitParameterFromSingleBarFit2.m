function paramTable = extractFitParameterFromSingleBarFit2(allSigFit)

% function extractFitParameterFromSingleBarFit2(allSigFit)
% 
% This is a newer iteration of the function using applyFunctionOnT4dataset(t4Cells, 'singleBar', @fitFuncToSingleBarSt)
% to generate the fit structure (dosn't use names just position in the
% structure)

numCells = length(allSigFit);

durVal = [20, 40, 80, 160]; % in ms

paramTable = table;

for ii=1:numCells
    
    relSt = allSigFit(ii).result{1};
    peakExt = allSigFit(ii).posExtPeak;
    cellPD = sign(allSigFit(ii).posinhPeak - peakExt);
    
    datSiz = size(relSt);
    
    fType = cell(datSiz(1)*datSiz(2), 1);
    maxVal = nan(datSiz(1)*datSiz(2), 1);
    maxTime = maxVal;   minTime = maxVal;   minVal = maxVal; 
    slope = maxVal;     tau = maxVal;       RsqLin = maxVal;
    RsqExp = maxVal;    zeroTime = maxVal;  position = maxVal;
    duration = maxVal;
    counter = 0;
    
    for jj=1:datSiz(1)
        
        for kk=1:datSiz(2)
            
            counter = counter+1;
            tempFT = relSt(jj,kk).fitType;
            
            position(counter) = jj;
            duration(counter) = durVal(kk);
            
            switch tempFT
                
                case 0
                    fType{counter} = 'None';
                    maxVal(counter) = 0;
                    maxTime(counter) = nan;
                    minVal(counter) = 0;
                    minTime(counter) = nan;
                    slope(counter) = nan;
                    tau(counter) = nan;
                    RsqLin(counter) = nan;
                    RsqExp(counter) = nan;
                    zeroTime(counter) = nan;
                    
                case 2
                    fType{counter} = 'Exp';
                    maxVal(counter) = 0;
                    maxTime(counter) = nan;
                    slope(counter) = nan;
                    RsqLin(counter) = nan;
                    
                    expFitResp = relSt(jj,kk).fitResp(2);
                    tau(counter) = expFitResp.fit.c;
                    RsqExp(counter) = expFitResp.gof.adjrsquare;
                    minVal(counter) = relSt(jj, kk).resp.minVal;
                    minTime(counter) = relSt(jj, kk).resp.minTime;
                    
                    tempA = expFitResp.fit.a;
                    tempB = expFitResp.fit.b;
                    
                    preZTVal = -tau(counter) * log(-tempA/tempB);
                    if isreal(preZTVal)
                        %since fit was done with time re-zeroed to the beginning of the fit window 
                        zFitTimeInd = find(expFitResp.relInds, 1, 'first');
                        allTime = relSt(jj, kk).subData.baseSub(:,1);
                        zeroFitTime = allTime - allTime(zFitTimeInd);
                        fitInd = find(zeroFitTime > preZTVal, 1, 'first');
                        zeroTime(counter) = allTime(fitInd);
                    else
                        zeroTime(counter) = nan; % this should mean something is wrong
                    end
                    
                case 3
                    fType{counter} = 'Both';
                    maxVal(counter) = relSt(jj, kk).resp.maxVal;
                    maxTime(counter) = relSt(jj, kk).resp.maxTime;
                    minVal(counter) = relSt(jj, kk).resp.minVal;
                    minTime(counter) = relSt(jj, kk).resp.minTime;
                    
                    linFitResp = relSt(jj, kk).fitResp(1);
                    slope(counter) = linFitResp.fit.p1;
                    RsqLin(counter) = linFitResp.gof.adjrsquare;
                    tempP2 = linFitResp.fit.p2;
                    
                    preZTVal = -tempP2/slope(counter);
                    zFitTimeInd = find(linFitResp.relInds, 1, 'first');
                    allTime = relSt(jj, kk).subData.baseSub(:,1);
                    zeroFitTime = allTime - allTime(zFitTimeInd);
                    fitInd = find(zeroFitTime > preZTVal, 1, 'first');
                    zeroTime(counter) = allTime(fitInd);
                    
                    tau(counter) = relSt(jj, kk).fitResp(2).fit.c;
                    RsqExp(counter) = relSt(jj, kk).fitResp(2).gof.adjrsquare;     
                    
            end   
            
        end
        
    end
    
    cellNum  = ones(datSiz(1)*datSiz(2), 1) * ii;
    totPos = ones(datSiz(1)*datSiz(2), 1) * datSiz(1);
    orient = ones(datSiz(1)*datSiz(2), 1) * relSt(1,1).data.table.orient; % since orientation is the same for the whole protocol
    cenPos = position - ceil(datSiz(1)/2);
    normPos = (cenPos - peakExt) * cellPD;
    
    tempTable = table(cellNum, totPos, normPos, duration, orient, ...
                      fType, maxVal, maxTime, minVal, minTime, ...
                      slope, tau, zeroTime, RsqLin, RsqExp);
                  
    paramTable = vertcat(paramTable, tempTable);
    
end



end