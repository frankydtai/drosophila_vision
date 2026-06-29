function outputSt = addDSIandLItoMinMovLinwSigBarSigmoid(addedMinMovSt)

% function addDSIandLItoMinMovLinwSigBarSigmoid(addedMinMovSt)
%
% this function is to be used internally in calcMinMovingBarBasedOnSingleBar
% its input is the output from addCompStatForMinMovLinwSigBar and it adds
% the computation of DSI (PD-ND)/PD for both linear and data and
% computation of LI (D-L)/L
%
% This function is the same as addDSIandLItoMinMovLinwSigBar only
% claculates stats to weighted sigmoid and linear only (no rlin, arlin and
% elin)



outputSt = addedMinMovSt; 

relPD = outputSt(1,1,1).normParameters.PD;

if relPD == -1
    pdInd = [1,2];
else
    pdInd = [2,1];
end

datSiz = size(outputSt);

for ii=1:datSiz(1)
    
    for jj=1:datSiz(2)
        
        tempPD = outputSt(ii,jj,pdInd(1)).stat.table;
        tempND = outputSt(ii,jj,pdInd(2)).stat.table;
        
        datDSI = (tempPD.maxVal('dat') - tempND.maxVal('dat'))/tempPD.maxVal('dat');
        linDSI = (tempPD.maxVal('lin') - tempND.maxVal('lin'))/tempPD.maxVal('lin');
        wSigDSI = (tempPD.maxVal('wSig') - tempND.maxVal('wSig'))/tempPD.maxVal('wSig');
        
        pdLI = (tempPD.maxVal('dat') - tempPD.maxVal('lin'))/tempPD.maxVal('lin'); 
        ndLI = (tempND.maxVal('dat') - tempND.maxVal('lin'))/tempND.maxVal('lin'); 
        
        pdWSLI = (tempPD.maxVal('dat') - tempPD.maxVal('wSig'))/tempPD.maxVal('wSig'); 
        ndWSLI = (tempND.maxVal('dat') - tempND.maxVal('wSig'))/tempND.maxVal('wSig'); 
        
        outputSt(ii,jj,pdInd(1)).stat.DS = table(datDSI, linDSI, wSigDSI, pdLI, ndLI, pdWSLI, ndWSLI);
    end
    
end
            
            








end