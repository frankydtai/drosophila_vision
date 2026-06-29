function outputSt = addDSIandLItoMinMovLinwSigBar(addedMinMovSt)

% function addDSIandLItoMinMovLinwSigBar(addedMinMovSt)
%
% this function is to be used internally in calcMinMovingBarBasedOnSingleBar
% its input is the output from addCompStatForMinMovLinwSigBar and it adds
% the computation of DSI (PD-ND)/PD for both linear and data and
% computation of LI (D-L)/L



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
        rlinDSI = (tempPD.maxVal('recLin') - tempND.maxVal('recLin'))/tempPD.maxVal('recLin');
        elinDSI = (tempPD.maxVal('enhLin') - tempND.maxVal('enhLin'))/tempPD.maxVal('enhLin');
        arlinDSI = (tempPD.maxVal('arLin') - tempND.maxVal('arLin'))/tempPD.maxVal('arLin');
        
        pdLI = (tempPD.maxVal('dat') - tempPD.maxVal('lin'))/tempPD.maxVal('lin'); 
        ndLI = (tempND.maxVal('dat') - tempND.maxVal('lin'))/tempND.maxVal('lin'); 
        
        pdRLI = (tempPD.maxVal('dat') - tempPD.maxVal('recLin'))/tempPD.maxVal('recLin'); 
        ndRLI = (tempND.maxVal('dat') - tempND.maxVal('recLin'))/tempND.maxVal('recLin'); 
        
        pdELI = (tempPD.maxVal('dat') - tempPD.maxVal('enhLin'))/tempPD.maxVal('enhLin'); 
        ndELI = (tempND.maxVal('dat') - tempND.maxVal('enhLin'))/tempND.maxVal('enhLin'); 
        
        pdARLI = (tempPD.maxVal('dat') - tempPD.maxVal('arLin'))/tempPD.maxVal('arLin'); 
        ndARLI = (tempND.maxVal('dat') - tempND.maxVal('arLin'))/tempND.maxVal('arLin'); 
        
        outputSt(ii,jj,pdInd(1)).stat.DS = table(datDSI, linDSI, rlinDSI, elinDSI, arlinDSI, pdLI, ndLI, pdRLI, ndRLI, pdELI, ndELI, pdARLI, ndARLI);
    end
    
end
            
            








end