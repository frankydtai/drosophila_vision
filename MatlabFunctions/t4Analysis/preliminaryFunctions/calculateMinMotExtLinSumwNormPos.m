function minMotExtwSigBarSt = calculateMinMotExtLinSumwNormPos(minMotExtProtSt, sigBarProtSt)

% function minMotExtwSigBarSt = calculateMinMotExtLinSumwNormPos(minMotProtSt, sigBarProtSt)
%
% This function is basically just calculateMinMotExtLinSumDiff with the
% addition of correcting all the positions based on the normalized position
% (zeroed to excitatory max and flipped if inhibition is on the leading
% side) and adding Direction Selectivity Indices for data and linear sum
%
% INPUT
% minMotEXtProtSt -     protocol structure from excitatory minimal motion
%                       protocol
% sigBarProtSt -        porotocol strucutre from single bar protocol
%
% OUTPUT
% same as in calculateMinMotExtLinSumDiff with .normPos added for each
% field. Also calculates DSI for Data and linear sum seperately for both
% bars. 




minMotSt = calculateMinMotExtLinSumDiff(minMotExtProtSt); 

alignSBST = generateAlignedSingleBarStwMinMax(sigBarProtSt);

maxEPosV = alignSBST(end,end).maxExtPosVal;
PDFlag = sign(alignSBST(end,end).maxInhPosVal - maxEPosV);

relPos = unique(minMotExtProtSt.gratingTable.FBPos);

normPos = (relPos - maxEPosV) * PDFlag;

minMotExtwSigBarSt = minMotSt;

datSize = size(minMotSt);

for ii=1:datSize(1)
    
    for jj=1:datSize(2)
        
        for kk=1:datSize(3)
            
            minMotExtwSigBarSt(ii,jj,kk).normPos = [normPos(ii), normPos(jj)]; 
            
            if ~isempty(minMotExtwSigBarSt(ii,jj,kk).linSum)
                
                minMotExtwSigBarSt(ii,jj,kk).linDiff(1).singleBarResp = minMotExtwSigBarSt(ii,ii,kk).linDiff(1).maxResp(2);
                minMotExtwSigBarSt(ii,jj,kk).linDiff(2).singleBarResp = minMotExtwSigBarSt(jj,jj,kk).linDiff(1).maxResp(2);
                
            
                if normPos(jj) - normPos(ii) > 0 && kk > 1 
                
                    % first bar prefered and non-prefered data and linear
                    datPDF = minMotExtwSigBarSt(ii,jj,kk).linDiff(1).maxResp(2);
                    linPDF = minMotExtwSigBarSt(ii,jj,kk).linDiff(1).maxLinResp(2);
                    
                    datNDF = minMotExtwSigBarSt(jj,ii,kk).linDiff(1).maxResp(2);
                    linNDF = minMotExtwSigBarSt(jj,ii,kk).linDiff(1).maxLinResp(2);
    
                    % second bar prefered and non-prefered data and linear
                    datPDS = minMotExtwSigBarSt(ii,jj,kk).linDiff(2).maxResp(2);
                    linPDS = minMotExtwSigBarSt(ii,jj,kk).linDiff(2).maxLinResp(2);
                
                    datNDS = minMotExtwSigBarSt(jj,ii,kk).linDiff(2).maxResp(2);
                    linNDS = minMotExtwSigBarSt(jj,ii,kk).linDiff(2).maxLinResp(2);
                
                    minMotExtwSigBarSt(ii,jj,kk).linDiff(1).datDSI = (datPDF - datNDF)/datPDF;
                    minMotExtwSigBarSt(ii,jj,kk).linDiff(1).linDSI = (linPDF -linNDF)/linPDF;
                    
                    minMotExtwSigBarSt(ii,jj,kk).linDiff(2).datDSI = (datPDS - datNDS)/datPDS;
                    minMotExtwSigBarSt(ii,jj,kk).linDiff(2).linDSI = (linPDS -linNDS)/linPDS;
                    
                end
            end
            
        end
        
    end
    
end
    





end