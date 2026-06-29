function sigBarSt = addNormMaxAndMinToSingleBarAligned(alignedSingleBarSt)

% function sigBarSt = addNormMAxAndMinToSingleBarAligned(alignedSingleBarSt)
%
% This function takes the finished output from generateAlignedSingleBarStwMinMax
% and adds normMax and normMin to it. For all the speed seperately and for
% the total cell. It is used internally in
% generateAlignedSingleBarStwMinMax after the alignement and max
% calculation has been done
%
% iNPUT
%
% alignedSingleBarSt -      used internally within
%                           generateAlignedSingleBarStwMinMax, but uses the output of that function
%                           as its input
% OUTPUT
%
% sigBarSt -                to the regualr fields that describe each
%                           position, duration combination, this function adds a numPos+1 numDur+1
%                           subStructure that incluedes the following fields:
%   max/min:                extracted max/min from all pos dur combintations 
%   normMax/Min:            normalized max and min across position by
%                           duration
%   maxExt/inhPosInd:       position in which max excitation/inhibition was measured (normalized per speed and summed for designated speeds)
%   maxExt/InhPosVal:       the value of that positon relative to the
%                           center of the stimulus
%   FWHM -                  full width half maximun at each position (if a
%                           response is found)



% indices for speeds to indlues in the total normMax/Min computation
extRespInds = 1:4;
inhibRespInds = 3:4;

sigBarSt = alignedSingleBarSt;

datSiz = size(alignedSingleBarSt);

allMax = nan(datSiz);
allMin = allMax;
allFWHM = allMax;

for ii=1:datSiz(1)
    
    for jj=1:datSiz(2)
        
        allMax(ii,jj) = alignedSingleBarSt(ii,jj).resp.maxVal;
        allMin(ii,jj) = alignedSingleBarSt(ii,jj).resp.minVal;
        if isfield(alignedSingleBarSt(ii,jj).resp, 'FWHM')
            allFWHM(ii,jj) = alignedSingleBarSt(ii,jj).resp.FWHM;
        end
    end
    
end

normMax = allMax * diag(1./max(allMax));
normMin = allMin * diag(1./min(allMin));
    
sigBarSt(datSiz(1)+1, datSiz(2)+1).max = allMax;
sigBarSt(datSiz(1)+1, datSiz(2)+1).min = allMin;
sigBarSt(datSiz(1)+1, datSiz(2)+1).normMax = normMax;
sigBarSt(datSiz(1)+1, datSiz(2)+1).normMin = normMin;
sigBarSt(datSiz(1)+1, datSiz(2)+1).FWHM = allFWHM;

maxSum = nansum(normMax(:,extRespInds), 2);
minSum = nansum(normMin(:,inhibRespInds), 2);
[~, maxI] = max(maxSum);
[~, minI] = max(minSum);


sigBarSt(datSiz(1)+1, datSiz(2)+1).maxExtPosInd = maxI;
sigBarSt(datSiz(1)+1, datSiz(2)+1).maxExtPosVal = sigBarSt(maxI, 1).data.table.position;
sigBarSt(datSiz(1)+1, datSiz(2)+1).minInhPosInd = minI;
sigBarSt(datSiz(1)+1, datSiz(2)+1).maxInhPosVal = sigBarSt(minI, 1).data.table.position;



end