function minMotSt = calcMinMotExtLinComp(pStruct)

% function minMotSt = calcMinMotExtLinComp(pStruct)
%
% This function is designed specifically for minimal motion protocols that
% are diagonnaly corrected (with gratingTable and not gratingInds). It find
% the single bar presentations and generates the predicted linear sum by
% shifting them appropriately. 
%
%
% INPUT 
%
% pStruct -         minMot protocol with manual addition of bar
%                   presentation frames to gratingTable. 
%                   specifically fAppear and sAppear 
%
% OUTPUT
% minMotSt -        structure that organizes the output from the minMot
%                   protocol. contains the following fields for each
%                   [firstBar, secondBar, timing] 
%
%  .data -
%    .align -       Generate by alignProtocolDataByTable while alinging to
%                   fAppear. Contain the following fields
%       .rep -      all the individual repeats for the stim
%       .mean -     mean response
%       .meanPos-   mean position and timing matrix (timimng corresponds to
%                   the mean fieid
%    .table -       relevant table row for the above data
%
%  .subData -       baseline subtracted data. Contains the following fields
%     .baseline -   the calculated baseline that was subtracted
%     .zeroInd -    index to when the time stamp crosses zero
%     .length -     number of samples in data
%     .baseSub -    baseline subtracted data
%  .linSum -        if applicable timing and data vector constructed by a
%                   linear summation of the individual bar responses



% base parameters

preStimBaseWin = [-250, -50]; % ms before stim to use for baseline calculation

% checking inputs
relTab = pStruct.gratingTable;

assert(all(ismember(['fAppear'; 'sAppear'], relTab.Properties.VariableNames)), ...
           'gratingTable is missing either fAppear or sAppear')

assert(ismember('FBStat', relTab.Properties.VariableNames), 'This function is designed for minMot protocols only')
assert(all(unique(relTab.FBStat) == 0), 'this function is designed for FBStat zero only')

ufPos = unique(relTab.FBPos);
usPos = unique(relTab.SBPos);
uTD = unique(relTab.timeDiff);


assert(all(ufPos == usPos), 'FBPos must be equal to SBPos')

alignSt = alignProtocolDataByTable(pStruct, 'fAppear');

% orginizing mean data

minMotSt = struct;

for ii=1:length(ufPos)
    
    for jj=1:length(usPos)
        
        for tt=1:length(uTD)
            
            relInd = ismember(relTab{:, {'FBPos'; 'SBPos'; 'timeDiff'}}, [ufPos(ii), usPos(jj), uTD(tt)], 'rows');
            
            if unique(relInd) == 0
                continue
            else
            
                minMotSt(ii, jj, tt).data = alignSt(relInd);
                
                relDat = minMotSt(ii, jj, tt).data.align.mean;
                baseInds = arrayfun(@(x) find(relDat(:,1) > x, 1, 'first'), preStimBaseWin);
                zeroInd = find(relDat(:,1) > 0, 1, 'first');
                
                baseVal = mean(relDat(baseInds(1):baseInds(2), 2));
                baseSubResp = relDat;
                baseSubResp(:,2) = baseSubResp(:,2) - baseVal;
                
                minMotSt(ii, jj, tt).subData.baseSub = baseSubResp;
                minMotSt(ii, jj, tt).subData.baseline = baseVal;
                minMotSt(ii, jj, tt).subData.zeroInd = zeroInd;
                minMotSt(ii, jj, tt).subData.length = size(baseSubResp, 1);
                
            end
            
        end
        
    end
    
end
             
for tt=1:length(uTD)
    
    for ii=1:length(ufPos)
        
        fbTemp = minMotSt(ii, ii, tt);
        
        for jj=1:length(usPos)
            
            sbTemp = minMotSt(jj, jj, tt);
            
            if ii==jj || isempty(minMotSt(ii, jj, tt).data)
                
                continue
                
            else
                
                % instead of estimating, this actually takes the mean
                % position for the second bar appearance and find the
                % appropriate time to copy the linSum response to
                relSt = minMotSt(ii,jj,tt);
                totLen = size(relSt.subData.baseSub, 1);
                relZeroInd = relSt.subData.zeroInd;
                sApFr = relSt.data.table.sAppear;
                oriSApInd = relSt.data.align.meanPos(relSt.data.align.meanPos(:,2) == sApFr, 1);
                
                fApInd = fbTemp.subData.zeroInd;
                sApInd = sbTemp.subData.zeroInd;
                
                fPreDiff = relZeroInd - fApInd;
                sPreDiff = oriSApInd - sApInd;
                
                if fPreDiff > 0
                    fTempMean = padarray(fbTemp.subData.baseSub(:,2), [fPreDiff, 0], 0, 'pre');
                else
                    fTempMean = fbTemp.subData.baseSub(abs(fPreDiff)+1:end,2);
                end
                fbLen = fbTemp.subData.length + fPreDiff;
                
                if sPreDiff > 0
                    sTempMean = padarray(sbTemp.subData.baseSub(:,2), [sPreDiff, 0], 0, 'pre');
                else
                    sTempMean = sbTemp.subData.baseSub(abs(sPreDiff)+1:end,2);
                end
                sbLen = sbTemp.subData.length + sPreDiff;
                
                fPostDiff = totLen - fbLen;
                sPostDiff = totLen - sbLen;
                
                if fPostDiff > 0
                    fTempMean = padarray(fTempMean, [fPostDiff, 0], 0, 'post');
                else
                    fTempMean = fTempMean(1:end-abs(fPostDiff));
                end
                
                if sPostDiff > 0
                    sTempMean = padarray(sTempMean, [sPostDiff, 0], 0, 'post');
                else
                    sTempMean = sTempMean(1:end-abs(sPostDiff));
                end
                
              
                minMotSt(ii, jj, tt).linSum = [relSt.subData.baseSub(:,1), sTempMean + fTempMean];
                
            end
            
        end
        
    end
    
end




                

end
       
       
       
       