function minMotSt = calcMinMotInhibLinComp(pStruct)

% function minMotSt = calcMinMotInhibLinComp(pStruct)
%
% This function is designed specifically for minimal motion protocols that
% probed the inhibitory effect with diagonnaly corrected stim (with gratingTable and not gratingInds). 
% It finds the single bar presentations and generates the predicted linear diff by
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

preFBBaseWin = [-4100, -100]; % samples before first bar appearance to use for baseline calculation

% checking inputs
relTab = pStruct.gratingTable;

assert(all(ismember(['fAppear'; 'sAppear'], relTab.Properties.VariableNames)), ...
           'gratingTable is missing either fAppear or sAppear')

assert(ismember('FBStat', relTab.Properties.VariableNames), 'This function is designed for minMot protocols only')

ufPos = unique(relTab.FBPos);
usPos = unique(relTab.SBPos);
uTD = unique(relTab.timeDiff);
uFBS = unique(relTab.FBStat);

for ii=1:length(usPos) % in case there is more than one second bar pos
    samePosInd(ii) = find(ufPos ==usPos(ii));
end


alignSt = alignProtocolDataByTable(pStruct, 'sAppear'); %aligns to appearance of second bar 

% orginizing mean data

minMotSt = struct;

for ii=1:length(ufPos)
    
    for jj=1:length(usPos)
        
        for tt=1:length(uTD)
            
            for kk=1:length(uFBS)
            
                relInd = ismember(relTab{:, {'FBPos'; 'SBPos'; 'timeDiff';'FBStat'}}, ...
                                 [ufPos(ii), usPos(jj), uTD(tt), uFBS(kk)], 'rows');
            
                if unique(relInd) == 0
                    continue
                else
                    
                    
                    minMotSt(ii, jj, tt, kk).data = alignSt(relInd);
                
                    relDat = minMotSt(ii, jj, tt, kk).data.align.mean;
                    fbAppPos = minMotSt(ii,jj,tt,kk).data.table.fAppear;
                    relMeanPos = minMotSt(ii,jj,tt,kk).data.align.meanPos;
                    fbAppInd = relMeanPos(relMeanPos(:,2) == fbAppPos, 1);
                    baseInds = fbAppInd + preFBBaseWin;
                    zeroInd = find(relDat(:,1) > 0, 1, 'first');
                
                    baseVal = mean(relDat(baseInds(1):baseInds(2), 2));
                    baseSubResp = relDat;
                    baseSubResp(:,2) = baseSubResp(:,2) - baseVal;
                
                    minMotSt(ii, jj, tt, kk).subData.baseSub = baseSubResp;
                    minMotSt(ii, jj, tt, kk).subData.baseline = baseVal;
                    minMotSt(ii, jj, tt, kk).subData.zeroInd = zeroInd;
                    minMotSt(ii, jj, tt, kk).subData.length = size(baseSubResp, 1);
                    
                end
                
            end
            
        end
        
    end
    
end
             
for tt=1:length(uTD)
    
    for ii=1:length(usPos)
        relFBInd = samePosInd(ii);
        
        sbTemp = minMotSt(relFBInd,ii, tt, 1); % since for second bar alone FBstat is meaningless
        
        for jj=1:length(ufPos)
            
            for kk=1:length(uFBS)
            
                if relFBInd==jj || isempty(minMotSt(jj, ii, tt, kk).data)
                    
                    continue
                    
                else
                
                    % instead of estimating, this actually takes the mean
                    % position for the second bar appearance and find the
                    % appropriate time to copy the linSum response to
                    relSt = minMotSt(jj,ii,tt,kk);
                    totLen = size(relSt.subData.baseSub, 1);
                    relZeroInd = relSt.subData.zeroInd;
                    
                    sApInd = sbTemp.subData.zeroInd;
                    sPreDiff = relZeroInd - sApInd;
                
                    if sPreDiff > 0
                        sTempMean = padarray(sbTemp.subData.baseSub(:,2), [sPreDiff, 0], 0, 'pre');
                    else
                        sTempMean = sbTemp.subData.baseSub(abs(sPreDiff)+1:end,2);
                    end
                    sbLen = sbTemp.subData.length + sPreDiff;
                    
                    sPostDiff = totLen - sbLen;
                    
                    if sPostDiff > 0
                        sTempMean = padarray(sTempMean, [sPostDiff, 0], 0, 'post');
                    else
                        sTempMean = sTempMean(1:end-abs(sPostDiff));
                    end
                    
                    minMotSt(jj, ii, tt, kk).linDiff = [relSt.subData.baseSub(:,1), relSt.subData.baseSub(:,2) - sTempMean];
                    
                end
                
            end
            
        end
        
    end
    
end




                

end
       
       
       
       