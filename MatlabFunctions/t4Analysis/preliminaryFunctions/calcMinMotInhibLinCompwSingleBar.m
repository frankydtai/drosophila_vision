function [minMotSt, varargout] = calcMinMotInhibLinCompwSingleBar(pInhibMinMotSt, pSingleBarSt)

% function minMotSt = calcMinMotInhibLinComp(pStruct)
%
% This function is an elaboration of calcMinMotInhibLinComp. It doens't just use 
% the single bar data from the minMot protocol but also uses singleBar data
% to generate the linear comparision (by shifting response to the
% appropriate time)
%
%
% INPUT 
%
% inhibMinMotSt -   minMot protocol with manual addition of bar
%                   presentation frames to gratingTable. 
%                   specifically fAppear and sAppear 
% singleBarSt -     singlaeBar protocl with 'appear' in the gratingTable 
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
%
% relSingBarSt-     (optional) if more than one ouptut is asked, this structure 
%                   has the info for the singleBar protocol positions that were 
%                   relevant for this calculation (data and subData as
%                   above). This serves as sanity check and can be used for
%                   plotting



% NOTE!! Currently linSum for FBStat=1 is calculated the same as FBStat=0

% base parameters

preStimBaseWin = [-250, -50]; % samples before first bar appearance to use for baseline calculation

% checking inputs
relTabMM = pInhibMinMotSt.gratingTable;
relTabSB = pSingleBarSt.gratingTable;

assert(all(ismember(['fAppear'; 'sAppear'], relTabMM.Properties.VariableNames)), ...
           'gratingTable is missing either fAppear or sAppear')

assert(ismember('appear', relTabSB.Properties.VariableNames), ...
           'gratingTable is missing either appear')
       

assert(ismember('FBStat', relTabMM.Properties.VariableNames), 'This function is designed for minMot protocols only')
simTD = pInhibMinMotSt.inputParams.stepDur; % stim time for bars presented sim

ufPos = unique(relTabMM.FBPos);
usPos = unique(relTabMM.SBPos);
uTD = unique(relTabMM.timeDiff);
uFBS = unique(relTabMM.FBStat);

uPos = unique(relTabSB.position);
uStD = unique(relTabSB.stimDur);

assert(all(ismember(uStD, uTD)), 'time delays in Singlebar are not identical to minMot') % didn't use equal since minMot also has 0 time delay
assert(all(ismember(ufPos, uPos)), 'some minMot fbPos not present in singleBar protocol') 
assert(all(ismember(usPos, uPos)), 'some minMot sbPos not present in singleBar protocol') 

alignMMSt = alignProtocolDataByTable(pInhibMinMotSt, 'fAppear'); %aligns to appearance of first bar (to take common baseline)
alignSBSt = alignProtocolDataByTable(pSingleBarSt, 'appear'); 

% orginizing mean data


%organizing single bar data just for the relevant positions (in 2 separate
%structures for FB and SB)
singleBarFSt = struct;
singleBarSSt = struct;

for ii=1:length(ufPos)
    for tt=1:length(uTD)     
        
        if uTD(tt) == 0 % sim presentation
            reluTD = simTD;
        else
            reluTD = uTD(tt);
        end
        relSBInd = ismember(relTabSB{:, {'position'; 'stimDur'}}, [ufPos(ii), reluTD], 'rows');
        singleBarFSt(ii,tt).data = alignSBSt(relSBInd);
        relDatSB = singleBarFSt(ii,tt).data.align.mean;
        
        baseInds = arrayfun(@(x) find(relDatSB(:,1) > x, 1, 'first'), preStimBaseWin);
        zeroInd = find(relDatSB(:,1) > 0, 1, 'first');        
        baseVals = relDatSB(baseInds(1):baseInds(2), 2);
        baseSubResp = relDatSB;
        baseSubResp(:,2) = baseSubResp(:,2) - mean(baseVals);
        
        singleBarFSt(ii, tt).subData.baseSub = baseSubResp;
        singleBarFSt(ii, tt).subData.zeroInd = zeroInd;
        singleBarFSt(ii, tt).subData.length = size(baseSubResp, 1);
    end
end

for ii=1:length(usPos)
    for tt=1:length(uTD) 
        if uTD(tt) == 0 % sim presentation
            reluTD = simTD;
        else
            reluTD = uTD(tt);
        end
        relSBInd = ismember(relTabSB{:, {'position'; 'stimDur'}}, [usPos(ii), reluTD], 'rows');
        singleBarSSt(ii,tt).data = alignSBSt(relSBInd);
        relDatSB = singleBarSSt(ii,tt).data.align.mean;
        
        baseInds = arrayfun(@(x) find(relDatSB(:,1) > x, 1, 'first'), preStimBaseWin);
        zeroInd = find(relDatSB(:,1) > 0, 1, 'first');        
        baseVals = relDatSB(baseInds(1):baseInds(2), 2);
        baseSubResp = relDatSB;
        baseSubResp(:,2) = baseSubResp(:,2) - mean(baseVals);
        
        singleBarSSt(ii, tt).subData.baseSub = baseSubResp;
        singleBarSSt(ii, tt).subData.zeroInd = zeroInd;
        singleBarSSt(ii, tt).subData.length = size(baseSubResp, 1);
    end
end

relSingBarSt = singleBarFSt;

if ~ismember(usPos, ufPos) % since usPos is always just 1 position
    relSingBarSt(end+1, :) = singleBarSSt;
end



minMotSt = struct;

for ii=1:length(ufPos)
    
    for jj=1:length(usPos)
        
        for tt=1:length(uTD)
            
            for kk=1:length(uFBS)
            
                relMMInd = ismember(relTabMM{:, {'FBPos'; 'SBPos'; 'timeDiff';'FBStat'}}, ...
                                 [ufPos(ii), usPos(jj), uTD(tt), uFBS(kk)], 'rows');
            
                if unique(relMMInd) == 0
                    continue
                else
                     
                    minMotSt(ii, jj, tt, kk).data = alignMMSt(relMMInd); 
                    minMotSt(ii, jj, tt, kk).subData.fb = singleBarFSt(ii, tt).subData;
                    minMotSt(ii, jj, tt, kk).subData.sb = singleBarSSt(jj, tt).subData;
                    if ufPos(ii) == usPos(jj)
%                         [ii,jj,tt,kk]
                        minMotSt(ii, jj, tt, kk).subData.sb.baseSub(:,2) = 0; % so as to not add the same response twice when sbar is at the same position
                    end
                    
                    relDatMM = minMotSt(ii, jj, tt, kk).data.align.mean;
                    sbAppPos = minMotSt(ii,jj,tt,kk).data.table.sAppear;
                    relMeanPos = minMotSt(ii,jj,tt,kk).data.align.meanPos;
                    sbAppInd = relMeanPos(relMeanPos(:,2) == sbAppPos, 1);
                    baseInds = arrayfun(@(x) find(relDatSB(:,1) > x, 1, 'first'), preStimBaseWin);
                    zeroInd = find(relDatMM(:,1) > 0, 1, 'first');
                    
                    baseVal = mean(relDatMM(baseInds(1):baseInds(2), 2));
                    baseSubResp = relDatMM;
                    baseSubResp(:,2) = baseSubResp(:,2) - baseVal;
                
                    minMotSt(ii, jj, tt, kk).subData.baseSub = baseSubResp;
                    minMotSt(ii, jj, tt, kk).subData.baseline = baseVal;
                    minMotSt(ii, jj, tt, kk).subData.zeroInd = zeroInd;
                    minMotSt(ii, jj, tt, kk).subData.sbInd = sbAppInd;
                    minMotSt(ii, jj, tt, kk).subData.length = size(baseSubResp, 1);
                    
                end
                
            end
            
        end
        
    end
    
end

% calculating linear differance

             
for tt=1:length(uTD)
    
    relTimeDiff = minMotSt(1,1,tt,1).data.table.fDisappear - minMotSt(1,1,tt,1).data.table.fAppear;
    
    for ii=1:length(usPos)
        
        for jj=1:length(ufPos)
            
            for kk=1:length(uFBS)
            
                if isempty(minMotSt(jj, ii, tt, kk).data)
                    
                    continue
                    
                else
                %                     [tt,ii,jj,kk]
                    % instead of estimating, this actually takes the mean
                    % position for the second bar appearance and find the
                    % appropriate time to copy the linSum response to
                    relSt = minMotSt(jj,ii,tt,kk);
                    totLen = relSt.subData.length;
                    relZeroInd = relSt.subData.zeroInd;
                    
                    sApInd = relSt.subData.sbInd;
                    
                    if uFBS(kk) == 0
                        shiftedFResp = padRespVec(relSt.subData.fb, relZeroInd, totLen);
                    elseif uFBS(kk) == 1
                        relTable = relSt.data.table;
                        relPos = relTable.fAppear:relTimeDiff:relTable.fDisappear-relTimeDiff;
                        
                        relInd = relSt.data.align.meanPos(ismember(relSt.data.align.meanPos(:,2), relPos), 1);
                        
                        totLen = relSt.subData.length;
                        shiftedVecs = zeros(totLen, length(relInd));
        
                        for pp=1:length(relInd)
                            shiftedVecs(:,pp) = padRespVec(relSt.subData.fb, relInd(pp), totLen);
                        end
                        
                        shiftedFResp = sum(shiftedVecs, 2);
                    end
                    
                    shiftedSResp = padRespVec(relSt.subData.sb, sApInd, totLen);
                    
                    minMotSt(jj, ii, tt, kk).linSumSB = shiftedFResp + shiftedSResp;
                    minMotSt(jj, ii, tt, kk).linDiff = relSt.subData.baseSub(:,2) - (shiftedFResp + shiftedSResp);
                    
                end
                
            end
            
        end
        
    end
    
end



if nargout > 1 
    varargout{1} = relSingBarSt;
end


                

end
       
       
       
       