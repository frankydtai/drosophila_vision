function [minMotSt, varargout] = calcMinMotDarkLinCompwSingleBar(pDarkMinMotSt, pSingleBarSt)

% function minMotSt = calcMinMotDarkLinComp(pStruct)
%
% This function is an elaboration of calcMinMotInhibLinComp. It doens't just use 
% the single bar data from the minMot protocol but also uses singleBar data
% to generate the linear comparision (by shifting response to the
% appropriate time)
%
%
% INPUT 
%
% pDarkMinMotSt -   minMot protocol with manual addition of bar
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
relTabMM = pDarkMinMotSt.gratingTable;
relTabSB = pSingleBarSt.gratingTable;

assert(all(ismember(['fAppear'; 'sAppear'], relTabMM.Properties.VariableNames)), ...
           'gratingTable is missing either fAppear or sAppear')

assert(ismember('appear', relTabSB.Properties.VariableNames), ...
           'gratingTable is missing either appear')
       

assert(ismember('FBStat', relTabMM.Properties.VariableNames), 'This function is designed for minMot protocols only')
simTD = pDarkMinMotSt.inputParams.stepDur; % stim time for bars presented sim

ufPos = unique(relTabMM.FBPos);
usPos = unique(relTabMM.SBPos);
uTD = unique(relTabMM.timeDiff);

uPos = unique(relTabSB.position);
uStD = unique(relTabSB.stimDur);

compUTD = uTD;
compUTD(compUTD == 0) = simTD;


assert(all(ismember(compUTD, uStD)), 'time delays in Singlebar are not identical to minMot') % didn't use equal since minMot also has 0 time delay
assert(all(ismember(usPos, uPos)), 'some minMot sbPos not present in singleBar protocol') 

alignMMSt = alignProtocolDataByTable(pDarkMinMotSt, 'sAppear'); %aligns to appearance of first bar (to take common baseline)
alignSBSt = alignProtocolDataByTable(pSingleBarSt, 'appear'); 

% orginizing mean data


%organizing single bar data just for the relevant positions =

singleBarSt = struct;


for ii=1:length(usPos)
    for tt=1:length(uTD) 
        
        relSBInd = ismember(relTabSB{:, {'position'; 'stimDur'}}, [usPos(ii), compUTD(tt)], 'rows');
        singleBarSt(ii,tt).data = alignSBSt(relSBInd);
        relDatSB = singleBarSt(ii,tt).data.align.mean;
        
        baseInds = arrayfun(@(x) find(relDatSB(:,1) > x, 1, 'first'), preStimBaseWin);
        zeroInd = find(relDatSB(:,1) > 0, 1, 'first');        
        baseVals = relDatSB(baseInds(1):baseInds(2), 2);
        baseSubResp = relDatSB;
        baseSubResp(:,2) = baseSubResp(:,2) - mean(baseVals);
        
        singleBarSt(ii, tt).subData.baseSub = baseSubResp;
        singleBarSt(ii, tt).subData.zeroInd = zeroInd;
        singleBarSt(ii, tt).subData.length = size(baseSubResp, 1);
    end
end


minMotSt = struct;

for ii=1:length(ufPos)
    
    for jj=1:length(usPos)
        
        for tt=1:length(uTD)
                        
            relMMInd = ismember(relTabMM{:, {'FBPos'; 'SBPos'; 'timeDiff'}}, ...
                                 [ufPos(ii), usPos(jj), uTD(tt)], 'rows');
            
            if unique(relMMInd) == 0
                continue
            else
                     
            minMotSt(ii, jj, tt).data = alignMMSt(relMMInd); 
            minMotSt(ii, jj, tt).subData.sb = singleBarSt(jj, tt).subData;
                    
            relDatMM = minMotSt(ii, jj, tt).data.align.mean;
            baseInds = arrayfun(@(x) find(relDatSB(:,1) > x, 1, 'first'), preStimBaseWin);
            zeroInd = find(relDatMM(:,1) > 0, 1, 'first');
                    
            baseVal = mean(relDatMM(baseInds(1):baseInds(2), 2));
            baseSubResp = relDatMM;
            baseSubResp(:,2) = baseSubResp(:,2) - baseVal;
                
            minMotSt(ii, jj, tt).subData.baseSub = baseSubResp;
            minMotSt(ii, jj, tt).subData.baseline = baseVal;
            minMotSt(ii, jj, tt).subData.zeroInd = zeroInd;
            minMotSt(ii, jj, tt).subData.length = size(baseSubResp, 1);
                    
            
                
            end
            
        end
        
    end
    
end

% calculating linear differance

             
for tt=1:length(uTD)
    
    for ii=1:length(usPos)
        
        for jj=1:length(ufPos)
            
           if isempty(minMotSt(jj, ii, tt).data)
                continue    
            else
                
            % instead of estimating, this actually takes the mean
            % position for the second bar appearance and find the
            % appropriate time to copy the linSum response to
            relSt = minMotSt(jj,ii,tt);
            totLen = relSt.subData.length;
            
            sApInd = relSt.subData.zeroInd;
%           [tt,ii,jj,kk]
            shiftedSResp = padRespVec(relSt.subData.sb, sApInd, totLen);
            
            minMotSt(jj, ii, tt).shiftedSB = shiftedSResp;
            minMotSt(jj, ii, tt).linDiff = relSt.subData.baseSub(:,2) - shiftedSResp;
                    
            
                
            end
            
        end
        
    end
    
end



if nargout > 1 
    varargout{1} = singleBarSt;
end


                

end
       
       
       
       