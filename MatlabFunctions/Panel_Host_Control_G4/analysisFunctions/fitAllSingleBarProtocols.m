function allSingleBarFit = fitAllSingleBarProtocols

% function fitAllSingleBarProtocols
%
% Thsi function runs fitFuncToSingleBarSt on all applicable singleBar
% protocol from first date to last. Should be run from the directory that
% contains all the relevant recording directories.
% 
% Note!
% p20160504_17_24 is not the optimal orientation for that cell (has better
% singlebar data in p20160504_17_48
% p20160527_12_24 is from the saem cell as p20160527_11_37 and was added to
% make sure center didn't move (there was something wierd with that cell)



firstDir = '20160412';
lastDir = '20160527';

protName = 'SingleBarDiagCorr';
stringDeduct = length('protocolStruct');

currDir = dir;

allNames = {currDir.name};

startInd = find(cellfun(@(x) strcmp(x, firstDir), allNames));
stopInd = find(cellfun(@(x) strcmp(x, lastDir), allNames));

for ii=startInd:stopInd
    
    intDir = dir(currDir(ii).name);
    intNames = {intDir.name};
    
    protInd = find(cellfun(@(x) ~isempty(strfind(x, protName)), intNames));
    
    for jj=1:length(protInd);
        
        protStructName = dir(fullfile(pwd, currDir(ii).name, intNames{protInd(jj)}, 'protocolStruct*'));
        
        load(fullfile(pwd, currDir(ii).name, intNames{protInd(jj)}, protStructName.name))
        
        % if it is a dark bar
        if protocolStruct.inputParams.stimBar == 0 
            continue
        end
        % or I didn't add appear to the table
        if ~ismember('appear', protocolStruct.gratingTable.Properties.VariableNames)
            continue
        end
        
        % or I aborted the protocol 
        if isempty(protocolStruct.stim(end).data)
            continue
        end
        
        tempName = protStructName.name(stringDeduct+1:end);
        validName = matlab.lang.makeValidName(tempName(1:end-4));
        validName = ['p', validName(2:end)]; % replacing x with p
        
        fprintf('fitting cell %d from date %s \n', jj, validName)
        tempFit = fitFuncToSingleBarSt(protocolStruct);
        
        allSingleBarFit.(validName) = tempFit;
        
        plotBaseSubSingleBarPlusFit(tempFit)
        set(gcf, 'name', validName, 'NumberTitle', 'off', ...
            'units', 'pixels', 'position', [1, 5, 2560, 1340])
        pause(1)
        print2PDF(['./T4recordingSummaryAndAnalysis/singleBarProtocols/singleBarFit', validName])
        close gcf
    end
    
end
        
        
    
    






end