function protocolStructAO  = runPosFuncProtocolwAO(inpStructAO, funcHand, pStruct)

% function data  = runPosFuncProtocolwAO(pStructAOComb, funcHand, pStruct)
%
% This function uses the function handle and the given protocol structure
% to generate the required stimuli and run them into the new panel controller (use tcp connection)
% It uses pStructAOComb to combine the pStruct generated stimuli with AO
% functions. 
%
% INPUTS
% 
% funcHand -    function handle to one of the createXXXProtocol functions
% pStruct -     corresponding protocol structure 
% 
% NOTE! If not given will prompt user for input. 
%
% pStructAO -               protocol structure that will be fed into the function above. 
%                           Doesn't have to have all the required fields, but the .aoVec has to be included 
%   .aoVec -                cell array with AOvectors (-10 to 10 will be read @ 1KHz)
%   .stimVecComb -          
%                           matrix describing the desired stim AOvec combinations.
%                           If not given all stim and all vec will be presented
%                           individually, and with all the combinations. Should be given based on uniStim indexing. 
%                           NX2 matrix with first column indexing uniStim and
%                           second column indexing AOvec. 0 signifies no
%                           presentation (of pattern if in first column and LED
%                           pulse if in the second)
%
% OUTPUT
% protocolStructAO -        protocol structure with all the relevant fields from
%                           createProtocol, plus stim structure for each stimulus created which includes the following fields
%   .inputStruct -          protocol structure that was used as the input after
%                           it had gone throught the createProtocol function (has .stim field)
%   .uniStim -              structure. unique stimuli from the original
%                           protocol structure (since it is generated with the repeats - stupid,
%                           but I am too lazy to change)
%   .stimVecComb -          Taken form the input or generated as all the
%                           options. Indicies for the stim are from uniStim. 
%   .aoVec -                cell array from input
%   .combGratingTable -     new table combinging gratingTable w/ aoVec
%                           inds, changing indices to the correct stimVecComb indices and adding
%                           stimVec comb into the matrix. Plus adding a column for aoFlag which is
%                           1 for just pattern, 2 for just aoVec and 3 for both. 
%   .randomStimSeq -        1XN vector. The sequence by which stimuli were
%                           presented (indicies from stimVecComb)
%   .stim -                 structure of all the presented stimuli. Contains
%                           the following fields:
%       .matCell
%       .relInds
%       .patVecMat
%       .fileName
%       .length             All the same as in runPosFuncProtocol
%       .aoVec -            Vector that was associated with particular stim
%       .aoVecLen -         Effective length of vector (length w/o the final stretch of zeros).
%                       
%       Note! the function adds empty patterns and empty aoVec for the
%       presentation of just pattern/aoVec
%   

%% initiating parameters
fudgeT = 0.25; % adds to session time to make sure pattern presentation is done
degPerPix = 2.25; % since my arena is 180deg and 96pix (X axis)
relCh = 2; % AO channel by panel host definition
protocolStructAO = struct;


% getting last file in the log file directory
load logDir % saved in "C:\Users\gruntmane\Documents\ExpCodeandRes\MatlabFunctions\Panel_Host_Control"

oldFileSt = dir(fullfile(logDir, '*.tdms'));
[~, oldInd] = max([oldFileSt.datenum]);
if ~isempty(oldInd)
    newestOldFileName = oldFileSt(oldInd).name;
else
    newestOldFileName = [];
end

%% Establish panel host connection 
[~ , res] = system('tasklist /fi "imagename eq Panel Host.exe" /fo table /nh');

if ~strcmpi(res(2:6), 'Panel')
    
    openPH = questdlg('Press OK to open Panel Host', ...
                        'BIAS', 'OK', 'Cancel', 'OK');
    switch openPH
        case 'OK'
            !E:\Panel Host\Panel Host.exe &
        case 'Cancel'
            error('runPosFuncProtocolwAO aborted - Panel Host required required')      
    end
    
end

establishtcp = questdlg('Would you like to establish TCP connection?', ...
                        'TCP Comm', 'Yes', 'No', 'No');
switch establishtcp
    case 'Yes'
        init_tcp;
        fprintf('TCP initiated \n')
    case 'No'
        fprintf('Verify TCP is initiated \n')
end


% Sets arena to mid GS level and switches half off (with the second config
% file
Panel_tcp_com('set_config_id', 3)
Panel_tcp_com('g_level_7')

%% run the desired function to generate the 32X96XN matrix to be presented

if nargin < 2 
    funcHand = listCreateProtocolFunctions;
end

if nargin < 3
    protocolStruct = feval(funcHand);
else
    protocolStruct = feval(funcHand, pStruct);
end

assert(isfield(protocolStruct, 'stim'), 'Protocol structure is missing stim field')

if isfield(protocolStruct.inputParams, 'freqCorrFlag') && protocolStruct.inputParams.freqCorrFlag
    warning('This function does not preform frequency correction')
    beep
end

if isfield(protocolStruct, 'freqCorrFlag') && protocolStruct.freqCorrFlag
    warning('This function does not preform frequency correction')
    beep
end

assert(isfield(inpStructAO, 'aoVec'), 'pStruct is missing aoVec field')
numAOVec = length(inpStructAO.aoVec);

aoEffLen = cellfun(@findAOVecEffLen, inpStructAO.aoVec);
aoEffLen(end+1) = 0; % since later on an empty AO vector will be added last (for presentation of just pattern) 

% generates unique stimuli field (a bit backwards, but that how
% createProtocol works)

tempInds = vertcat(protocolStruct.stim.relInds);
[uniInds, stimInd, ~] = unique(tempInds, 'rows');

for ii=1:length(stimInd)
        protocolStructAO.uniStim(ii).matCell = protocolStruct.stim(stimInd(ii)).matCell;
        protocolStructAO.uniStim(ii).relInds = uniInds(ii,:);
        protocolStructAO.uniStim(ii).length = size(protocolStruct.stim(stimInd(ii)).matCell, 3); 
end

numUStim = length(protocolStructAO.uniStim);

% generate all individual presentations and all combinations
if ~isfield(inpStructAO, 'stimVecComb') || isempty(inpStructAO.stimVecComb)
    aoCol = reshape(repmat(1:numAOVec, 1, numUStim), [], 1);
    stCol = reshape(repmat(1:numUStim, numAOVec, 1), [], 1);
    justStim = [(1:numUStim)', zeros(numUStim, 1)];
    justAO = [zeros(numAOVec, 1), (1:numAOVec)'];
    inpStructAO.stimVecComb = vertcat(justStim, justAO, [stCol, aoCol]);
else
    assert(min(inpStructAO.stimVecComb(:,1)) >=0 && max(inpStructAO.stimVecComb(:,1)) <= numUStim, ...
           'stim values for stimVecComb are out of range')
    assert(min(inpStructAO.stimVecComb(:,2)) >=0 && max(inpStructAO.stimVecComb(:,2)) <= numAOVec, ...
           'aoVec values for stimVecComb are out of range')
end

% converts from degrees per second to Position function frequency (pixel per second);
assert(isfield(protocolStruct, 'generalFrequency'), 'Missing field: generalFrequency')
dpsFreq = protocolStruct.generalFrequency * degPerPix;
protocolStruct.dpsFreq = dpsFreq;
fprintf('\n posFunc Freq of %d is %.2f degPerSec \n', protocolStruct.generalFrequency, dpsFreq);

% adding empty aoVec and empty pattern to be used when pattern/aoVec are
% presented alone
protocolStructAO.aoVec = inpStructAO.aoVec;
protocolStructAO.stimVecComb = inpStructAO.stimVecComb;
protocolStructAO.aoVec{numAOVec+1} = zeros(1, 1000); % one sec of nothing
protocolStructAO.uniStim(numUStim+1) = protocolStructAO.uniStim(1);
bkgdNum = protocolStruct.inputParams.gsLevel;
protocolStructAO.uniStim(numUStim+1).matCell(:) = bkgdNum; % bkgdlevel matrix
protocolStructAO.uniStim(numUStim+1).relInds = [0, 0, 0, 0];

tempComb = inpStructAO.stimVecComb; % adding the right indices for the empty stim
for ii=1:2
    tempComb(tempComb(:,ii) == 0, ii) = max(tempComb(:,ii))+1;
end

for ii=1:numUStim
    tempPatVec = convertPatternMatrix(protocolStructAO.uniStim(ii).matCell);
    protocolStructAO.uniStim(ii).patVecMat = tempPatVec; 
end
fprintf('converted stimuli to serial vectors \n')

numComb = size(protocolStructAO.stimVecComb, 1);
randSeq = zeros(1, protocolStruct.repeats * numComb);

for ii=1:protocolStruct.repeats
    tInds = numComb * (ii-1) + (1:numComb);
    randSeq(tInds) = randperm(numComb);
end

protocolStructAO.randomStimSeq = randSeq;
protocolStructAO.stim = protocolStructAO.uniStim(tempComb(randSeq, 1));
for ii=1:length(randSeq)
    protocolStructAO.stim(ii).aoVec = protocolStructAO.aoVec{tempComb(randSeq(ii), 2)};
    protocolStructAO.stim(ii).aoVecInd = tempComb(randSeq(ii), 2); 
    protocolStructAO.stim(ii).aoVecLen = aoEffLen(tempComb(randSeq(ii), 2));
    protocolStructAO.stim(ii).combInds = protocolStructAO.stimVecComb(randSeq(ii),:);
    
    % to conform to the regular relInds convention in runPosFunProtocol (should allow easier plotting)
    protocolStructAO.stim(ii).combRelInds = [randSeq(ii), randSeq(ii), 1, 1]; 
end

numStim = length(protocolStructAO.stim);

protocolStructAO.inputStruct = protocolStruct;
protocolStructAO.gsLevel = protocolStruct.inputParams.gsLevel; % to be used by later functions (so they wont have to dig deep)

% adds combGratingTable to the stucture
tempTab = protocolStructAO.inputStruct.gratingTable;
stInd = protocolStructAO.stimVecComb;

emptyR = num2cell(zeros(1,width(tempTab)));
newT = [tempTab; emptyR];

allTable = [];
for ii=1:size(stInd,1)
    sigRowInd = newT.index == stInd(ii,1);
    assert(sum(sigRowInd) == 1, 'more than one row matching')
    sigRow = newT(sigRowInd, :);
    sigRow.index = ii;
    origInd = stInd(ii,1);
    aoVecInd = stInd(ii,2);
    aoFlag = sign(stInd(ii,1)) + 2*sign(stInd(ii,2));
    allTable = [allTable; [sigRow, table(origInd, aoVecInd, aoFlag)]];
end

protocolStructAO.combGratingTable = allTable; 


% making panel controller files
statPat = make_vSDpattern_image(protocolStructAO);
statPos = make_vSDposfunction_image(protocolStructAO);
statVec = make_vSDAO_imagewInd(protocolStructAO);
assert(statPat == 0, 'Problem creating pattern files')
assert(statPos == 0, 'Problem creating function files')
assert(statVec == 0, 'Problem creating AO files')

message = sprintf('Please check that red LED is set to ON and TTL');
uiwait(msgbox(message));


% Add waitbar to be able to abort protocol cleanly
wbh = waitbar(0,'1','Name','Presenting Protocol',...
            'CreateCancelBtn',...
            'setappdata(gcbf,''canceling'',1)');
setappdata(wbh,'canceling',0)

% save protocolStruct before experiment starts
timeStamp = datestr(now, 'yyyymmdd_HH-MM');
funcStr = func2str(funcHand);
folderName = fullfile(pwd, [funcStr(7:end), 'AO', timeStamp]); %gets rid of the word 'create'
[stat, mess] = mkdir(folderName);
assert(stat==1, 'error creating folder: %s', mess)
save(fullfile(folderName, ['protocolStructAO', timeStamp]), 'protocolStructAO')

%% Starts the experiment
% Config 2 block a third of the arena to speed up performance
% Since T4s respond to middle of arena it should be avoided


relFreq = round(protocolStruct.generalFrequency); % This function does not perform freq correction
chInBin = dec2bin(relCh, 4); % true in this context, but not always (since relCh 3 would mean activating channels 1 and 2 - which is wrong)

Panel_tcp_com('set_config_id', 3) % config 3 is with cut corners (to avoid getting the Ack Error)  

Panel_tcp_com('set_mode', [4, 0]);
Panel_tcp_com('send_gain_bias', [0 0 0 0]);
Panel_tcp_com('set_funcx_freq', relFreq);
Panel_tcp_com('set_active_analog_channel', chInBin)
Panel_tcp_com('reset_counter')
figH = figure('position', [1450, 50, 450, 150]);
figH.MenuBar = 'none';
maxValforFig = 2^(protocolStruct.inputParams.gsLevel)-1;



for ii=1:numStim
    
    Panel_tcp_log('start') % to minimize non-recorded time loop iteration begin and end in log commands
    
    
    patTime = size(protocolStructAO.stim(ii).patVecMat, 2)/relFreq;
    aoVTime = protocolStructAO.stim(ii).aoVecLen/1000; % since AO channels are always @ 1KHz
    tempVecInd = protocolStructAO.stim(ii).aoVecInd; 
    
    stimTime = max(patTime, aoVTime);
    
    Panel_tcp_com('set_pattern_id', ii)
    Panel_tcp_com('set_position', [1 1]);
    Panel_tcp_com('set_posfunc_id', [1,ii]);
    Panel_tcp_com('set_analog_output_function', [relCh-1, tempVecInd])
    
    waitbar(ii/numStim, wbh, sprintf('Presenting protocl %d of %d',ii, numStim))
    plotMidFrame(mean(protocolStructAO.stim(ii).matCell,3), maxValforFig)
    tH = title(num2str(protocolStructAO.stim(ii).combInds));
    tH.VerticalAlignment = 'top';
    
    Panel_tcp_com('start')
    
    pause(stimTime + fudgeT)
    
    Panel_tcp_com('stop')
    pause(0.01)
    tempFileName = Panel_tcp_log('stop');
    
    protocolStructAO.stim(ii).fileName = tempFileName;
  
    
    if getappdata(wbh,'canceling')
        break
    end
    
end

totStimNum = ii;

%% clean up after exp is done
delete(wbh)
Panel_tcp_com('set_config_id', 3)
Panel_tcp_com('g_level_7')
% closing the active channels
Panel_tcp_com('set_active_analog_channel', '0000')

% getting all the new file names
% fileSt = dir(fullfile(logDir, '*.tdms'));
% if ~isempty(newestOldFileName) % gets rid of old files in the directory
%     oldFilesInd = find(arrayfun(@(x) strcmp(fileSt(x).name, newestOldFileName), 1:length(fileSt)));
%     fileSt = fileSt(oldFilesInd+1:end);
% end
% 
% assert(length(fileSt) == totStimNum, 'Generated TDMS files do not match number of stimuli presented')
% 
% [~, fileNameInd] = sort([fileSt.datenum], 'ascend');
% 
% 
% for ii=1:totStimNum
%     protocolStructAO.stim(ii).fileName = fileSt(fileNameInd(ii)).name;
% end



fileNameCell = arrayfun(@(x) protocolStructAO.stim(x).fileName, 1:numStim, 'uniformoutput', 0)';
copyLogFiletoCurrDir(fileNameCell, folderName) 
save(fullfile(folderName, ['protocolStructAO', timeStamp]), 'protocolStructAO')


protocolStructAO = consolidateData(folderName);
close(figH)

% finding the directory in which to place file
load panelContConfigFileDir % saved in "C:\Users\gruntmane\Documents\ExpCodeandRes\MatlabFunctions\Panel_Host_Control"

pConfig = fileread(panelContConfigFileDir);
pConfigFormatted = textscan(pConfig, '%s');
pathInd = find(cellfun(@(x) strcmp(x, 'Output]'), pConfigFormatted{1})) + 3; % add 3 since there is 'path', and '=' in between
temp_path = pConfigFormatted{1}{pathInd};
dos(['del /Q "' temp_path '\*.ao"']); % delete all the remaining AO files



end


