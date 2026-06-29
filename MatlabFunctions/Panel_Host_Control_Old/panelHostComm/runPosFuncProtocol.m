function protStruct  = runPosFuncProtocol(funcHand, pStruct)

% function data  = runPosFuncProtocol(funcHand, pStruct)
%
% This function uses the function handle and the given protocol structure
% to generate the required stimuli and dump them into the new panel controller (use tcp connection)
%
% INPUTS
% funcHand -    (optional) function handle to one of the createXXXProtocol functions
%               if not given 'listCreateProtocolFunctions' is envoked
%
% pStruct -     (optional) protocol structure that will be fed into the
%               function above. If not given user will be asked for the necessary inputs
%
% OUTPUT
% protStruct -         protocol structure with all the relevant fields from
%                   createProtocol, plus stim structure for each stimulus created which includes the following fields
%   .matCell -      arenaSize(1)XarenaSize(2)XN matrix that is the stimulus
%                   preseted in matrix format (createProtocol)
%   .relInds -      1X4 vector with indicies for stimSeq, mask, orientation, and
%                   maskPosition from which stimulus was created (createProtocol)
%   .freqCorr -     factor by which timer period was corrected to equalize
%                   temporal frequencies (for protocol that contain stimuli with different
%                   spatial frequencies) (createProtocol)
%   .dpsFreq -      degrees per second conversion (based on degrees per
%                   pixel for the arena)
%   .patVecMat -    vector format of matCell (created here)
%   .fileName -     TDMS file for this stim (created here)
%   .data -         cell array with first cell being channels recorded and second
%                   xPosition (timer updated based on rows in rows in patVecMat)
%

%% initiating parameters
fudgeT = 0.25; % adds to session time to make sure pattern presentation is done
degPerPix = 2.25; % since my arena is 180deg and 96pix (X axis)


% getting last file in the log file directory
load logDir % saved in "C:\Users\gruntmane\Documents\ExpCodeandRes\MatlabFunctions\Panel_Host_Control"
oldFileSt = dir(fullfile(logDir, '*.tdms'));
[~, oldInd] = max([oldFileSt.datenum]);
if ~isempty(oldInd)
    newestOldFileName = oldFileSt(oldInd).name;
else
    newestOldFileName = [];
end

if nargin < 1 
    funcHand = listCreateProtocolFunctions;
end

%% Establish panel host connection 
[~ , res] = system('tasklist /fi "imagename eq Panel Host.exe" /fo table /nh');

while ~strcmpi(res(2:6), 'Panel')
    warnh = warndlg('Panel Host is not open; Open and press OK', 'Panel Host Warning', 'modal');
    uiwait(warnh)
    [~ , res] = system('tasklist /fi "imagename eq Panel Host.exe" /fo table /nh');
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
    protocolStruct = feval(funcHand);
else
    protocolStruct = feval(funcHand, pStruct);
end

assert(isfield(protocolStruct, 'stim'), 'Protocol structure is missing stim field')
numStim = length(protocolStruct.stim);

% converts from degrees per second to Position function frequency (pixel per second);
assert(isfield(protocolStruct, 'generalFrequency'), 'Missing field: generalFrequency')
dpsFreq = protocolStruct.generalFrequency * degPerPix;
protocolStruct.dpsFreq = dpsFreq;
fprintf('\n posFunc Freq of %d is %.2f degPerSec \n', protocolStruct.generalFrequency, dpsFreq);

for ii=1:numStim
    tempPatVec = convertPatternMatrix(protocolStruct.stim(ii).matCell);
    protocolStruct.stim(ii).patVecMat = tempPatVec; 
end
fprintf('converted stimuli to serial vectors \n')


statPat = make_vSDpattern_image(protocolStruct);
statPos = make_vSDposfunction_image(protocolStruct);
assert(statPat == 0, 'Problem creating pattern files')
assert(statPos == 0, 'Problem creating function files')

% Add waitbar to be able to abort protocol cleanly
wbh = waitbar(0,'1','Name','Presenting Protocol',...
            'CreateCancelBtn',...
            'setappdata(gcbf,''canceling'',1)');
setappdata(wbh,'canceling',0)

% save protocolStruct before experiment starts
timeStamp = datestr(now, 'yyyymmdd_HH-MM');
funcStr = func2str(funcHand);
folderName = fullfile(pwd, [funcStr(7:end), timeStamp]); %gets rid of the word 'create'
[stat, mess] = mkdir(folderName);
assert(stat==1, 'error creating folder: %s', mess)
save(fullfile(folderName, ['protocolStruct', timeStamp]), 'protocolStruct', '-v7.3')

%% Starts the experiment
% Config 2 block a third of the arena to speed up performance
% Since T4s respond to middle of arena it should be avoided

Panel_tcp_com('set_config_id', 3) % config 3 is with cut corners (to avoid getting the Ack Error)  

Panel_tcp_com('set_mode', [4, 0]);
Panel_tcp_com('send_gain_bias', [0 0 0 0]);
Panel_tcp_com('reset_counter')

figH = figure('position', [1450, 50, 450, 150]);
figH.MenuBar = 'none';
maxValforFig = 2^(protocolStruct.inputParams.gsLevel)-1;

for ii=1:numStim

    Panel_tcp_log('start'); % to minimize non-recorded time loop iteration begin and end in log commands
    pause(0.1)
    relFreq = round(protocolStruct.generalFrequency/protocolStruct.stim(ii).freqCorr); % to avoid weird behavior
    
    Panel_tcp_com('set_pattern_id', ii)
    Panel_tcp_com('set_position', [1 1]);
    Panel_tcp_com('set_posfunc_id', [1,ii]);
    Panel_tcp_com('set_funcx_freq', relFreq);
    stimTime = size(protocolStruct.stim(ii).patVecMat, 2)/relFreq;
    %inStimInd = floor(size(protocolStruct.stim(ii).patVecMat, 2)*3/5);
    
    waitbar(ii/numStim, wbh, sprintf('Presenting protocl %d of %d',ii, numStim))
    plotMidFrame(mean(protocolStruct.stim(ii).matCell,3), maxValforFig)
    tH = title(num2str(protocolStruct.stim(ii).relInds));
    tH.VerticalAlignment = 'top';
    
    Panel_tcp_com('start')

    pause(stimTime + fudgeT)

    Panel_tcp_com('stop')
    pause(0.01)
    tempFileName = Panel_tcp_log('stop');
    
    protocolStruct.stim(ii).fileName = tempFileName;
    
    if getappdata(wbh,'canceling')
        break
    end
    
    
end

totStimNum = ii;

%% clean up after exp is done
delete(wbh)
Panel_tcp_com('set_config_id', 3)
Panel_tcp_com('g_level_7')

% % getting all the new file names
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
%     protocolStruct.stim(ii).fileName = fileSt(fileNameInd(ii)).name;
% end

% To get file names if function crashes while trying to move files
save(fullfile(folderName, ['protocolStruct', timeStamp]), 'protocolStruct', '-v7.3')

fileNameCell = arrayfun(@(x) protocolStruct.stim(x).fileName, 1:numStim, 'uniformoutput', 0)';
copyLogFiletoCurrDir(fileNameCell, folderName) 
save(fullfile(folderName, ['protocolStruct', timeStamp]), 'protocolStruct', '-v7.3')


protStruct = consolidateData(folderName);
close(figH)


end


