function protStruct  = runDumpProtocol(funcHand, pStruct)

% function data  = runDumpProtocol(funcHand, pStruct)
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
%   .patVecMat -    vector format of matCell (created here)
%   .fileName -     TDMS file for this stim (created here)
%   .data -         cell array with first cell being channels recorded and second
%                   xPosition (timer updated based on rows in rows in patVecMat)
%
% Note! Supp functions below

% initiating parameters and tcp connection

% getting last file in the log file directory
logDir = 'F:\Panel Host\Support Files\Log Files';
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


% Establish panel host connection 
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


% run the desired function to generate the 32X96XN matrix to be presented
if nargin < 2
    protocolStruct = feval(funcHand);
else
    protocolStruct = feval(funcHand, pStruct);
end

assert(isfield(protocolStruct, 'stim'), 'Protocol structure is missing stim field')
numStim = length(protocolStruct.stim);


for ii=1:numStim
    tempPatVec = convertPatternMatrix(protocolStruct.stim(ii).matCell);
    protocolStruct.stim(ii).patVecMat = tempPatVec; 
end
fprintf('converted stimuli to serial vectors \n')

assert(isfield(protocolStruct, 'generalFrequency'), 'Missing field: generalFrequency')
freq = 1/protocolStruct.generalFrequency; % convert from Hz to period

% Used to count the total number of frames
timerCounter = 0; 


% Add waitbar to be able to abort protocol cleanly
wbh = waitbar(0,'1','Name','Presenting Protocol',...
            'CreateCancelBtn',...
            'setappdata(gcbf,''canceling'',1)');
setappdata(wbh,'canceling',0)


% Sets arena to mid GS level and switches half off (with the second config
% file
Panel_tcp_com('set_config_id', 1)
Panel_tcp_com('g_level_7')
Panel_tcp_com('pc_dumping_mode')
Panel_tcp_com('set_config_id', 2)



for ii=1:numStim
    
    Panel_tcp_com('start_log') % to minimize non-recorded time loop iteration begin and end in log commands
    
    %% Equalize the temporal frequencies
    relPatVec = protocolStruct.stim(ii).patVecMat;
    
    if protocolStruct.freqCorrFlag 
        tempCorr = protocolStruct.stim(ii).freqCorr;
    else
        tempCorr = 1;
    end
    
    waitbar(ii/numStim, wbh, sprintf('Presenting protocl %d of %d',ii, numStim))
        
    relPer = round(1000*freq*tempCorr)/1000; % since timer objects are limited to ms precision
    tim = timer('ExecutionMode', 'fixedDelay', 'BusyMode', 'error', 'period', relPer, ...
                'TasksToExecute', size(relPatVec, 2));
            
    tim.TimerFcn = {@dumpFrameCallB,relPatVec, 1152}; 
    tim.ErrorFcn = @dumpErrorCB;
    tim.UserData = timerCounter;
    
    
    start(tim)
    wait(tim)
    
    delete(tim)
    clear('tim')
    
    Panel_tcp_com('stop_log')
  
    
    if getappdata(wbh,'canceling')
        break
    end
    
    
end

totStimNum = ii;

delete(wbh)
Panel_tcp_com('set_config_id', 1)
Panel_tcp_com('g_level_7')

% getting all the new file names
fileSt = dir(fullfile(logDir, '*.tdms'));
if ~isempty(newestOldFileName) % gets rid of old files in the directory
    oldFilesInd = find(arrayfun(@(x) strcmp(fileSt(x).name, newestOldFileName), 1:length(fileSt)));
    fileSt = fileSt(oldFilesInd+1:end);
end

assert(length(fileSt) == totStimNum, 'Generated TDMS files do not match number of stimuli presented')

[~, fileNameInd] = sort([fileSt.datenum], 'ascend');


for ii=1:totStimNum
    protocolStruct.stim(ii).fileName = fileSt(fileNameInd(ii)).name;
end




timeStamp = datestr(now, 'yyyymmdd_HH-MM');
funcStr = func2str(funcHand);
folderName = fullfile(pwd, [funcStr(7:end), timeStamp]); %gets rid of the word 'create'

fileNameCell = arrayfun(@(x) protocolStruct.stim(x).fileName, 1:numStim, 'uniformoutput', 0)';
copyLogFiletoCurrDir(fileNameCell, folderName) 
save(fullfile(folderName, ['protocolStruct', timeStamp]), 'protocolStruct')


protStruct = consolidateData(folderName);

end




%% Supp Functions

function dumpFrameCallB(obj, event, patVec, patVecLen)
% patVec -          vector to be presented
% patVecLen -       length of vector to be presented

ind = obj.TasksExecuted;

temp = obj.UserData;
temp = temp+1;
obj.UserData = temp;

Panel_tcp_com('dump_frame', [patVecLen, temp, 0, 48, 3, 0, patVec(:, ind)']);

end




