function protocolStruct = continuousAIRecord(uniVal)

% This function uses the new panel controller to record continuously until
% prompted to stop by user
% output is the relevant data from AI channels of TDMS files
%
% INPUT
% uniVal - (optional) uniform value of arena background. will be fed into generateEmptyPatAndVec
%          default is 0.


%% Establish panel host connection 

if nargin < 1
    uniVal =0;
end

generateEmptyPatAndVec(uniVal)


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
Panel_tcp_com('set_config_id', 1)
Panel_tcp_com('g_level_0')

% sets the pattern and position functions to the empty ones
Panel_tcp_com('set_pattern_id', 1)
Panel_tcp_com('set_posfunc_id', [1, 1])
Panel_tcp_com('set_active_analog_channel', '0000')

%% Running the experiment

Panel_tcp_log('start')
Panel_tcp_com('start')

diaH = dialog('Position',[300 300 250 150],'Name','Experiment Running');

uicontrol('Parent',diaH,...
          'Style','text',...
          'Position',[20 80 210 40],...
          'String','Click STOP when done.');

uicontrol('Parent',diaH,...
          'Position',[85 20 70 25],...
          'String','STOP',...
          'Callback','delete(gcf)');

uiwait(diaH)
      
Panel_tcp_com('stop')
fileName = Panel_tcp_log('stop');

%% Copying file to current directory and moving data into a different format

protocolStruct.stim(1).fileName = fileName;
protocolStruct.type = 'contRecord';

timeStamp = datestr(now, 'yyyymmdd_HH-MM');
folderName = fullfile(pwd, ['contRecording', timeStamp]);
[stat, mess] = mkdir(folderName);
assert(stat==1, 'error creating folder: %s', mess)

fileNameCell = arrayfun(@(x) protocolStruct.stim(x).fileName, 1, 'uniformoutput', 0)';
copyLogFiletoCurrDir(fileNameCell, folderName) 
save(fullfile(folderName, ['protocolStruct', timeStamp]), 'protocolStruct')


protocolStruct = consolidateData(folderName);

% cleaning up and closing AO channels
load panelContConfigFileDir % saved in "C:\Users\gruntmane\Documents\ExpCodeandRes\MatlabFunctions\Panel_Host_Control"

pConfig = fileread(panelContConfigFileDir);
pConfigFormatted = textscan(pConfig, '%s');
pathInd = find(cellfun(@(x) strcmp(x, '[Function]'), pConfigFormatted{1})) + 3; % add 3 since there is 'path', and '=' in between
temp_path = pConfigFormatted{1}{pathInd};
dos(['del /Q "' temp_path '\*.fun"']); %SS


pathInd = find(cellfun(@(x) strcmp(x, '[Pattern]'), pConfigFormatted{1})) + 3; % add 3 since there is 'path', and '=' in between
temp_path = pConfigFormatted{1}{pathInd};
dos(['del /Q "' temp_path '\*.pat"']); % SS


end
