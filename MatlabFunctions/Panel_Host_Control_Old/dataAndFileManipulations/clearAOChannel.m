function clearAOChannel

% This function is to be used if AO channels need to be cleared (delete
% files and close active channels)
%
% assumes TCP connection is active

load panelContConfigFileDir % saved in "C:\Users\gruntmane\Documents\ExpCodeandRes\MatlabFunctions\Panel_Host_Control"

pConfig = fileread(panelContConfigFileDir);
pConfigFormatted = textscan(pConfig, '%s');
pathInd = find(cellfun(@(x) strcmp(x, 'Output]'), pConfigFormatted{1})) + 3; % add 3 since there is 'path', and '=' in between
temp_path = pConfigFormatted{1}{pathInd};
dos(['del /Q "' temp_path '\*.ao"']); % delete all the remaining AO files

Panel_tcp_com('set_active_analog_channel', '0000')

end

