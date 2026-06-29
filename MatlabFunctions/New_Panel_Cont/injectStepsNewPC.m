function protocolStructAO = injectStepsNewPC(maxCurr, numSteps, reps)

% function data = injectSteps(maxCurr, reps, injectRate)
%
% This function injects a 500ms current steps and records the responses in both
% current and voltage traces. The function assumes a 400pA/V conversion 
% ratio (defined in the amplifier gains)   
%
% INPUT
% maxCurr - maxCurr to be injected in pA
% numSteps - number fo steps between maxCurr and rest
% reps -    (optional), number of times to repeat the injection (defualt 5)
% 
%
% OUTPUT
% stepData -    a NX2 matrix with voltage and current as both channels. 
%           Sampling rate is 10K and N is determined by both maxCurr (due to the preset rate)
%           and the number of repeats. 
% Data is atomatically saved in the working directory. 



%% Defining core variables
sampRate = 1000;
interStepInt = sampRate/2; % 0.5 sec between steps
direc = pwd;
relCh = 1; % AO channel by panel host definition
chInBin = dec2bin(relCh, 4);

protocolStructAO = struct;

if nargin < 3
    reps = 5;
end

% getting last file in the log file directory
load logDir % saved in "C:\Users\gruntmane\Documents\ExpCodeandRes\MatlabFunctions\Panel_Host_Control"

oldFileSt = dir(fullfile(logDir, '*.tdms'));
[~, oldInd] = max([oldFileSt.datenum]);
if ~isempty(oldInd)
    newestOldFileName = oldFileSt(oldInd).name;
else
    newestOldFileName = [];
end

protocolStructAO.signalPar.maxCurr = maxCurr; 
protocolStructAO.signalPar.numSteps = numSteps;
protocolStructAO.signalPar.repeats = reps;

%% Generating the signal
conversionRatio = 40/1; % 40pA/V when feedback resistor is 5GOhm
maxVol = maxCurr/conversionRatio; % conversion ratio from a velfunc to voltage
assert(maxVol >= -10 && maxVol <= 10, 'current cannot exceed -10 to 10 when converted to V')

stepLength = sampRate/2; % 500ms
stepVals = linspace(0, maxVol, numSteps+1);
stepVals = stepVals(2:end); %gets rid of zero

valInds = repmat(1:numSteps, 1, reps);
permInds = randperm(length(valInds));
shufInds = valInds(permInds);

oneStep = [ones(1, stepLength), zeros(1, interStepInt)];

sigMat = zeros(length(oneStep), numSteps*reps);

for ii=1:length(shufInds)
    sigMat(:,ii) = oneStep*stepVals(shufInds(ii));
end


anaSig = vertcat(zeros(interStepInt, 1), reshape(sigMat, [], 1), zeros(interStepInt, 1));
stat = make_vSDAO_image_forVec(anaSig);
assert(stat == 0, 'Problem creating AO file')

protocolStructAO.signal = anaSig;


%% Generating empty pattern and posFunc

generateEmptyPatAndVec(0) % gives a value of 0 to the arena 

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
Panel_tcp_com('set_config_id', 1)
Panel_tcp_com('g_level_0')

% sets the pattern and position functions to the empty ones
Panel_tcp_com('set_pattern_id', 1)
Panel_tcp_com('set_posfunc_id', [1, 1])


%% Running the experiment

% Panel_tcp_com('set_mode', [4, 0]);
% Panel_tcp_com('send_gain_bias', [0 0 0 0]);
Panel_tcp_com('set_active_analog_channel', chInBin)
Panel_tcp_com('set_analog_output_function', [relCh-1, 1]) % since counting starts at zero
Panel_tcp_com('reset_counter')

Panel_tcp_log('start')
Panel_tcp_com('start')

pause(length(anaSig)/sampRate)
Panel_tcp_com('stop')
fileName = Panel_tcp_log('stop');

%% Cleaning up after the experiment is done

pause(1)

% getting all the new file names
% fileSt = dir(fullfile(logDir, '*.tdms'));
% if ~isempty(newestOldFileName) % gets rid of old files in the directory
%     oldFilesInd = find(arrayfun(@(x) strcmp(fileSt(x).name, newestOldFileName), 1:length(fileSt)));
%     fileSt = fileSt(oldFilesInd+1:end);
% end
% 
% % just one file should be generated for this experiment
% assert(length(fileSt) == 1, 'Generated TDMS files do not match number of stimuli presented')

protocolStructAO.stim(1).fileName = fileName;
protocolStructAO.type = 'currentInj';

timeStamp = datestr(now, 'yyyymmdd_HH-MM');
folderName = fullfile(pwd, ['stepData', timeStamp]);
[stat, mess] = mkdir(folderName);
assert(stat==1, 'error creating folder: %s', mess)

fileNameCell = arrayfun(@(x) protocolStructAO.stim(x).fileName, 1, 'uniformoutput', 0)';
copyLogFiletoCurrDir(fileNameCell, folderName) 
save(fullfile(folderName, ['protocolStructAO', timeStamp]), 'protocolStructAO')


protocolStructAO = consolidateData(folderName);

% cleaning up and closing AO channels
load panelContConfigFileDir % saved in "C:\Users\gruntmane\Documents\ExpCodeandRes\MatlabFunctions\Panel_Host_Control"

pConfig = fileread(panelContConfigFileDir);
pConfigFormatted = textscan(pConfig, '%s');
pathInd = find(cellfun(@(x) strcmp(x, 'Output]'), pConfigFormatted{1})) + 3; % add 3 since there is 'path', and '=' in between
temp_path = pConfigFormatted{1}{pathInd};

dos(['del /Q "' temp_path '\*.ao"']); %deleting ao files

Panel_tcp_com('set_active_analog_channel', '0000')

% % Plotting data
% if numSteps > 6
%     subpar = [2, ceil(numSteps/2)];
% else
%     subpar = [1, numSteps];
% end
% 
% figure;
% set(gcf, 'position', [180, 700, 1600, 300])
% axh = zeros(1, numSteps);
% for ii=1:numSteps
%     tempInds = find(anaSig == stepVals(ii));
%     spInds = SplitVec(tempInds, 'consecutive');
%     spInds = cellfun(@(x) [x', x(end) + (1:250)], spInds, 'uniformoutput', 0);
%     axh(ii) = subplot(subpar(1), subpar(2), ii);
%     hold on
%     cellfun(@(x) plot(1:length(x), stepData(x, 2), 'b'), spInds)
%     hold off
%     title(num2str(stepVals(ii)))
% end
% 
% yylim = get(axh(:), 'ylim');
% yylim = vertcat(yylim{:});
% yymax = max(yylim(:,2));
% yymin = min(yylim(:,1));
% 
% set(axh(:), 'ylim', [yymin, yymax])



end
