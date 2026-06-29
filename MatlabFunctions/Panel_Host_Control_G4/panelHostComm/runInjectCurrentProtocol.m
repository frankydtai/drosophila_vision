function protocolStructAO = runInjectCurrentProtocol(inputStruct)

% function protocolStructAO = runInjectCurrentProtocol(inputStruct)

% This function injects current and records the response. Protocols can be 
% either step or ramp and will generate a default structure once user input is recieved.  
%
% 
% INPUT
% inputStruct -         Structure to specify injection type and parameters.
%                       Should have the following fields:
%   .type -             string. { 'Step' } or 'Ramp' 
%   .maxCurr -          maximum current to inject (in pA)
%   .reps -             number of repeats
%   .numSteps -         Number of steps between 0 and maxCurr. 
%                       For step only (will be disregarded in ramp) { 4 }
%   .stepDur -          Duration of step injection in ms. { 500 }
%                       Again irrelevant for ramp (calculated as a function of rate and max) 
%   .rate -             1XN vector. Rate for ramp increase in pA per second. For ramp only, disregarded
%                       in step. { 20 }
%   .isi -              in ms { 500 }
%   .hypPulse -         logical. whether to include an hyper-polarization
%                       pulse or not. { 0 }
%   .hypCurr -          (*) size of the current injection prior to the
%                       Step/Ramp in pA { -0.5 } <should be negative >
%   .hypDur -           (*) Duration of the hyper-polarization pulse { 250 }
%   .arenaVal -         luminance value for arena while injCurr { 0 }. 
%                       Function assumes gsLevel 3
%   .randomize -        logical. whether to randomize the order of stim {1}
% 
% * only used if hypPulse is 1
%
% OUTPUT
% stepData -            a NX2 matrix with voltage and current as both channels. 
%                       Sampling rate is 20K and N is determined by both maxCurr (due to the preset rate)
%                       and the number of repeats.
%
% Data is atomatically saved in the working directory. 

%% Defualt structures

default.type = 'step';
default.maxCurr = 'UI';
default.numSteps = 4;
default.stepDur = 500;
default.rate = [5,10,20]; %pA per sec
default.isi = 500;
default.reps = 3;
default.hypPulse = 0;
default.hypCurr = -1;
default.hypDur = 500;
default.arenaVal = 0;
default.randomize = 1;

fixed.preStim = 250; % 250ms before each stim
fixed.fudgeT = 0.1; % time in secs (since it is for the matlab puase function)
fixed.relCh = 1; % AO channel by panel host definition
fixed.sampRate = 1000;
fixed.conversionRatio = 40/1; % 40pA/V when feedback resistor is 5GOhm

chInBin = dec2bin(fixed.relCh, 4);

%% Defining core variables

% combining default and input structures
if nargin ==0
    default = modifyDefaultStruct(default);
else
    default = modifyDefaultStruct(default, inputStruct);
end



protocolStructAO = struct;

protocolStructAO.inputParams = default;

%% Generating the signal

maxCurr = default.maxCurr;
maxVol = maxCurr/fixed.conversionRatio; % conversion ratio from a velfunc to voltage
assert(maxVol >= -10 && maxVol <= 10, 'current cannot exceed -10 to 10 when converted to V')

protType = lower(default.type);
assert(ismember(protType, {'step', 'ramp'}), 'type should be either ramp or step')

interStimInt = default.isi;
assert(isvector(interStimInt) && length(interStimInt) == 1, 'isi should be a single number')
assert(interStimInt > 0, 'isi should be a positive number')

switch protType
    case 'step'
        stepLength = default.stepDur * (fixed.sampRate/1000); % since input duration is in ms
        numSteps = default.numSteps;
        assert(isvector(stepLength) && length(stepLength) == 1, 'stepLength should be a single number')
        assert(stepLength > 0, 'stepLength should be a positive number')
        
        stepVals = linspace(0, maxVol, numSteps+1);
        stepVals = stepVals(2:end); %gets rid of zero
        
        allStim = cell(1,numSteps); 
        indTag = 1;
        
        for ii=1:length(stepVals)
            tempStim = zeros(1, stepLength+interStimInt);
            tempStim(1:stepLength) = stepVals(ii);
            allStim{ii} = tempStim;
        end
        
    case 'ramp'
        
        injectRate = default.rate;
        assert(isvector(injectRate), 'rate should be a 1XN vector')
        assert(max(injectRate) <= 1000 && min(injectRate) > 0, 'Possible range for rate is 1-1000pA per sec')
        rampLength = maxVol./(injectRate/fixed.conversionRatio) * fixed.sampRate;
        
        allStim = cell(1,length(rampLength));
        indTag = 2;
        
        for ii=1:length(rampLength)
            tempStim = zeros(1, rampLength(ii)+interStimInt);
            tempStim(1:rampLength(ii)) = linspace(0, maxVol, rampLength(ii));
            allStim{ii} = tempStim;
        end
end

%% adding prePulse if necessary

prePulseMax = -5.1; % to avoid strong negative injections

preP = default.hypPulse;
assert(ismember(preP, [0,1]), 'hypPulse should be a logical argument')
if preP
    preCurr = default.hypCurr;
    preDur =  default.hypDur * (fixed.sampRate/1000); %again convert ms to samples 
    
    assert(length(preCurr) == 1, 'hypCurr should be a single number')
    assert(preCurr < 0 && preCurr > prePulseMax, ...
           'hypCurr should be a negative number between 0 and %d', prePulseMax)
    assert(length(preCurr) == 1, 'hypCurr should be a single number')
    assert(preDur > 0, 'hypDur should be a positive number')
    
    preVol = preCurr/fixed.conversionRatio;
    prePulse = ones(1, preDur) * preVol;
    
    allStim = cellfun(@(x) [prePulse, x], allStim, 'uniformoutput', 0);
end

%% Bulding the stimulus structure

preStimDur = fixed.preStim * (fixed.sampRate/1000);
preStim = zeros(1, preStimDur);

allStim = cellfun(@(x) [preStim, x], allStim, 'uniformoutput', 0);

reps = round(default.reps); % just in case user puts non-round number
assert(reps > 0, 'reps should be a positive number');

numUStim = length(allStim);
randStim = default.randomize;
assert(ismember(randStim, [0,1]), 'randomize should be logical')

counter= 0;
for ii=1:reps
    
    if randStim
        tempInds = randperm(numUStim);
    else
        tempInds = 1:numUStim;
    end
    
    for jj=1:numUStim
        counter = counter+1;
        protocolStructAO.stim(counter).aoVec = allStim{tempInds(jj)};
        protocolStructAO.stim(counter).relInds = [indTag, 0, 0, tempInds(jj)];
        protocolStructAO.stim(counter).length = length(allStim{tempInds(jj)});
    end
end

statVec = make_vSDAO_image(protocolStructAO);
assert(statVec == 0, 'Problem creating AO files')

allStim = length(protocolStructAO.stim);

%% Generating empty pattern and posFunc

bkgdVal = default.arenaVal;
assert(ismember(bkgdVal, 0:7), 'arenaVal should be an integer between 0-7') %assumes gsLevel 3
generateEmptyPatAndVec(bkgdVal) % gives a value of 0 to the arena 

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

% sets the pattern and position functions to the empty ones
Panel_tcp_com('set_pattern_id', 1)
Panel_tcp_com('set_posfunc_id', [1, 1])

Panel_tcp_com('set_active_analog_channel', chInBin)
Panel_tcp_com('reset_counter')

% Add waitbar to be able to abort protocol cleanly
wbh = waitbar(0,'1','Name','Presenting Protocol',...
            'CreateCancelBtn',...
            'setappdata(gcbf,''canceling'',1)');
setappdata(wbh,'canceling',0)

%% Running the experiment

for ii=1:allStim

    Panel_tcp_com('set_analog_output_function', [fixed.relCh-1, ii]) % since counting starts at zero
    waitbar(ii/allStim, wbh, sprintf('Presenting protocl %d of %d',ii, allStim))
    
    Panel_tcp_log('start')
    Panel_tcp_com('start')
    pause(protocolStructAO.stim(ii).length/fixed.sampRate + fixed.fudgeT)
    Panel_tcp_com('stop')
    fileName = Panel_tcp_log('stop');
    
    protocolStructAO.stim(ii).fileName = fileName;
    
    if getappdata(wbh,'canceling')
        break
    end
end

%% Cleaning up after the experiment is done

delete(wbh)
pause(1)



protocolStructAO.type = 'currentInj';

timeStamp = datestr(now, 'yyyymmdd_HH-MM');
folderName = fullfile(pwd, ['stepData', timeStamp]);
[stat, mess] = mkdir(folderName);
assert(stat==1, 'error creating folder: %s', mess)

fileNameCell = arrayfun(@(x) protocolStructAO.stim(x).fileName, 1:allStim, 'uniformoutput', 0)';
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


% reminder to avoid colleting from blind flies 
helpdlg('Use Flashlight before proceeding')

end
