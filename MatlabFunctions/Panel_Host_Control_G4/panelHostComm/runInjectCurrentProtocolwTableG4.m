function protocolStructAO = runInjectCurrentProtocolwTableG4(inputStruct)

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
%
%   NOTE! if hypCurr and hypDur aere both vectors they should be the same length since one index will be used for both
%
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
default.hypPulse = 1;
default.hypCurr = -20;
default.hypDur = [0, 250];
default.arenaVal = 7;
default.randomize = 1;

fixed.preStim = 250; % 250ms before each stim
fixed.fudgeT = 0.1; % time in secs (since it is for the matlab puase function)
fixed.relCh = 0; % AO channel by panel host definition
fixed.sampRate = 1000;
% fixed.conversionRatio = 40/1; % 40pA/V when feedback resistor is 5GOhm
fixed.conversionRatio = 400/1; % 400pA/V when feedback resistor is 500MOhm
fixed.arenaSize = [48, 192];
fixed.gsLevel = 4;

chInBin = dec2bin(bitset(0,fixed.relCh+1,1),4);

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

        stimTab = table(repmat({protType}, length(stepVals), 1), stepVals', 'VariableNames', {'ProtType'; 'stepVal'});

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

        stimTab = table(repmat({protType}, length(rampLength), 1), rampLength', 'VariableNames', {'ProtType'; 'rampDur'});
end

%% adding prePulse if necessary

prePulseMax = -100.1; % to avoid strong negative injections (adjusted to big cells)
gratingTable = [];
preP = default.hypPulse;
assert(ismember(preP, [0,1]), 'hypPulse should be a logical argument')
if preP
    preCurr = default.hypCurr;
    preDur =  default.hypDur * (fixed.sampRate/1000); %again convert ms to samples

    assert(isvector(preCurr), 'hypCurr should a 1XN vector')
    assert(max(preCurr) < 0 && min(preCurr) > prePulseMax, ...
           'hypCurr should be a negative number between 0 and %d', prePulseMax)
    assert(isvector(preDur), 'hypDur should be a 1XN vector')
    assert(all(preDur >= 0), 'hypDur should be a positive number')

    preCurrLen = length(preCurr);
    preDurLen = length(preDur);
    relL =1;
    if preCurrLen == 1 && preDurLen > 1
        preCurr = ones(1, preDurLen) * preCurr;
        relL = preDurLen;
    elseif preCurrLen > 1 && preDurLen == 1
        preDur = ones(1, preCurrLen) * preDur;
        relL = preCurrLen;
    elseif preCurrLen > 1 && preDurLen > 1
        assert(preCurrLen == preDurLen, 'if both hypCurr and hypDur are vectors they should have the same length')
        relL = preDurLen;
    end

    allStim2 = cell(1, length(allStim) * preCurrLen);
    index=0;

    for hh=1:relL

        hypC = preCurr(hh);
        hypD = preDur(hh);

        preVol = hypC/fixed.conversionRatio;
        prePulse = ones(1, hypD) * preVol;

        for st=1:length(allStim)
            index=index+1;
            allStim2{index} = [prePulse, allStim{st}];

            gratingTable = [gratingTable; [table(index), stimTab(st, :), table(hypC, hypD)]];

        end

    end

else

    allStim2 = allStim;
    gratingTable = stimTab;

end

%% Bulding the stimulus structure

preStimDur = fixed.preStim * (fixed.sampRate/1000);
preStim = zeros(1, preStimDur);

allStim2 = cellfun(@(x) [preStim, x], allStim2, 'uniformoutput', 0);

reps = round(default.reps); % just in case user puts non-round number
assert(reps > 0, 'reps should be a positive number');

numUStim = height(gratingTable);
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
        protocolStructAO.stim(counter).aoVec = allStim2{tempInds(jj)};
        protocolStructAO.stim(counter).relInds = [tempInds(jj), tempInds(jj), 0, 0];
        protocolStructAO.stim(counter).length = length(allStim2{tempInds(jj)});
    end
end

numStim = length(protocolStructAO.stim);

protocolStructAO.gratingTable = gratingTable;

%% Generating empty pattern and posFunc

bkgdVal = default.arenaVal;
assert(ismember(bkgdVal, 0:15), 'arenaVal should be an integer between 0-15') %assumes gsLevel 4

timeStamp = datestr(now, 'yyyymmdd_HH-MM');
folderName = fullfile(pwd, ['stepData', timeStamp]);
[stat, mess] = mkdir(folderName);
assert(stat==1, 'error creating folder: %s', mess)

statVec = make_vSDAO_imageG4(protocolStructAO, folderName);
assert(statVec == 0, 'Problem creating AO files')

emptyStruct.stim(1).matCell = ones([fixed.arenaSize, 500]) * default.arenaVal; % to generate a non empty position function
emptyStruct.gsLevel = fixed.gsLevel;
emptyStruct.stim(1).posFuncCell = ones(1,1000);

statPat = make_vSDpattern_imageG4(emptyStruct, folderName);
statPos = make_vSDposfunction_imageG4(emptyStruct, folderName);
assert(statPat == 0, 'Problem creating pattern files')
assert(statPos == 0, 'Problem creating function files')


if ~isfolder(fullfile(folderName, 'Log Files'))
    mkdir(fullfile(folderName, 'Log Files'))
end

create_currentExp(folderName)

%% Establish panel host connection
connectHost %currently will break a connection if already open

Panel_com('change_root_directory', folderName);

Panel_com('set_control_mode', 1); % %0=streaming, 1=position function, 2=constant rate, 3=position change, 4=Closed-loop (CL), 5=CL+bias, 6=CL+OL

% sets the pattern and position functions to the empty ones
Panel_com('set_pattern_id', 1)
Panel_com('set_pattern_func_id', 1)
Panel_com('set_active_ao_channels', chInBin)


% Add waitbar to be able to abort protocol cleanly
wbh = waitbar(0,'1','Name','Presenting Protocol',...
            'CreateCancelBtn',...
            'setappdata(gcbf,''canceling'',1)');
setappdata(wbh,'canceling',0)

%% Running the experiment

Panel_com('reset_counter')

for ii=1:numStim

    Panel_com('start_log')
    pause(0.1)
    Panel_com('set_ao_function_id', [fixed.relCh, ii]) % since counting starts at zero
    waitbar(ii/numStim, wbh, sprintf('Presenting protocl %d of %d',ii, numStim))

    Panel_com('start_display', protocolStructAO.stim(ii).length/fixed.sampRate + fixed.fudgeT)
    pause(protocolStructAO.stim(ii).length/fixed.sampRate + fixed.fudgeT)
    
    Panel_com('stop_log');
    pause(0.01)

    % since stop log is no longer reporting the file name
    logDirs = dir([folderName, '\Log Files']);
    lgDrI = [logDirs.isdir] & cellfun(@length, {logDirs.name}) > 3; % gets rid of files or reference directories
    logDirs = logDirs(lgDrI); % select only directories

    [~,dirInd] = max([logDirs(:).datenum]);

    if ~isempty(dirInd)
        protocolStructAO.stim(ii).dirName = logDirs(dirInd).name;
    end

    if getappdata(wbh,'canceling')
        break
    end
end

%% Cleaning up after the experiment is done

delete(wbh)
pause(1)

protocolStructAO.type = 'currentInj';

% To get file names if function crashes while trying to move files
save(fullfile(folderName, ['protocolStructAO', timeStamp]), 'protocolStructAO', '-v7.3')

for ii=1:numStim
  if isempty(protocolStructAO.stim(ii).dirName)
    break
  else
    tempStimDir = fullfile(folderName, 'Log Files', protocolStructAO.stim(ii).dirName);
    G4_TDMS_folder2struct(tempStimDir)
  end
end

protocolStructAO = consolidateDataG4(folderName);

Panel_com('set_active_ao_channels', '0000')

% reminder to avoid colleting from blind flies
helpdlg('Use Flashlight before proceeding')

end
