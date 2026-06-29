function protStruct  = runPosFuncProtocolG4(funcHand, pStruct)

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
fudgeT = 0.25; % adds to session time to make sure pattern presentation is done (in sec)

if nargin < 1
    funcHand = listCreateProtocolFunctions;
end

connectHost;



%% run the desired function to generate the 32X96XN matrix to be presented
if nargin < 2
    protocolStruct = feval(funcHand);
else
    protocolStruct = feval(funcHand, pStruct);
end

% create experimental directory
timeStamp = datestr(now, 'yyyymmdd_HH-MM');
funcStr = func2str(funcHand);
folderName = fullfile(pwd, [funcStr(7:end), timeStamp]); %gets rid of the word 'create'
[stat, mess] = mkdir(folderName);
assert(stat==1, 'error creating folder: %s', mess)

Panel_com('change_root_directory', folderName);

assert(isfield(protocolStruct, 'stim'), 'Protocol structure is missing stim field')
numStim = length(protocolStruct.stim);

statPat = make_vSDpattern_imageG4(protocolStruct, folderName);
statPos = make_vSDposfunction_imageG4(protocolStruct, folderName);
assert(statPat == 0, 'Problem creating Patterns files')
assert(statPos == 0, 'Problem creating Functions files')

% will be empty for these protocol types - but needed
if ~isfolder(fullfile(folderName, 'Analog Output Functions'))
    mkdir(fullfile(folderName, 'Analog Output Functions'))
end

if ~isfolder(fullfile(folderName, 'Log Files'))
    mkdir(fullfile(folderName, 'Log Files'))
end

% Add waitbar to be able to abort protocol cleanly
wbh = waitbar(0,'1','Name','Presenting Protocol',...
            'CreateCancelBtn',...
            'setappdata(gcbf,''canceling'',1)');
setappdata(wbh,'canceling',0)

create_currentExp(folderName)

save(fullfile(folderName, ['protocolStruct', timeStamp]), 'protocolStruct', '-v7.3')

%% Starts the experiment
% Config 2 block a third of the arena to speed up performance
% Since T4s respond to middle of arena it should be avoided

Panel_com('set_control_mode', 1); % %0=streaming, 1=position function, 2=constant rate, 3=position change, 4=Closed-loop (CL), 5=CL+bias, 6=CL+OL
Panel_com('reset_counter')

figH = figure('position', [1450, 50, 450, 150]);
figH.MenuBar = 'none';
maxValforFig = 2^4-1; % since gsLevel is 4

for ii=1:numStim

    Panel_com('start_log'); % to minimize non-recorded time loop iteration begin and end in log commands
    pause(0.1)
    relFreq = round(protocolStruct.generalFrequency/protocolStruct.stim(ii).freqCorr); % to avoid weird behavior

    Panel_com('set_pattern_id', ii)
    Panel_com('set_pattern_func_id', ii)
    Panel_com('set_position_x', 1);
    stimTime = length(protocolStruct.stim(ii).posFuncCell)/relFreq;

    waitbar(ii/numStim, wbh, sprintf('Presenting protocl %d of %d',ii, numStim))
    plotMidFrameG4(mean(protocolStruct.stim(ii).matCell,3), maxValforFig)
    tH = title(num2str(protocolStruct.stim(ii).relInds));
    tH.VerticalAlignment = 'top';

    Panel_com('start_display', stimTime + fudgeT) %duration expected in 100ms units
    pause(stimTime + fudgeT)

    Panel_com('stop_log');
    pause(0.1)

    logDirs = dir([folderName, '\Log Files']);
    lgDrI = [logDirs.isdir] & cellfun(@length, {logDirs.name}) > 3; % gets rid of files or reference directories
    logDirs = logDirs(lgDrI); % select only directories

    [~,dirInd] = max([logDirs(:).datenum]);

    if ~isempty(dirInd)
        protocolStruct.stim(ii).dirName = logDirs(dirInd).name;
    end

    if getappdata(wbh,'canceling')
        break
    end

end

%% clean up after exp is done
delete(wbh)


% To get file names if function crashes while trying to move files
save(fullfile(folderName, ['protocolStruct', timeStamp]), 'protocolStruct', '-v7.3')
count = 0;
for ii=1:numStim
  if isempty(protocolStruct.stim(ii).dirName)
    break
  else
      count=count+1; % changed the loop to allow for parfor use
      tempStimDir = fullfile(folderName, 'Log Files', protocolStruct.stim(ii).dirName);
      tempDirCell{count} = tempStimDir;
  end
end

parfor ii=1:length(tempDirCell)
    G4_TDMS_folder2struct(tempDirCell{ii})
end


protStruct = consolidateDataG4(folderName);
close(figH)


end
