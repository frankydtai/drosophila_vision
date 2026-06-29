function fullProtStruct = consolidateDataG4(direct)

% function fullProtStruct = consolidateDataG4(direct)
%
% This function is a modification of consolidateData. It no longer converts the
% TDMS files just consolidates them into the protcolStruct.
% loads the protocol structure in the given directory, takes
% the mat file (TDMS converted) names from it, and gets the relevant data from them. This
% is then saved as a single field in the loaded protocol structure.
%
% INPUT
% direct - directory that both the protocol structure with .stim.fileNames field
%          and the 'Log Files' directory, where the TDMS directories and the Log files are saved
%         (after they have been converted using G4_TDMS_folder2struct)
%
% OUTPUT
% fullProtStruct - after the protocol structure is loaded it is saved with
% the relevant data (Analog inputs, and x Position).

% confirm that there is only one protocolStruct in the folder
pTest = dir(fullfile(direct, 'protocolStruct*.mat'));
assert(~isempty(pTest), 'no protocol structure in directory')
assert(length(pTest) == 1, 'More than one protocol structure in directory')
assert(isfolder(fullfile(direct, 'Log Files')), 'Log Files Directory missing')


load(fullfile(direct, pTest.name));
if strcmp(pTest.name(15:18), 'Comb')
    protFlag = 1;
    protocolStruct = protocolStructComb;
elseif strcmp(pTest.name(15:16), 'AO')
    protFlag = 2;
    protocolStruct = protocolStructAO;
else
    protFlag = 0;
end


for ii=1:length(protocolStruct.stim)

    fname = ['G4_TDMS_Logs_', protocolStruct.stim(ii).dirName];
    if isempty(protocolStruct.stim(ii).dirName)
        fprintf('No file for stimulus %d \n', ii)
        continue
    end
    load(fullfile(direct, 'Log Files', fname)); % loads Log variable
    % added double so that it won't zero VOlts (since Time is in uint64)
    inputChDat = [double(Log.ADC.Time(1,:))', Log.ADC.Volts']; % since time is euqal between channels
    xPosDat = [Log.Frames.Time', Log.Frames.Position'];
    commTime = Log.Commands.Time;
    commName = Log.Commands.Name;
    protocolStruct.stim(ii).data{1} = inputChDat;
    protocolStruct.stim(ii).data{2} = xPosDat;
    protocolStruct.stim(ii).comm{1} = commTime;
    protocolStruct.stim(ii).comm{2} = commName;
    if protFlag == 2
        outputChDat = [Log.AO.Time(1,:)', Log.AO.Volts']; % since time is euqal between channels
        protocolStruct.stim(ii).data{3} = outputChDat;
    end
    clear Log
    fprintf('Exctraced data from stim %d of %d\n', ii, length(protocolStruct.stim))

end

if protFlag == 1
    protocolStructComb = protocolStruct;
    save(fullfile(direct, pTest.name), 'protocolStructComb', '-v7.3')
elseif protFlag == 2
    protocolStructAO = protocolStruct;
    save(fullfile(direct, pTest.name), 'protocolStructAO', '-v7.3')
else
    save(fullfile(direct, pTest.name), 'protocolStruct', '-v7.3')
end

fprintf('Structure was saved with input data consolidated \n')
fullProtStruct = protocolStruct;




end
