function stat = make_vSDAO_imageG4(pStruct, saveDir)

% This is the change of make_function_image where the vel function part was
% taken out and the input was changed to be the vector being sent out
% through the AO channel (-10 to 10).
% pStruct should have a .stim.aoVec field in it (generate in
% runPosFuncProtocolwAO)
% Each vector will be played out at 1KHz.
%
%
%
% output is simply the sum of fclose status

if ~isfolder(saveDir)
  error('%s folder does not exsit', saveDir)
end

assert(isfield(pStruct, 'stim'), 'pStruct is missing .stim field')
assert(isfield(pStruct.stim, 'aoVec'), 'pStruct.stim missing .aoVec field')

numAO = length(pStruct.stim); % each stimulus has the file associated with it (redundant)

% make sure folder is empty (should have no files in Patterns folder)
if isfolder(fullfile(saveDir, 'Analog Output Functions'))
    testForFiles = dir(fullfile(saveDir, 'Analog Output Functions'));
    filesInDir = testForFiles(~[testForFiles.isdir]);
    assert(isempty(filesInDir), 'Analog Output Functions directory not empty')
  else
    mkdir(fullfile(saveDir, 'Analog Output Functions'))
end


stat = zeros(1, numAO);

for ii=1:numAO
    relAOV = pStruct.stim(ii).aoVec;
    assert(max(relAOV) <= 10 && min(relAOV) >= -10, 'inputVec %d values are out of range (-10 to 10)', ii)
    %   convert -10 to 10 into -32767 to 32767
    % corrInpVec = relAOV * 3276.7; %converted in save_function_G4

    param.type = 'afn';
    param.ID = ii;

    stat(ii) = save_function_G4(relAOV, param, [saveDir, '\Analog Output Functions'], ['AO_' num2str(param.ID, '%04d') '_G4.mat']);
    pStruct.stim(ii).aoVecFileName = ['AO_' num2str(param.ID, '%04d') '_G4.mat'];

end

stat = sum(stat);
fprintf('%d AO files written to Analog Output Functions folder (out of %d) \n', ii, numAO)

end
