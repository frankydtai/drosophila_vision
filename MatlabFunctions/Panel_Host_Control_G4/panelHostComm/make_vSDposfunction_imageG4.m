function totStat = make_vSDposfunction_imageG4(pStruct, saveDir)


% function totStat = make_vSDposfunction_imageG4(pStruct, saveDir)
%
% This function was changed to generate a position function from the
% pStruct field 'posFuncCell' - instead of the previous version that simply
% generated a 1:len posFunc (changed Sep 2020)
% this is the change of make_function_image where the vel function part was
% taken out and the input was changed from fileNames to protocol structure
% with .stim field.
% Also deleted SD structure
%
% output is simply the sum of fclose status 

frameBuffer = 1000; % number of frames to add in the end of the function (so that stimulus wont go back to beginning
stimFreq = 500; % in Hz

num_functions = length(pStruct.stim);

if ~isfolder(saveDir)
  error('%s folder does not exsit', saveDir)
end


% make sure folder is empty (should have no files in Functions folder)
if isfolder(fullfile(saveDir, 'Functions'))
    testForFiles = dir(fullfile(saveDir, 'Functions'));
    filesInDir = testForFiles(~[testForFiles.isdir]);
    assert(isempty(filesInDir), 'Functions directory not empty')
else
  mkdir(fullfile(saveDir, 'Functions'))
end

stat = zeros(1, num_functions);

for ii = 1:num_functions
    relFun = pStruct.stim(ii).posFuncCell;
    relLen = length(relFun);
    func = [relFun, ones(1, frameBuffer)]; % shifts to zero when saved
    param.type = 'pfn';
    param.ID = ii;
    param.dur = relLen / stimFreq;
    stat(ii) = save_function_G4(func, param, [saveDir, '\Functions'], ['Function_' num2str(param.ID, '%04d') '_G4.mat']);

end

totStat = sum(stat);

fprintf('%d function files written to Function folder (out of %d) \n', ii, num_functions)

end
