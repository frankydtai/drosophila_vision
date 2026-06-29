
function createExpStruct(direct)

% function createExpStruct(direct)
% This function should be applied on the on_SDcard directory (that contains
% both pattern and position functions). The function will first rename all
% the files by adding enumerating both patterns and position functions and
% then create an expStruct structure, which contains details on all the
% patterns and position functions.
% For patterns it contains name, x and y frames, and date last modified
% for posFunc it contains name, number of unique values, total length of
% function, and whether it is intended for X or Y (based on the name), and
% date last modified
% The output is the expStruct saved in the directory given as an input

renameSDfiles(direct)


allFiles = dir(fullfile(direct, '*.mat'));
namesCell = {allFiles(:).name};

patternInd = find(cellfun(@(x) strcmpi('pattern', x(1:7)), namesCell));
posFuncInd = find(cellfun(@(x) strcmpi('position_func', x(1:13)), namesCell));


for ii=1:length(patternInd)
    
    load(fullfile(direct, namesCell{patternInd(ii)})); 
    
    expStruct.pattern(ii).name =  namesCell{patternInd(ii)}(1:(end-4));
    expStruct.pattern(ii).x_frames = pattern.x_num;
    expStruct.pattern(ii).y_frames = pattern.y_num;
    expStruct.pattern(ii).modDate = allFiles(patternInd(ii)).date;
    
    clear pattern
    
end

for ii=1:length(posFuncInd)
    
    load(fullfile(direct, namesCell{posFuncInd(ii)})); 
    
    expStruct.posFunc(ii).name =  namesCell{posFuncInd(ii)}(1:(end-4));
    expStruct.posFunc(ii).numUniqueVals = length(unique(func));
    expStruct.posFunc(ii).length = length(func);
    if ~isempty( regexp(namesCell{posFuncInd(ii)}, '_X_', 'ONCE'))
        expStruct.posFunc(ii).relDim = 'X';
    elseif ~isempty( regexp(namesCell{posFuncInd(ii)}, '_Y_', 'ONCE'))
         expStruct.posFunc(ii).relDim = 'Y';
    end
    expStruct.posFunc(ii).modDate = allFiles(posFuncInd(ii)).date;
    
    clear func
    
end

save(fullfile(direct, 'expStruct.mat'), 'expStruct')

end