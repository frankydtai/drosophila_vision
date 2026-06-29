function renameSDfiles(direct)

% This functions renames all the files in a the SD directory and adds 
% number to the patterns and position functions seperately. 
%
% INPUT
% direct - relevant directory where position functions and pattern for the
% experiment reside. 
% Function will only deal w/ 'Pattern_*' and 'position_function_*' files


%aux function from stephen
pad_num2str_w_zeros = @(num,num_zeros)([repmat('0',1,num_zeros - numel(num2str(num))) num2str(num)]);


allFiles = dir(fullfile(direct, '*.mat'));
namesCell = {allFiles(:).name};

patternInd = find(cellfun(@(x) strcmpi('pattern', x(1:7)), namesCell));
posFuncInd = find(cellfun(@(x) strcmpi('position_func', x(1:13)), namesCell));



for ii=1:length(patternInd)
    tempCheck = namesCell{patternInd(ii)}(9:11); % assumes 3 digit code
    if ~isnan(str2double(tempCheck))
        disp('Pattern file already contains number - enumerating patterns aborted')
        break
    end
    
    newname = ['Pattern_', pad_num2str_w_zeros(ii, 3), namesCell{patternInd(ii)}(8:end)];
    movefile(fullfile(direct, namesCell{patternInd(ii)}),fullfile(direct, newname)) 
end
    

for ii=1:length(posFuncInd)
    tempCheck = namesCell{posFuncInd(ii)}(19:21); % assumes 3 digit code
    if ~isnan(str2double(tempCheck))
        disp('position function file already contains number - enumerating posFunc aborted')
        break
    end
    
    
    newname = ['position_function_', pad_num2str_w_zeros(ii, 3), namesCell{posFuncInd(ii)}(18:end)];
    movefile(fullfile(direct, namesCell{posFuncInd(ii)}),fullfile(direct, newname)) 
    
end



end

