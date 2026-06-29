function copyLogFiletoCurrDir(fileNameCellArray, newFolderName)

% function moveLogFiletoCurrDir(fileNameCellArray)
%
% This function copies the log files named in the fileNameCellArray into
% the new given folder
%
% fileNameCellArray -   cell array of file names to transfer
% newFolderName     -   String. Can be Full path or relative path and name of new folder

load logDir

folderStat = exist(newFolderName, 'dir');

if folderStat == 0 % directory does not exist
    [stat, mess] = mkdir(newFolderName);
    assert(stat==1, 'error creating folder: %s', mess)
end

relFlag = cellfun(@(x) ~isempty(x), fileNameCellArray);
relCellArray = fileNameCellArray(relFlag);

for ii=1:length(relCellArray)
    fileToCopy = fullfile(logDir, relCellArray{ii});
    destFile = fullfile(newFolderName, relCellArray{ii});
    
    [tempStat, tempMess] = copyfile(fileToCopy, destFile);
    
    assert(tempStat == 1, 'error %s in copying %s', tempMess, relCellArray{ii})
    
end



end
