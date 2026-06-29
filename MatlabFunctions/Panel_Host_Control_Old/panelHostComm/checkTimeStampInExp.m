function relInds = checkTimeStampInExp(expFolder)

% This function was designed to test the inconsistencies in the length of
% ADC chnnels logged by Panel Host. It reads all the TDMS files in the
% directory looks for ones with inconsistent number of points and plot timestamps from the
% ADC channels 
%
% INPUT
% expFolder - directory that contains TDMS files
%
% OUTPUT
% relInds - idicies of files with wrong number of points 

fnames = dir(fullfile(expFolder, '*tdms'));
[~, fileNameInd] = sort([fnames.datenum], 'ascend');

numPoints = cell(length(fnames), 1);

for ii=1:length(fnames)
    tic
    tdmsSt = TDMS_readTDMSFile(fullfile(expFolder, fnames(fileNameInd(ii)).name));
    numPoints{ii} = tdmsSt.numberDataPointsRaw;
    toc
end

numUpoints = cellfun(@unique, numPoints, 'uniformoutput', 0);
ulength = cellfun(@length, numUpoints);
relInds = find(ulength == 4);


for ii=1:length(relInds)
    
    
    tdmsSt = TDMS_readTDMSFile(fullfile(expFolder, fnames(relInds(ii)).name));
    tsInds = findTimeInds(tdmsSt);
    tmStamp = tdmsSt.data(tsInds);
    cols = {'r', 'b', 'k'};
    figure
    hold on
    for jj=1:3
        plot(tmStamp{jj}, '-o', 'linewidth', 2, 'color', cols{jj})
    end
    title(['file number', num2str(relInds(ii))])
    hold off
end



end

%%

function TSinds = findTimeInds(tdmsStruct)

TSinds = zeros(1,3);


%find proper Indices for the data
for ii=1:3
    relStr = ['ADC', num2str(ii-1)];
    grpInd = find(cellfun(@(x) strcmp(x, relStr), tdmsStruct.groupNames));
    tInd = find(cellfun(@(x) strcmp(x, 'Time'), tdmsStruct.chanNames{grpInd}));
    TSinds(ii) = tdmsStruct.chanIndices{grpInd}(tInd);
end


end



