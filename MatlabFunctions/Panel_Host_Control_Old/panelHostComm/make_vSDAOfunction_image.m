function stat = make_vSDAOfunction_image(inputVec)

% This is the change of make_function_image where the vel function part was
% taken out and the input was changed to be the vector being sent out
% through the AO channel (-10 to 10). InputVec will be played out at 1KHz
%
% 
%
% output is simply the sum of fclose status

% finding the directory in which to place file
panelContConfigFile = 'F:\Panel Host\Support Files\HHMI Panels Configuration.ini';
pConfig = fileread(panelContConfigFile);
pConfigFormatted = textscan(pConfig, '%s');
pathInd = find(cellfun(@(x) strcmp(x, 'Output]'), pConfigFormatted{1})) + 3; % add 3 since there is 'path', and '=' in between
temp_path = pConfigFormatted{1}{pathInd};

aoF = dir([temp_path, '\*.ao']);
fileNum = length(aoF)+1;

assert(max(inputVec) <= 10 && min(inputVec) >= -10, 'inputVec values are out of range (-10 to 10)')
% convert -10 to 10 into -32767 to 32767
corrInpVec = inputVec * 3276.7;

inpLen = dec2char(length(inputVec), 4); % first 4 bytes specify length of file
%inpLen = fliplr(inpLen); % was used in big-endian version

AOData = signed_16Bit_to_char(corrInpVec);
%AOData  = reshape(flipud(reshape(AOData, 2, [])), 1, []); % same as for inpLen
funcFileName = ['anaOut00', num2str(fileNum) '.ao'];
        
Data_to_write = [inpLen AOData]; 
fid = fopen([temp_path '\' funcFileName] , 'w');
assert(fid ~= -1, 'Unable to create file %s', funcFileName)
fwrite(fid, Data_to_write(:),'uchar');
stat = fclose(fid);


end
