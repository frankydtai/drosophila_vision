function stat = make_vSDAO_image_forVec(anaSig)

% function stat = make_vSDAO_image_forVec(anaSig)
%
% This function is identical to make_vSDAO_image, only it works for individual vectors instead of stim structures. 
% It is to be used for current injection purposes. 
% The input is a vector being sent out
% through the AO channel (-10 to 10). 
% 
% The vector will be played out at 1KHz.
%
% 
%
% output is simply the sum of fclose status

assert(isvector(anaSig), 'input should be a vector')
assert(max(anaSig) <= 10 && min(anaSig) >= -10, 'input exceeds -10 to 10V range')

% finding the directory in which to place file
load panelContConfigFileDir % saved in "C:\Users\gruntmane\Documents\ExpCodeandRes\MatlabFunctions\Panel_Host_Control"
pConfig = fileread(panelContConfigFileDir);
pConfigFormatted = textscan(pConfig, '%s');
pathInd = find(cellfun(@(x) strcmp(x, 'Output]'), pConfigFormatted{1})) + 3; % add 3 since there is 'path', and '=' in between
temp_path = pConfigFormatted{1}{pathInd};

dos(['del /Q "' temp_path '\*.ao"']); %SS




   
%   convert -10 to 10 into -32767 to 32767
corrInpVec = anaSig * 3276.7;

inpLen = dec2char(length(anaSig), 4); % first 4 bytes specify length of file
%   inpLen = fliplr(inpLen); % was used in big-endian version

AOData = signed_16Bit_to_char(corrInpVec);
%AOData  = reshape(flipud(reshape(AOData, 2, [])), 1, []); % same as for inpLen
    

funcFileName = 'anaOut0001.ao';
        
Data_to_write = [inpLen AOData]; 
fid = fopen([temp_path '\' funcFileName] , 'w');
assert(fid ~= -1, 'Unable to create file %s', funcFileName)
fwrite(fid, Data_to_write(:),'uchar');
stat = fclose(fid);



end
