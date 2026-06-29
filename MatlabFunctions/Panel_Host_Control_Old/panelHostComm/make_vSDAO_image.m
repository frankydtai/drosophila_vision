function stat = make_vSDAO_image(pStruct)

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

assert(isfield(pStruct, 'stim'), 'pStruct is missing .stim field')
assert(isfield(pStruct.stim, 'aoVec'), 'pStruct.stim missing .aoVec field')

numAO = length(pStruct.stim); % each stimulus has the file associated with it (redundant)

% finding the directory in which to place file
load panelContConfigFileDir % saved in "C:\Users\gruntmane\Documents\ExpCodeandRes\MatlabFunctions\Panel_Host_Control"

pConfig = fileread(panelContConfigFileDir);
pConfigFormatted = textscan(pConfig, '%s');
pathInd = find(cellfun(@(x) strcmp(x, 'Output]'), pConfigFormatted{1})) + 3; % add 3 since there is 'path', and '=' in between
temp_path = pConfigFormatted{1}{pathInd};

dos(['del /Q "' temp_path '\*.ao"']); %SS

stat = zeros(1, numAO);

for ii=1:numAO
    relAOV = pStruct.stim(ii).aoVec;
    assert(max(relAOV) <= 10 && min(relAOV) >= -10, 'inputVec %d values are out of range (-10 to 10)', ii)
    %   convert -10 to 10 into -32767 to 32767
    corrInpVec = relAOV * 3276.7;

    inpLen = dec2char(length(relAOV), 4); % first 4 bytes specify length of file
    %   inpLen = fliplr(inpLen); % was used in big-endian version

    AOData = signed_16Bit_to_char(corrInpVec);
    %AOData  = reshape(flipud(reshape(AOData, 2, [])), 1, []); % same as for inpLen
    
    addedZeros = repmat('0', 1, 4-length(num2str(ii)));
    funcFileName = ['anaOut', addedZeros, num2str(ii) '.ao'];
        
    Data_to_write = [inpLen AOData]; 
    fid = fopen([temp_path '\' funcFileName] , 'w');
    assert(fid ~= -1, 'Unable to create file %s', funcFileName)
    fwrite(fid, Data_to_write(:),'uchar');
    stat(ii) = fclose(fid);
end

stat = sum(stat);
fprintf('%d function files written to Function folder (out of %d) \n', ii, numAO)

end
