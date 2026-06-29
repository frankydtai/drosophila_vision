function totStat = make_vSDposfunction_image(pStruct)

% this is the change of make_function_image where the vel function part was
% taken out and the input was changed from fileNames to protocol structure
% with .stim field. 
% Also deleted SD structure
%
% output is simply the sum of fclose status

frameBuffer = 1000; % number of frames to add in the end of the function (so that stimulus wont go back to beginning
block_size = 512; % all data must be in units of block size
load panelContConfigFileDir % saved in "C:\Users\gruntmane\Documents\ExpCodeandRes\MatlabFunctions\Panel_Host_Control"

num_functions = length(pStruct.stim);
Header_block = zeros(1, block_size);
%SD.numfunc = num_functions;

%clean the temp folder
pConfig = fileread(panelContConfigFileDir);
pConfigFormatted = textscan(pConfig, '%s');
pathInd = find(cellfun(@(x) strcmp(x, '[Function]'), pConfigFormatted{1})) + 3; % add 3 since there is 'path', and '=' in between
temp_path = pConfigFormatted{1}{pathInd};

dos(['del /Q "' temp_path '\*.fun"']); %SS

pos_func_counter = 0;
stat = zeros(1, num_functions);

for ii = 1:num_functions
    relLen = size(pStruct.stim(ii).patVecMat, 2);
    func = [(1:relLen)-1, zeros(1, frameBuffer)]; % assumes first frame is empty
    
    Header_block(1:4) = dec2char(length(func)*2, 4);     %each function datum is stored in two bytes in the SD card
    fileName = ['position_stim_', num2str(ii)];
    Header_block(5) = length(fileName);
    Header_block(6: 6 + length(fileName) -1) = fileName;
 
    % set up SD structure with function info
    %SD.functionName{ii} = fileName;
    %SD.functionSize{ii} = length(func)*2; 

    
    % eliminated the vel function part from 
    pos_func_counter = pos_func_counter + 1;
    switch length(num2str(pos_func_counter))
        case 1
            funcFileName = ['pos000' num2str(pos_func_counter) '.fun'];
        case 2
            funcFileName = ['pos00', num2str(pos_func_counter) '.fun'];
        case 3
            funcFileName = ['pos0', num2str(pos_func_counter) '.fun'];
        case 4
            funcFileName = ['pos', num2str(pos_func_counter) '.fun'];
        otherwise
            disp('The number of function you choose exceeds the maximum.');
            break;
    end
    %SD.posFunctionName{pos_func_counter} = file_list(ii).FileName;
    function_Data = signed_16Bit_to_char(func);     
    
        
    Data_to_write = [Header_block function_Data]; 
    fid = fopen([temp_path '\' funcFileName] , 'w');
    assert(fid ~= -1, 'Unable to create file %s', funcFileName)
    fwrite(fid, Data_to_write(:),'uchar');
    stat(ii) = fclose(fid);

    %fprintf('%d of %d function written to Function folder: %s \n', ii, num_functions, funcFileName)
end

totStat = sum(stat);

fprintf('%d function files written to Function folder (out of %d) \n', ii, num_functions)

end
