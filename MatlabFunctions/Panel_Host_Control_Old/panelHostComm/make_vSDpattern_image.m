
function totStat = make_vSDpattern_image(pStruct)


% this is the a changed version of make_flash_image where instead of
% file names the input is a protocol structure with .stim fields and the SD structure is deleted 
%
% output is simply the sum of fclose status


block_size = 512; % all data must be in units of block size
num_panels = 48;
load panelContConfigFileDir % saved in "C:\Users\gruntmane\Documents\ExpCodeandRes\MatlabFunctions\Panel_Host_Control"

if isfield(pStruct, 'gsLevel')
    gs_val = pStruct.gsLevel; % deals with protocolStructComb
elseif isfield(pStruct, 'gratingStruct')
    gs_val = pStruct.gratingStruct(1).gsLevel; % assumes they all have the same gsLevel 
else
    error('Missing gsLevel field')
end
num_patterns = length(pStruct.stim);
Header_block = zeros(1, block_size);
%SD.num_patterns = num_patterns;

%clean the temp folder
pConfig = fileread(panelContConfigFileDir);
pConfigFormatted = textscan(pConfig, '%s');
pathInd = find(cellfun(@(x) strcmp(x, '[Pattern]'), pConfigFormatted{1})) + 3; % add 3 since there is 'path', and '=' in between
temp_path = pConfigFormatted{1}{pathInd};


dos(['del /Q "' temp_path '\*.pat"']); % SS

stat = zeros(1, num_patterns);
for ii = 1:num_patterns
    relStim = pStruct.stim(ii);
    
    % took out row compression
    current_frame_size(ii) = num_panels*gs_val*8;
    x_num = size(relStim.patVecMat, 2);
    y_num = 1;
    
    blocks_per_frame(ii) = ceil(current_frame_size(ii)/block_size);
    current_num_frames(ii) = x_num*y_num;
    num_blocks_needed(ii) = blocks_per_frame(ii)*current_num_frames(ii);
    
    Header_block(1:8) = [dec2char(x_num,2), dec2char(y_num,2), num_panels, gs_val, dec2char(current_frame_size(ii),2)];
    
%     % set up SD structure with pattern info
%     SD.x_num(ii) = x_num;
%     SD.y_num(ii) = y_num;
%     SD.num_panels(ii) = num_panels;
%     SD.gs_val(ii) = gs_val; % unclear if we should change this to reflect 11, 12, 13 hack
%     SD.frame_size(ii) = current_frame_size(ii);
%     SD.pattNames{ii} = ['pattern_stim_', num2str(ii)];
    
    Pattern_Data = zeros(1, num_blocks_needed(ii)*block_size);
    
    block_indexer = 1;  % points to the first block
    % now write all of the frame info
    for jj = 1:current_num_frames(ii)
        sd_start_address = (block_indexer - 1)*block_size + 1;
        sd_end_address = sd_start_address + current_frame_size(ii) - 1;    
        % always forced to start frame at a block boundary
        pat_start_address = (jj - 1)*current_frame_size(ii) + 1;
        pat_end_address = pat_start_address + current_frame_size(ii) - 1;    
        Pattern_Data(sd_start_address:sd_end_address) = relStim.patVecMat(pat_start_address:pat_end_address);
        block_indexer = block_indexer + blocks_per_frame(ii);
    end    

    Data_to_write = [Header_block Pattern_Data];
    
    
    switch length(num2str(ii))
        case 1
            patFileName = ['pat000' num2str(ii) '.pat'];
        case 2
            patFileName = ['pat00', num2str(ii) '.pat'];
        case 3
            patFileName = ['pat0', num2str(ii) '.pat'];
        case 4
            patFileName = ['pat', num2str(ii) '.pat'];
        otherwise
            disp('The pattern number is too big.');
    end
    
    fid = fopen([temp_path '\' patFileName] , 'w');
    
    assert(fid ~= -1, 'Unable to create file %s', patFileName)
    fwrite(fid, Data_to_write(:),'uchar');
    stat(ii) = fclose(fid);
    
    %fprintf('%d of %d patterns written to Pattern folder: %s \n', ii, num_patterns, patFileName)
end


totStat = sum(stat);

fprintf('%d pattern files written to Pattern folder (out of %d) \n', ii, num_patterns)

end

