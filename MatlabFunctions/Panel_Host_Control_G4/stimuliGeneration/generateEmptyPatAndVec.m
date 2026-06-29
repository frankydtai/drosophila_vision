function generateEmptyPatAndVec(uniformVal, saveDir)

% function generateEmptyPatAndVec
%
% This function allows to use of the G4 panel controller just as nidaq
% board (for AO and AI recording) without presenting any pattern. It
% generates an empty pattern and an empty posFunc vector, so that the start
% command plays nothing while recording AI or gnerating AO
%
% INPUTS
%
% uniformVal -  value to be presneted on the entire arena (assuming GS
% level defualt)
% saveDir -     experimantal directory that wull contain the Patterns and Functions folders
%
%
% OUTPUT
%
% files will be generated in the relevant directory for patterns and
% position functions (based on panelContConfigFileDir)


 % Note!! Function not corrected for G4 yet

%% default parameters
% Two parameters are irrelevant for G4
% block_size = 512; % all data must be in units of block size
% num_panels = 48;
arenaSize = [48,192];
gsLevel = 4;

assert(uniformVal <= 2^gsLevel -1, 'Value out of range')



%% Generating empty pattern
% Cleaning pattern dir
load panelContConfigFileDir
pConfig = fileread(panelContConfigFileDir);
pConfigFormatted = textscan(pConfig, '%s');
pathInd = find(cellfun(@(x) strcmp(x, '[Pattern]'), pConfigFormatted{1})) + 3; % add 3 since there is 'path', and '=' in between
temp_path = pConfigFormatted{1}{pathInd};


dos(['del /Q "' temp_path '\*.pat"']); % SS
Header_block = zeros(1, block_size);

emptyPat = ones([arenaSize, 1]) * uniformVal;
emptyPatVec = convertPatternMatrix(emptyPat);


current_frame_size = num_panels*gsLevel*8;
x_num = size(emptyPatVec, 2);
y_num = 1;

blocks_per_frame = ceil(current_frame_size/block_size);
current_num_frames = x_num*y_num;
num_blocks_needed = blocks_per_frame*current_num_frames;

Header_block(1:8) = [dec2char(x_num,2), dec2char(y_num,2), num_panels, gsLevel, dec2char(current_frame_size,2)];

Pattern_Data = zeros(1, num_blocks_needed*block_size);

block_indexer = 1;  % points to the first block
% now write all of the frame info; cluncky since I simply copied it from
% the function that actually writes patterns
for jj = 1:current_num_frames
    sd_start_address = (block_indexer - 1)*block_size + 1;
    sd_end_address = sd_start_address + current_frame_size - 1;
    % always forced to start frame at a block boundary
    pat_start_address = (jj - 1)*current_frame_size + 1;
    pat_end_address = pat_start_address + current_frame_size - 1;
    Pattern_Data(sd_start_address:sd_end_address) = emptyPatVec(pat_start_address:pat_end_address);
    block_indexer = block_indexer + blocks_per_frame;
end

Data_to_write = [Header_block Pattern_Data];
fid = fopen([temp_path '\pat0001.pat'] , 'w');

assert(fid ~= -1, 'Unable to create pattern file')
fwrite(fid, Data_to_write(:),'uchar');
stat = fclose(fid);

assert(stat == 0, 'Failed to close pattern file')

%% Generating empty position function

% finding path and deleting old files from directory
pathInd = find(cellfun(@(x) strcmp(x, '[Function]'), pConfigFormatted{1})) + 3; % add 3 since there is 'path', and '=' in between
temp_path = pConfigFormatted{1}{pathInd};

dos(['del /Q "' temp_path '\*.fun"']); %SS

Header_block = zeros(1, block_size);

func = zeros(1, 1000);

Header_block(1:4) = dec2char(length(func)*2, 4);     %each function datum is stored in two bytes in the SD card
funcFileName = 'pos0001.fun';
function_Data = signed_16Bit_to_char(func);

Data_to_write = [Header_block function_Data];
fid = fopen([temp_path '\' funcFileName] , 'w');
assert(fid ~= -1, 'Unable to create function file')
fwrite(fid, Data_to_write(:),'uchar');
stat = fclose(fid);

assert(stat == 0, 'Failed to close posFunc file')


end
