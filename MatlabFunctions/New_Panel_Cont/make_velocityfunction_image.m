function make_velocityfunction_image(velfunc, direct)

if nargin < 2
    direct = pwd;
end

block_size = 512; % all data must be in units of block size
Header_block = zeros(1, block_size);

Header_block(1:4) = dec2char(length(velfunc)*2, 4); 
funcFileName = 'vel001.fun';
function_Data = signed_16Bit_to_char(round(20.*velfunc));   % 20 = 1V

Data_to_write = [Header_block function_Data]; 
fid = fopen([direct '\' funcFileName] , 'w');
fwrite(fid, Data_to_write(:),'uchar');
stats = fclose(fid);

if stats == -1
    error('problem creating velocity function file')
end

end