function out = gs2vec2dec_faster(vec)
% GS2VEC2DEC Convert a 2 level grayscale vector to the appropriate decimal number pair
% _fast is a version from 12/7/10
% point is to avoid dec2bin, bin2dec
% _faster just removes error checking and is restricted to 3 level gscale


bin_vals =  [0 0 0; 0 0 1; 0 1 0; 0 1 1; 1 0 0; 1 0 1; 1 1 0; 1 1 1];
bin2dec_mat = [1 2 4 8 16 32 64 128]; %[128 64 32 16 8 4 2 1];

% here we turn each value in the gsvec into a binvec itself
% suppose we are using 3 bits, e.g.
% 2 -> 0 1 1
% 4 -> 1 0 0

binvec = bin_vals((vec+1),:);

% then just turn each binvec into a dec value
out = (binvec'*bin2dec_mat')';
