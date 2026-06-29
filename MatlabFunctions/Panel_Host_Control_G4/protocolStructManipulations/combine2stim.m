function combStim = combine2stim(stim1, stim2, totalLength, firstFrame1, firstFrame2, bkgdLevel)

% function combStim = combine2stim(stim1, stim2, totalLength, firstFrame1, firstFrame2)
%
% This function combines 2 stim into one. It is used by
% runPosFunc2Protocols to generate a single stimulus from 2 protocol
% structures. Stimuli are combined by identifying pixels that differe from
% background levles and changing them only.
%
% INPUT
%
% stim1/2 -         arenaSize(1,2) X N matrix. N (3rd dim) is the number of frames
% totalLength -     total number of frames for the combined stim
% firstFrame1/2 -   frame within the total number in which the first frame
%                   of stim1/2 (respectively) will be placed
% bkgdLevel -       level of background (in panelController lum values)
%                   that is used in the stimuli. Function uses this value
%                   to identify the actual stim pixels in stim2 and
%                   changing only them in the combined version. 
% NOTE! bkgdLevel should be the same for both stim
%
% OUTPUT
% combStim -        stimulus containing both. 
%
% NOTE! any pixels that are shared by the 2 stim will have the value they
% have in the second stim. 


stSiz1 = size(stim1);
if length(stSiz1) == 2
    stSiz1(3) = 1;
end
stSiz2 = size(stim2);
if length(stSiz2) == 2
    stSiz2(3) = 1;
end

assert(stSiz1(1) == stSiz2(1) && stSiz1(2) == stSiz2(2), 'arena size is not identical between the 2 stim')
assert(firstFrame1 > 0 && firstFrame2 > 0, 'First frame cannot be a negative number')
assert(firstFrame1 + stSiz1(3) <= totalLength, 'Stim1 starting at the spacified frame exceeds totalLength')
assert(firstFrame2 + stSiz2(3) <= totalLength, 'Stim2 starting at the spacified frame exceeds totalLength')

combStim = ones(stSiz1(1), stSiz1(2), totalLength) * bkgdLevel;

combStim(:, :, firstFrame1:firstFrame1+stSiz1(3)-1) = stim1;

tempStim = combStim(:,:,firstFrame2:firstFrame2+stSiz2(3)-1);

tempStim(stim2 ~= bkgdLevel) = stim2(stim2 ~= bkgdLevel);

combStim(:,:,firstFrame2:firstFrame2+stSiz2(3)-1) = tempStim;


end