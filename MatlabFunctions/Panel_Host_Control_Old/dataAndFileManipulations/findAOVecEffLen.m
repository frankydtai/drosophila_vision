function aoEffLen = findAOVecEffLen(aoVec)

% This function finds the first zero in the long string of zeros at the end
% and returns the length of the vector w/o padding ztring of zeros.
% 
% The function is used by runPosFuncProtocolwAO to find what is the length
% of each experiment (longer between pattern and effective AO)
% aoVec should be a 1XN vector

totLength = length(aoVec);
firstNumFromEnd = find(fliplr(aoVec), 1, 'first');

aoEffLen = totLength - firstNumFromEnd + 1;

end