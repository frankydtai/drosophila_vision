function [qVal, qInd] = findQuantAndInd(relVec, startInd, stopInd, quan)

% This function get indices and finds the quantile and its ind within the
% relevant range.
%
% INPUT
% relVec -              1XN vector 
% start/stopInd -       indices into the relevant range in the vector
% quan -                vector of 0-1 requested quantiles
% 

assert(isvector(quan), 'quan should be a 1XM vector')
assert(min(quan) >0 & max(quan)<1, 'quan should be between 0-1')

qVal = nan(1, length(quan));
qInd = qVal;

for ii=1:length(quan)
    qVal(ii) = quantile(relVec(startInd:stopInd), quan(ii));
    preQInd = find(relVec(startInd:stopInd) > qVal(ii), 1, 'first');
    
    qInd(ii) = startInd - 1 + preQInd;
    
end




end