function randInds = randomizeMatrixInds(matSize, randDim, randWithinFlag)

% function randInds = randomizeMatrixInds(matSize, randDim)
%
% This functions generates a random permutation of the indicies of a matrix
% by permuting just the 1st dim, just the second or both. Verifies that
% each index is repeated only once. 
%
% INPUT
%
% matSize -         1X2 vector describing the size of the 2-dim matrix to be randomized
% randDim -         binary vector tagging which dimension to randomize. Should be same length as matSize  
% randWithinFlag -  logical (optional). in cases where just one dimension is
%                   randomized, this flag determines whether to shuffle the non-randomized
%                   dimension within the randomized one. 
%   1   2   vs  2   2
%   2   2       3   2
%   3   2       1   2
%   1   1       3   1
%   2   1       1   1
%   3   1       2   1   
%   
%
% OUTPUT
% randInds -    (MXN)X2 vector of indices into the required matrix (x,y
%               coordinates of the given matrix size

if nargin < 3
    randWithinFlag = 0;
end


if size(matSize) ~= 2
    error('MatSize should be refer to a 2D matrix')
else
    fDim = matSize(1);
    sDim = matSize(2);
end

if length(matSize) ~= length(randDim)
    error('matSize and randDim should be of equal length')
end

if ~prod(ismember((randDim), [0,1])) %verifies that only 0,1 are the input
    error('randDim should be a logical array')
end


randInds = zeros(fDim*sDim, 2);

if  isequal(randDim, [0,0]) % does not randomize
        
        for ii=1:(fDim*sDim)
             [xx, yy] = ind2sub(matSize, ii);
             randInds(ii,:) = [xx, yy];
        end
        
elseif isequal(randDim, [1, 0])
        
        permInds = randperm(fDim);
        for ii=1:fDim
            tInd = (ii-1)*sDim+1;
            randInds(tInd:(tInd+sDim-1), 1) = permInds(ii);
            if randWithinFlag
                randInds(tInd:(tInd+sDim-1), 2) = randperm(sDim);
            else
                randInds(tInd:(tInd+sDim-1), 2) = 1:sDim;
            end
        end
        
elseif isequal(randDim, [0, 1])
        
        permInds = randperm(sDim);
        for ii=1:sDim
            tInd = (ii-1)*fDim+1;
            if randWithinFlag
                randInds(tInd:(tInd+fDim-1), 1) = randperm(fDim);
            else
                randInds(tInd:(tInd+fDim-1), 1) = 1:fDim;
            end
            randInds(tInd:(tInd+fDim-1), 2) = permInds(ii);
        end
         
elseif isequal(randDim, [1, 1])  
            % randomizes the second dimension and then both together 
            %(if only the second randomization is used, the second dimension is not shuffeld enough)
        
        permInds = randperm(sDim);
        for ii=1:sDim
            tInd = (ii-1)*fDim+1;
            randInds(tInd:(tInd+fDim-1), 1) = 1:fDim;
            randInds(tInd:(tInd+fDim-1), 2) = permInds(ii);
        end
         
        tmpP = randperm(fDim*sDim);
        randInds = randInds(tmpP, :);
        
else
        error('randDim values should be 0s or 1s')
end
        
            
        






end