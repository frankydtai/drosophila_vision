function maskPos = createMaskPosGrid(xVec, yVec)

% function maskPos = createMaskPosGrid(xVec, yVec)
%
% this function takes a vector for x positions and a vector for y positions
% and generate a 2D matric with all their combinations
%
% INPUT
%
% x/yVec -      1XN vector for the relevant positions
% 
% OUTPUT 
% 
% maskPos - Nx2 matrix of all combination with x in the first position

assert(isrow(xVec), 'xVec should be a 1XN vector')
assert(isrow(yVec), 'yVec should be a 1XN vector')

[allX,allY] = meshgrid(xVec,yVec);

preMaskPos = [allX', allY'];
maskPos = reshape(preMaskPos, [], 2);

end


