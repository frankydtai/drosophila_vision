function addLinetoPlot(axHands, xVec, yVec, varargin)

% function addline(axHands, xVec, yVec)
%
% This function adds a line to all the axes given in axHands
% xVec and yVec should both be 2 element vectors
%
% varargin can be used to input line specs (should be given in pairs

colFlag = 0;
varInFlag = 0;

if nargin > 3
    
    varInFlag = 1;
    testLen = nargin -3;
    assert(rem(testLen,2) == 0, 'lineSpecs inputs should be in pairs')
    
    if ismember('color', varargin)
        colFlag = 1;
    end
       
end
    

lineCol = [1,1,1]*0.85;

assert(length(xVec) == 2, 'xVec must be a 2 elelment vector')
assert(length(yVec) == 2, 'yVec must be a 2 elelment vector')

for ii=1:length(axHands)
    
    figH = get(axHands(ii), 'parent');
    figC = class(figH);
    
    assert(strcmpi(figC(end-5:end), 'figure'), 'given axes are contain within something other than a figure')
    
    set(figH, 'currentaxes', axHands(ii))
    
    if varInFlag && colFlag
        line(xVec, yVec, 'color', varargin)
    elseif varInFlag && ~colFlag
        line(xVec, yVec, 'color', lineCol, varargin)
    else
        line(xVec, yVec, 'color', lineCol)
    end
    
end


end
    
    
    
    