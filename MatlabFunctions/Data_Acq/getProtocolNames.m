function varargout = getProtocolNames(expSt, protInd)

% function varargout = getProtocolNames(expSt, protInd)
%
% This function returns the names of the pattern and position functions
% used in the protocol specified by protInd
%
% INPUTS
% expSt -   expStruct after it was created w/ createExpStruct and modified w/
%           createProtocolScript
% protInd - index of the required protocol (could be a number or vector)
%
% OUTPUT
% if an output is given the names will be given in a cell array. Otherwise
% they will be printed on the screen

indLen = length(protInd);
output = cell(indLen,3);

for ii=1:indLen
    patInd = expSt.protocol(protInd(ii)).PatternID;
    posxInd = expSt.protocol(protInd(ii)).PosFuncX(2);
    posyInd = expSt.protocol(protInd(ii)).PosFuncY(2);

    output{ii,1} = expSt.pattern(patInd).name;
    output{ii,2} = expSt.posFunc(posxInd).name;
    output{ii,3} = expSt.posFunc(posyInd).name;
end

if nargout
    varargout{1} = output;
else
    for ii=1:indLen 
        fprintf('Protocol %d data:\n pattern: %s\n posFuncX: %s\n posFuncY: %s\n\n', ...
                    protInd(ii), output{ii,1}, output{ii,2}, output{ii,3})
    end
end



end