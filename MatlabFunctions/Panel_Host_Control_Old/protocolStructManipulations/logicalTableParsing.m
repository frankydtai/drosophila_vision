

function subTab = logicalTableParsing(xTab, stringCond)

% function subTab = logicalTableParsing(xTab, stringCond)
%
% This function uses any logical statment on a single table variable to
% generate a smaller table. 
%
% INPUT
% xTab -        Table with several variable names (not left empty)
% stringCond -  logical condition given as a string where the table is
%               referenced as xTab. 
%
% OUTPUT
% subTab -      table with the same properties and variables only limited
%               to rows meeting the logical condition
%
% Examples:
% A table with 'width' variable with values 2,3 and 4 can be used to
% generate a table with 'width' eqaul only to 4 by using the statment:
% 'xTab.width == 4' or with 2 and 4 by using the statement
% 'ismember(xTab.width, [2,4])'
%
% A table with 'orient' variable from 0 to 7 can generate only the diagonal
% directions by using the condition 'rem(xTab.orient,2) == 1'




assert(~isempty(strfind(stringCond, 'xTab')), 'stringCond should have refer to table as xTab')

varNames = xTab.Properties.VariableNames;

varFind = cellfun(@(x) strfind(stringCond, x), varNames, 'uniformoutput', 0);
assert(any(cellfun(@(x) ~isempty(x), varFind)), 'none of the table variable names appear in stringCond')

tabInd = eval(stringCond);
assert(~isempty(tabInd), 'condition evaluated to empty table')

subTab = xTab(tabInd, :);


end
