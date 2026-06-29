function print2PDF(fileName, fh)

% function print2PDF(fileName)
%
% This function prints the current figure to a nicely sized PDF figure
% fh is optional figure handle (if not given uses gcf)

if nargin < 2
    fh = gcf;
end

set(fh,'Units','inches');
screenposition = get(fh,'Position');
set(fh,...
    'PaperPosition',[0 0 screenposition(3:4)],...
    'PaperSize',[screenposition(3:4)]);
print(fh, fileName, '-dpdf', '-painters', '-r0') 


end