function adjustAllFigures

% function adjustAllFigures
%
% This function prompts the user to adjust the size of a figure and then
% adjusts the rest of the figures accordingly (all or just the given
% handles)
%



figHand = findobj('type', 'figure');
assert(~isempty(figHand), 'No open figures - aborted')

fprintf('Drag one figure to the desired size and press any key \n')

waitforbuttonpress;

relUnits = get(gcf, 'units');
relSize = get(gcf, 'position');

set(figHand(:), 'units', relUnits, 'position', relSize)


end