function pretestPattern(patternStruct, freq, posFunc)

% function pretestPattern(patternStruct, freq, posFunc)
%
% This function presents the given pattern using the freq and the position
% functions given to run it. Function starts to present from 1,1, and then
% uses the position functions. Displays the pattern 3 times
%
% INPUT
% patternStruct - regular pattern structure needed for the panel controller
% freq -          This function assumes the same freqeuncy for X and Y, so only one
%                 number has to be given in Hz
% posFunc -       This variable can be given in 2 formats: (1) Cell array with both
%                 position functions for X and Y (should be of equal length, 
%                 first is x second is y, and in 1XN form). 
%                (2) Cell array with one position function and one constant value
%
% Note position function should be given the same way it is given to the
% controller (with the first position in the matrix being zero). And so
% should the onstant value

times = 1;

patt = patternStruct.Pats+1; % to work properly with image
maxval = 2^patternStruct.gs_val; 

xfunc = posFunc{1}; 
yfunc = posFunc{2};
xfuncdim = size(xfunc); 
yfuncdim = size(yfunc); 

if xfuncdim(1)*yfuncdim(1) > 1
    error('Position functions should be either 1XN vectors or a single number')
end

if xfuncdim(2) == 1 && yfuncdim(2)> 1
    xfunc = ones(yfuncdim)* (xfunc+1);
    yfunc = yfunc+1;
elseif xfuncdim(2) > 1 && yfuncdim(2) == 1
    xfunc = xfunc+1;
    yfunc = ones(xfuncdim)* (yfunc+1); 
elseif (xfuncdim(2) == yfuncdim(2)) && xfuncdim(2) > 1
    xfunc = xfunc+1;
    yfunc = yfunc+1;
else 
    error('Position functions are in a wrong format')
end

figure
set(gcf, 'position', [ 900   630   870   280])
cmap = zeros(maxval,3);
cmap(:,2) = linspace(0,1,maxval);
colormap(cmap)

imh = image('CData', patt(:,:,1,1), 'CDataMapping', 'direct');
% draws red lines for panels

set(gca, 'xtick', [], 'ytick', [], 'clim', [0,1], 'drawmode', 'fast', ...
    'xlim', [0.5, 96.5], 'ylim', [0.5, 32.5])
set(imh, 'erasemode', 'none')

% arrayfun(@(x) line([x x], [0.5 32.5], 'linewidth', 2, 'color', 'r'), 8:8:95)
% arrayfun(@(x) line([0.5 96.5], [x x], 'linewidth', 2, 'color', 'r'), 8:8:31)
for kk=1:times
    for ii=1:length(xfunc)
        set(imh,'CData',flipud(patt(:,:,xfunc(ii), yfunc(ii))),'CDataMapping', 'direct')
        drawnow
        pause(1/freq);
    end
end
    






end