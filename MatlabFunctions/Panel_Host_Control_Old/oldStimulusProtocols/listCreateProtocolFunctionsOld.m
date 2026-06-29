function funcHand = listCreateProtocolFunctions

% function funcHand = listCreateProtocolFunctions
%
% This function lists the available createXXXProtocol functions and
% provides a function handle to the desired one

fprintf(' \n Available createXXXProtocol functions: \n')

avFuncCell = {'createGratingProtocol'; ...
              'createGratingFadeProtocol'; ...
              'createEdgeProtocol'; ...
              'createEdgeFadeProtocol'; ...
              'createCenterSurroundProtocol'; ...
              'createCenterSurroundProtocol2'; ...
              'createMovingLinearObjectProtocol';...
              'createDirectionalEdgeProtocol'; ...
              'createMinimalMotionProtocol';...
              'createSingleStripeProtocol';...
              'createMinimalMotionStripeProtocol';...
              'createMinimalMotionStripewNoiseModProtocol'; ...
              'createAdaptingEdgeProtocol';...
              'createFlickerBarProtocol';...
              'createMovingBarProtocol' ...
              };

for ii=1:length(avFuncCell)
    fprintf('%d. %s \n', ii, avFuncCell{ii})
end

while 1
    inp = input('Please provide the desired function number:');
    inpTest = zeros(1, length(inp));
    for ii=1:length(inp)  % allowing for more than one input
        if inp(ii) > 0 && inp(ii) < (length(avFuncCell)+1)
            inpTest(ii) = 1;
        end
        fprintf('\n')
    end
    if sum(inpTest) == length(inp)
        break
    else
        fprintf('Input %d is out of range, choose again \n', find(inpTest == 0, 1, 'first'))
    end
end

fprintf('selected %s \n', avFuncCell{inp})

funcHand = arrayfun(@(x) str2func(avFuncCell{x}), inp, 'uniformoutput', 0);

if length(funcHand) == 1 % if there is just one Function handle there is no need for it to be a cell
    funcHand = funcHand{1};
end



end