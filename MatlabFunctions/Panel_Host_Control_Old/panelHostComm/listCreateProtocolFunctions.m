function funcHand = listCreateProtocolFunctions

% function funcHand = listCreateProtocolFunctions
%
% This function lists the available createXXXProtocol functions and
% provides a function handle to the desired one. It uses the protocol
% directory and searches all the files ending with 'Protcol.m'

protocolsDir = 'E:\ExpCodeandRes\MatlabFunctions\Panel_Host_Control\stimulusProtocols';

dirRes = dir(fullfile(protocolsDir, '*Protocol.m'));

fprintf(' \n Available createXXXProtocol functions: \n')

avFuncCell = {dirRes.name};
avFuncCell = cellfun(@(x) x(1:end-2), avFuncCell, 'uniformoutput', 0);

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