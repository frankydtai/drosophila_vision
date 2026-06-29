function ledData = recordBaseLEDresp(numSteps)

% function ledData = recordBaseLEDresp(numSteps)
%
% This function injects voltage pulses from 0-10V of length stepDur 
% (defined within the function) with numSteps. To avoid adptation steps are
% interspersed with 0 vol. 
% Function assumes the use of a manual switch for the 385 (A) 530 (B) and
% 590 (C) channels (connected to AO2). The 435 channel is connected directly to AO1
% OUTPUT
% ledData - a NX4 matrix with current (or PD) responses for each LED. 
%           Sampling rate is 10K and N is total number os smaples. 
% Data is atomatically saved in the working directory. 


direc = pwd;

% Defining core variables
sampRate = 10000;



stepDur = 1000; % in samples (100ms)
numReps = 5; % number of times to repeat the whole step series

rampVals = linspace(0, 10, numSteps); 
baseSig = repmat(rampVals, stepDur, 1);
zeroSig = zeros(size(baseSig));
anaSig = repmat(reshape(vertcat(baseSig, zeroSig), [],1), numReps, 1);

anaSigBlue = [anaSig, zeros(size(anaSig))];
anaSigRest = [zeros(size(anaSig)), anaSig];

ledData = zeros(length(anaSig), 4);

% Setting acquisition session 

ses = daq.createSession('ni');
ses.addAnalogInputChannel('Dev1',1,'Voltage'); % record just current (5 is photodiode)
ses.addAnalogOutputChannel('Dev1',1:2,'Voltage');
%ses.DurationInSeconds = length(anaSig)/sampRate; 
ses.Channels(1).InputType = 'SingleEnded';
%ses.Channels(2).InputType = 'SingleEnded';
ses.Rate = sampRate;

relStrings = {'A', '', 'B', 'C'};

for ii=1:4
    if ii ~= 2 
        ses.queueOutputData(anaSigRest);
        f = figure('position', [  800   500   275   160]);
        figCol = get(gcf, 'color');
        text(0.075,  0.75, {['Change knob to position ', relStrings{ii}]; 'and press OK'})
        set(gca, 'color', figCol, 'xcolor', figCol, 'ycolor', figCol)
        h = uicontrol('Position',[20 20 200 40],'String','OK',...
                      'Callback','uiresume(gcbf)');
        uiwait(gcf); 
        close(f);
    else
        ses.queueOutputData(anaSigBlue);
    end
    
    ledData(:,ii) = ses.startForeground();
end

% Saving the data
timeStamp = arrayfun(@(x) ['_', num2str(fix(x))], clock, 'uniformoutput', 0);
timeStamp = [timeStamp{:}];
save(fullfile(direc, ['ledBaseExp', timeStamp,]), 'ledData', 'anaSig')


% plotting results
plotLEDbaseExp(ledData, anaSig)


end
