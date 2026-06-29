function ledExpData = findZeroBlueLEDvsRest(constMaxMatrix, numSteps)

% function ledExpData = findZeroBlueLEDvsRest(constMaxMatrix)
%
% This function uses the values in constMaxMatrix to generate alternating
% pulses of blue vs other LEDs to find the point of equal response (zero
% difference). Each light that is measured vs blue is measured twice: once
% as changing from zero to max while blue is held constant and once while
% being held constant and blue changing from zero to its max. 
%
% INPUT
% constMaxMatrix - 4X3 matrix with values desgnating constant and maximal
%                  values for each comparison (between 0 and 10). Values should be entered
%                  this way   [const Blue, max Blue, const UV, max UV;
%                              const Blue, max Blue, const Green , max Green;
%                              const Blue, max Blue, const Amber,  max Amber]
% numSteps - (optional) number of steps between 0 and max. If not given
%            default value is 11
%
% OUTPUT
%
% ledExpData -  NX10 matrix of current response to LED stim. 1-3 are UV
%               alone, UV ramp vs stable blue, and stable UV vs blue ramp. 4-6 are for
%               Green, and 7-9 are for Amber. 10 is blue alone. 
% ledComm -     analogoutput signal for the corresponding experiments (on both
%               channels.



if nargin < 2
    numSteps = 11;
end
sampRate = 10000;
stepDur = 1000; % in samples (100ms)
numReps = 5; % number of times to repeat the whole step series



if max(constMaxMatrix(:)) > 10
    error('Maximal value should not exceed 10V')
elseif min(constMaxMatrix(:))<= 0 
    error('Minimal value should be higher than 0V')
end

checkMax = sum(diff(constMaxMatrix(:,1:2), 1,2) < 0) + ...
           sum(diff(constMaxMatrix(:,3:4), 1,2) < 0);

if checkMax > 0 
    error('Max value is smaller than constant');
end



rampVals = linspace(0, 1, numSteps); 
rampSig = repmat(rampVals, stepDur, 1);

zeroSig = zeros(size(rampSig));
anaSigRamp = vertcat(repmat(reshape(vertcat(rampSig, zeroSig), [],1), numReps, 1), zeros(stepDur*2, 1));
anaSigConst = circshift(anaSigRamp > 0, stepDur);
anaSigZero = zeros(size(anaSigRamp));
%anaSig2ch = [anaSigRamp, circshift(anaSigConst, stepDur)];


ledExpData = zeros(length(anaSigRamp), 10);
ledComm = zeros([size(ledExpData), 2]);
counter = 0;

% Setting acquisition session 
daqreset
ses = daq.createSession('ni');
ses.addAnalogInputChannel('Dev1',5,'Voltage'); % record just current (5 is photodiode)
ses.addAnalogOutputChannel('Dev1',1:2,'Voltage');

ses.Channels(1).InputType = 'SingleEnded';

ses.Rate = sampRate;

relStrings = {'A', 'B', 'C'};

for ii=1:3
    relFac = constMaxMatrix(ii, :);
    counter = counter+1;
    ses.queueOutputData([anaSigZero, anaSigRamp*relFac(4)]); % first get the color alone
    f = figure('position', [  800   500   275   160]);
    figCol = get(gcf, 'color');
    text(0.075,  0.75, {['Change knob to position ', relStrings{ii}]; 'and press OK'})
    set(gca, 'color', figCol, 'xcolor', figCol, 'ycolor', figCol)
    h = uicontrol('Position',[20 20 200 40],'String','OK',...
                  'Callback','uiresume(gcbf)');
    uiwait(gcf); 
    close(f);
    ledExpData(:,counter) = ses.startForeground();
    ledComm(:,counter, :) = [anaSigZero, anaSigRamp*relFac(4)]; 
    
    counter=counter+1;
    ses.queueOutputData([anaSigConst*relFac(1), anaSigRamp*relFac(4)]); % color vs. const blue
    ledExpData(:,counter) = ses.startForeground();
    ledComm(:,counter, :) = [anaSigConst*relFac(1), anaSigRamp*relFac(4)];
    
    counter=counter+1;
    ses.queueOutputData([anaSigRamp*relFac(2), anaSigConst*relFac(3)]); % blue vs. const color
    ledExpData(:,counter) = ses.startForeground();
    ledComm(:,counter, :) = [anaSigRamp*relFac(2), anaSigConst*relFac(3)];
end

maxBlue = max(constMaxMatrix(:,2));
counter = counter+1;
ses.queueOutputData([anaSigRamp*maxBlue, anaSigZero]); % blue alone
ledExpData(:,counter) = ses.startForeground();
ledComm(:,counter, :) = [anaSigRamp*maxBlue, anaSigZero];

% Saving the data
timeStamp = arrayfun(@(x) ['_', num2str(fix(x))], clock, 'uniformoutput', 0);
timeStamp = [timeStamp{:}];
direc = pwd;
save(fullfile(direc, ['ledRampExp', timeStamp,]), 'ledExpData', 'ledComm')



end