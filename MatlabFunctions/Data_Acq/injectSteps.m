function stepData = injectSteps(maxCurr, numSteps, reps)

% function data = injectSteps(maxCurr, reps, injectRate)
%
% This function injects a 500ms current steps and records the responses in both
% current and voltage traces. The function assumes a 400pA/V conversion 
% ratio (defined in the amplifier gains)   
%
% INPUT
% maxCurr - maxCurr to be injected in pA
% numSteps - number fo steps between maxCurr and rest
% reps -    (optional), number of times to repeat the injection (defualt 5)
% 
%
% OUTPUT
% stepData -    a NX2 matrix with voltage and current as both channels. 
%           Sampling rate is 10K and N is determined by both maxCurr (due to the preset rate)
%           and the number of repeats. 
% Data is atomatically saved in the working directory. 

% due to syn clock problems with the NIDAQ board
daq.reset
daq.HardwareInfo.getInstance('DisableReferenceClockSynchronization',true);

% Defining core variables
sampRate = 10000;
interStepInt = 5000; % 0.5 sec between steps
direc = pwd;


if nargin < 3
    reps = 5;
end

conversionRatio = 400/1; % 400pA/V
maxVol = maxCurr/conversionRatio;

stepLength = 5000; % 500ms
stepVals = linspace(0, maxVol, numSteps+1);
stepVals = stepVals(2:end); %gets rid of zero

valInds = repmat(1:numSteps, 1, reps);
permInds = randperm(length(valInds));
shufInds = valInds(permInds);

oneStep = [ones(1, stepLength), zeros(1, interStepInt)];

sigMat = zeros(length(oneStep), numSteps*reps);

for ii=1:length(shufInds)
    sigMat(:,ii) = oneStep*stepVals(shufInds(ii));
end


anaSig = vertcat(zeros(interStepInt, 1), reshape(sigMat, [], 1));


% Setting acquisition session 

ses = daq.createSession('ni');
ses.addAnalogInputChannel('Dev1',0:1,'Voltage');
ses.addAnalogOutputChannel('Dev1',0,'Voltage');
%ses.DurationInSeconds = length(anaSig)/sampRate; 
ses.Channels(1).InputType = 'SingleEnded';
ses.Channels(2).InputType = 'SingleEnded';
ses.Rate = sampRate;

ses.queueOutputData(anaSig);

stepData = ses.startForeground();

% Saving the data
timeStamp = datestr(clock, 'yyyymmdd_HH-MM-SS');
save(fullfile(direc, ['stepData', timeStamp,]), 'stepData')
disp(['Saved: ', timeStamp]);

% Plotting data
if numSteps > 6
    subpar = [2, ceil(numSteps/2)];
else
    subpar = [1, numSteps];
end

figure;
set(gcf, 'position', [180, 700, 1600, 300])
axh = zeros(1, numSteps);
for ii=1:numSteps
    tempInds = find(anaSig == stepVals(ii));
    spInds = SplitVec(tempInds, 'consecutive');
    spInds = cellfun(@(x) [x', x(end) + (1:250)], spInds, 'uniformoutput', 0);
    axh(ii) = subplot(subpar(1), subpar(2), ii);
    hold on
    cellfun(@(x) plot(1:length(x), stepData(x, 2), 'b'), spInds)
    hold off
    title(num2str(stepVals(ii)))
end

yylim = get(axh(:), 'ylim');
yylim = vertcat(yylim{:});
yymax = max(yylim(:,2));
yymin = min(yylim(:,1));

set(axh(:), 'ylim', [yymin, yymax])



end
