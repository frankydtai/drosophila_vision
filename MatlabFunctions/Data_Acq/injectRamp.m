function rampData = injectRamp(maxCurr, reps, injectRate)

% function data = injectRamp(maxCurr, reps, injectRate)
%
% This function injects a current ramp and records the responses in both
% current and voltage traces. The function assumes a 400pA/V conversion 
% ratio (defined in the amplifier gains) and maintains a constant dC/dt of 20pA/sec  
%
% INPUT
% maxCurr - maxCurr to be injected in pA
% reps -    (integer), number of times to repeat the injection
% injectRate -   (optional) default is 20pA per second. Should be entered in pA per second 
%
% OUTPUT
% data -    a NX2 matrix with voltage and current as both channels. 
%           Sampling rate is 10K and N is determined by both maxCurr (due to the preset rate)
%           and the number of repeats. 
% Data is atomatically saved in the working directory. 



% Defining core variables
sampRate = 10000;
interRampInt = 10000; % 1 sec between ramps
direc = pwd;


if nargin < 3
    injectRate = 20/1; % 20pA per second
end

conversionRatio = 400/1; % 400pA/V
maxVol = maxCurr/conversionRatio;

rampLength = maxVol/(injectRate/conversionRatio) * sampRate;

oneRamp = [linspace(0, maxVol, rampLength), zeros(1, interRampInt)];

anaSig = vertcat(zeros(interRampInt, 1), repmat(oneRamp, 1, reps)');


% Setting acquisition session 

ses = daq.createSession('ni');
ses.addAnalogInputChannel('Dev1',0:1,'Voltage');
ses.addAnalogOutputChannel('Dev1',0,'Voltage');
%ses.DurationInSeconds = length(anaSig)/sampRate; 
ses.Channels(1).InputType = 'SingleEnded';
ses.Channels(2).InputType = 'SingleEnded';
ses.Rate = sampRate;

ses.queueOutputData(anaSig);

rampData = ses.startForeground();

% Saving the data
timeStamp = arrayfun(@(x) ['_', num2str(fix(x))], clock, 'uniformoutput', 0);
timeStamp = [timeStamp{:}];
save(fullfile(direc, ['rampData', timeStamp,]), 'rampData')



end
