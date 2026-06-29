function ledData = recordLEDresp()

% function ledData = recordLEDresp()
%
% This function injects the appropriate voltage to control the 4 channel LED. 
% it uses a predified signal the combines the channels in all combinations.
% core parameters can be defined at the beginning of the function
% OUTPUT
% ledData -    a NX2 matrix with voltage and current as both channels. 
%           Sampling rate is 10K and N is determined by both maxCurr (due to the preset rate)
%           and the number of repeats. 
% Data is atomatically saved in the working directory. 


direc = pwd;

% Defining core variables
sampRate = 10000;
interRampInt = 2500; % 0.25 sec between ramps

% values should not exceed 10mV
maxGreen = 5;
maxBlue = 10;
maxUV = 10;
numSteps = 10; % number of steps within each ramp
stepDur = 1000; % in samples (100ms)
constFactor = 0.5; %factor with which to determine the constant value presented with the other color

rampVals = linspace(0, 1, numSteps); 
baseSig = reshape(repmat(rampVals, stepDur, 1), [], 1);
baseLen = length(baseSig);

intSig = zeros(interRampInt, 1);
constSig = ones(baseLen,1)*constFactor;

greenSig = vertcat(intSig, ...
                   baseSig*maxGreen, intSig, constSig*0, intSig, constSig*0, intSig,... % each color presented separetly
                   baseSig*maxGreen, intSig, baseSig*maxGreen, intSig,... % green with the 2 others as constants
                   constSig*maxGreen, intSig, constSig*0, intSig, ... % green constant with UV
                   constSig*maxGreen, intSig, constSig*0, intSig,... % green constant with Blue
                   baseSig*maxGreen, intSig); % all together
                   
UVSig = vertcat(intSig, ...
                   constSig*0, intSig, baseSig*maxUV, intSig, constSig*0, intSig,...
                   constSig*maxUV, intSig, constSig*0, intSig,... 
                   baseSig*maxUV, intSig, baseSig*maxUV, intSig, ... 
                   constSig*0, intSig, constSig*maxUV, intSig, ...
                   baseSig*maxUV, intSig);
               
blueSig = vertcat(intSig, ...
                   constSig*0, intSig, constSig*0, intSig, baseSig*maxBlue, intSig,...
                   constSig*0, intSig, constSig*maxBlue, intSig,... 
                   constSig*0, intSig, constSig*maxBlue, intSig, ... 
                   baseSig*maxBlue, intSig, baseSig*maxBlue, intSig, ...
                   baseSig*maxBlue, intSig);

GUBSig = horzcat(greenSig, UVSig, blueSig);


% Setting acquisition session 

ses = daq.createSession('ni');
ses.addAnalogInputChannel('Dev1',0:1,'Voltage');
ses.addAnalogOutputChannel('Dev1',1:3,'Voltage');
%ses.DurationInSeconds = length(anaSig)/sampRate; 
ses.Channels(1).InputType = 'SingleEnded';
ses.Channels(2).InputType = 'SingleEnded';
ses.Rate = sampRate;

ses.queueOutputData(GUBSig);

ledData = ses.startForeground();

% Saving the data
timeStamp = arrayfun(@(x) ['_', num2str(fix(x))], clock, 'uniformoutput', 0);
timeStamp = [timeStamp{:}];
save(fullfile(direc, ['ledExp', timeStamp,]), 'ledData', 'GUBSig')



end
