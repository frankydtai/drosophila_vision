function convData = convertPhotoDiodeData2Pulses(pdData)

% function convData = convertPhotoDiodeData2Pulses(pdData)
%
% This function takes the photodiode data and converts it into 0s and 1s
% using certain assumptions. Ths function is necessary since erading from
% LEDs are actually short pulses (within the ON pulse) and not constant values
% Assmes 100KHz sampling rate
%
% INPUT
% pdData -      1XN vector of photodiode values
%
% convData -    1XN vector of 0 (OFF) and 1 (ON) values


% getting rid of slow drifts
%filHP = fdesign.highpass('Fst,Fp,Ast,Ap', 0.01, 0.04, 60, 1);
%filHP = fdesign.bandpass('Fst1,Fp1,Fp2,Fst2,Ast1,Ap,Ast2',0.01,0.03,0.5,0.6,60,1,60);

Hd = fdatooLowPass;

filtData = filtfilt(Hd.Numerator, 1, pdData);
gFit = gmdistribution.fit(filtData',2);

threshVal = mean(gFit.mu);

threshData = filtData > threshVal;
convData = threshData;



end
