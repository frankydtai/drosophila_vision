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

% Parameters for conversion
thresh = 0.75; % fraction of max filtered signal to consider as 1  
sampBetweenLEDPeaks = 20; % number of samples between PD peaks within an ON pulse
numberOfInt = 6; %number of intervals without signal to still consider consequetive


% getting rid of slow drifts
%filHP = fdesign.highpass('Fst,Fp,Ast,Ap', 0.01, 0.04, 60, 1);
%filHP = fdesign.bandpass('Fst1,Fp1,Fp2,Fst2,Ast1,Ap,Ast2',0.01,0.03,0.5,0.6,60,1,60);

Hd = fdatooLowPass;

filtData = filtfilt(Hd.Numerator, 1, pdData);

threshVal = findSecondHistPeak(filtData)*thresh;

threshData = filtData > threshVal;
convData = threshData;

for ii=1:length(threshData)-numberOfInt*sampBetweenLEDPeaks-1  
    if threshData(ii) == 1 && threshData(ii+1) == 0
        tempD = threshData(ii+1:ii+numberOfInt*sampBetweenLEDPeaks);
        tempInd = find(tempD, 1, 'first');
        convData(ii+1:ii+tempInd) = 1;
    end
end



end


%%

function secPeak = findSecondHistPeak(dat)

[bcount, bpos] = hist(dat, 100);
 step = 3;
 [~ , maxInd] = max(bcount); 
 
 secPeak = [];
 
 for ii=maxInd+10:length(bcount)-3
     tempC = bcount(ii-step:ii+step);
     if max(tempC) == bcount(ii)
         secPeak = bpos(ii);
         break
     end
 end

end


