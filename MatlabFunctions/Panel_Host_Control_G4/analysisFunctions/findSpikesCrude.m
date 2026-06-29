function spikeInds = findSpikesCrude(vVec)

% function spikeInds = findSpikesCrude(vVect)
% 
% This function is designed to find spikes fast for online plotting
% purposes. Uses both voltage and derivative to determine spikes
%
% output is a logical index vector for spikes

spikeSizT = 5; % threshold for spikes above baseline in mV 
spikeInds = zeros(size(vVec));

vecLen = length(vVec); 
medWinSiz = ceil(vecLen/10000)*100; 


smVec = movmean(vVec, medWinSiz/10);
diffVec = [diff(smVec);0]; % padding it 
medVec = movmedian(smVec, medWinSiz); % gets rid of spikes to determine baseline
movMeanMedV = movmean(medVec,  medWinSiz/10);

diffT = nanmedian(abs(diffVec)) * 10; % so long as it is not spiking all the time


diffVInd = diffVec > diffT;
vmInd = vVec - movMeanMedV > spikeSizT;

if sum(diffVInd) > 0 && sum(vmInd) > 0 % only correlated if there is something to correlate

    [corr, lags] = xcorr(diffVInd, vmInd);
    [~, maxCI] = max(corr); 
    relLag = -lags(maxCI); % since the der precedes the voltage
    if relLag <5 || relLag > 70 % determined empirically
        warning('relLag out of bounds set to 40')
        relLag = 40; 
    end


    diffVInd2 = [zeros(relLag, 1); diffVInd(1:end-relLag)];

    [spikeInd, indVal] = SplitVec(vmInd & diffVInd2, 'equal', 'first', 'firstval');


    relSpInd = spikeInd(indVal==1);

    spikeInds(relSpInd) = 1; 
    
end

end
