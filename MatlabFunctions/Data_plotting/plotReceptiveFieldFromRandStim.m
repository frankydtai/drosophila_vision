function varargout = plotReceptiveFieldFromRandStim(expRes, expPattern, relFreq)

% function plotReceptiveFieldFromRandStim(expRes, expPattern, relFreq)
% This function takes the data from random stimulus presentation and tries
% to construct a receptive field based on linearity assumption (weights
% each frame by the response and multiplies it). 
%
% INPUT
% expRes -      experimental results (should include current and x and y
%               positions)
% expPattern -  pattern structure that was used in the experiment (should
%               contain all the relevant fields in a regular pattern structure)
% relFreq -     3 element vector that includes the relevant frequencies. (1) sampling frequency 
%               (2) x freq and (3) y freq 



currData = expRes(:,2);

convXpos = findPositionFromVoltage(expRes(:, 3), relFreq(1:2), expPattern.x_num);
convYpos = findPositionFromVoltage(expRes(:, 4), relFreq([1, 3]), expPattern.y_num);

combConvs = convXpos + convYpos*1000; % to split the vector based on both X and Y
[spCombBrack, spCombVal] = SplitVec(combConvs, 'equal', 'bracket', 'firstval');
% to get rid of the segments before and after stimulus presentation
spCombBrack = spCombBrack(2:(end-2), :);
spCombVal = spCombVal(2:(end-2));


%relStep = relFreq(1)/1000; % to make 1ms steps
medBrackSize = median(spCombBrack(:,2)-spCombBrack(:, 1));

% calculating response vector
resVector = zeros(size(spCombBrack, 1), 1);
for ii=1:size(spCombBrack, 1)
    %baseInds = (spCombBrack(ii, 1) - 5*medBrackSize):(spCombBrack(ii, 1)-1); % takes 5 lengths of a stim as a baseline
    %tempBase = median(currData(baseInds));
    %tempResp = quantile(currData(spCombBrack(ii,1):spCombBrack(ii,2)), 0.95); % Doesn't take the max as resp
    %resVector(ii) = tempResp - tempBase;
    
    resVector(ii) = mean(currData(spCombBrack(ii,1):spCombBrack(ii,2)));

end

resVector = resVector/length(resVector); %normalizing to 1


% generating pattern matrix

matInds = [mod(spCombVal, 1000), floor(spCombVal/1000)]; %seperating x and y indices
allPats = expPattern.Pats;
patSiz = size(allPats);

stimByTimeMat = zeros(patSiz(1)*patSiz(2), size(matInds,1));

for ii=1:size(matInds,1)
    tempPat = allPats(:,:,matInds(ii, 1), matInds(ii,2));
    stimByTimeMat(:,ii) = sign(tempPat(:)-3); % since 7 is max, 3 is mid and 0 is min
end


% plotting results

posCell = generatePositionCell(0.05, 0.95, 0.05, 0.95, 0.02, 0.02, [5, 2]);
filt = fspecial('gaussian');


figure;
axhand = zeros(1, 10);
recTest = zeros(patSiz(1)*patSiz(2), 10);
for ii=1:10
    recpField = stimByTimeMat * circshift(resVector, (ii-1)*10);
    %filtRF = filter2(filt, reshape(recpField, patSiz(1), patSiz(2)));
    
    recTest(:, ii) = recpField;
    axhand(ii) = axes('position', posCell{ii});
    imagesc(reshape(recpField, patSiz(1), patSiz(2))) %, [-1 1]);
    
end


if nargout
    varargout{1} = axhand;
end



end