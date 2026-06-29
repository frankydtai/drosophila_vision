function noiseCell = parseNoiseFromMinMotProtocl(pStruct)

% function parseNoiesFromMinMotProtocl(pStruct)
%
%


relCh = 3;
numSamp = 3000; % samples before and after stim
cMap = cbrewer('qual', 'Paired', 2);


assert(isfield(pStruct, 'noiseStruct'), 'pStruct is missing noiseStruct field')

noiseInds = vertcat(pStruct.noiseStruct.stim(:).relInds);
noiseInds = noiseInds(:,1); % since only the first col changes in this stim

intF = pStruct.noiseStruct.intFrames;

numNoiseStim = length(unique(noiseInds));
noiseCell = cell(1, numNoiseStim);
stimSiz = size(pStruct.noiseStruct.stim(1).matCell);
expFrames = zeros([stimSiz([1, 2]), numNoiseStim]);


for ii=1:numNoiseStim
    
    tempNInds = find(noiseInds == ii);
    tempMat = zeros(2*numSamp+1, length(tempNInds));
    
    expFrames(:,:,ii) = pStruct.noiseStruct.stim(tempNInds(1)).matCell(:,:,1+intF);
    for jj=1:length(tempNInds)
        
        tempDat = pStruct.stim(tempNInds(jj)).data;
        posDat = tempDat{2};
        ephyDat = tempDat{1};
        relPosTimeS = posDat(end-intF-1, 1); % -1 since last frame is zero 
        datInd = find(ephyDat(:,1) - double(relPosTimeS) > 0, 1, 'first');
        tempMat(:, jj) = ephyDat(datInd-numSamp:datInd+numSamp, relCh) * 10;
    end
    noiseCell{ii} = tempMat;
    
end


totMax = max(cellfun(@max, cellfun(@max, noiseCell, 'uniformoutput', 0)));
totMin = min(cellfun(@min, cellfun(@min, noiseCell, 'uniformoutput', 0)));
buff = (totMax - totMin)/10;

fh = figure('units', 'normalized', 'position', [0.15, 0.45, 0.23, 0.45]);
axh1 = axes('position', [0.05, 0.3, 0.9, 0.65]);
axh2 = axes('position', [0.05, 0.05, 0.9, 0.2]);

for ii=1:length(noiseCell)
    
    plot(axh1, noiseCell{ii}, 'color', cMap(:, 1))
%     hold on
%     plot(mean(noiseCell{ii}, 2), 'color', cMap(:, 2), 'linewidth', 3)
%     hold off
    
    set(axh1, 'ylim', [totMin-buff, totMax+buff], 'xlim', [1, 2*numSamp])
    
    fh.CurrentAxes = axh2;
    imagesc(expFrames(:,:,ii), [0, 7])
    
    
    k = waitforbuttonpress;
  
end
    

        
        
end     