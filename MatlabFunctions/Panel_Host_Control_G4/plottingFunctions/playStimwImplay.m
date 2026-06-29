function varargout = playStimwImplay(stimSeq)

% this function uses implay and sets some defaults automatically


pHand = implay(stimSeq, 100);

pHand.Visual.Axes.Position = [1 70 410 136];

maxVal = 15;
cmap = zeros(maxVal,3);
cmap(:,2) = linspace(0,1,maxVal);

pHand.Visual.ColorMap.Map = cmap;

pHand.Visual.ColorMap.UserRange = 1;
pHand.Visual.ColorMap.UserRangeMax = maxVal;


if nargout == 1
    
    varargout{1} = pHand;
    
end

end

