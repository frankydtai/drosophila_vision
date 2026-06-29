function varargout = plotMinMotExtMaxRatioMatrices(pStruct)

% function plotMinMotExtMaxRatioMatrices(pStruct)
%
% This function is desinged to plot the ration between the peak responses
% to the second bar in the excitatory minmot protocol. 

% fill on later


numCol = 11;

minMotSt = calculateMinMotExtLinSumDiff(pStruct);

datSiz = size(minMotSt);

allMaxRatios = zeros(datSiz);


% orginizing the data

for ii=1:datSiz(1)
    
    for jj=1:datSiz(2)
        
        for kk=1:datSiz(3)
            
            if ~isempty(minMotSt(ii,jj,kk).linDiff)
                
                allMaxRatios(ii,jj,kk) = log(minMotSt(ii,jj,kk).linDiff(2).maxResp(2) ./ minMotSt(ii,jj,kk).linDiff(2).maxLinResp(2)); %just second bar for now
                
            end
            
        end
        
    end
    
end
                

figure
posCell = generatePositionCell(0.15, 0.975, 0.05, 0.975, 0.02, -1, datSiz(3));
axh = zeros(1,datSiz(3)+1);

for kk=1:datSiz(3)
   
    axh(kk) = axes('position', posCell{kk});
    imagesc(allMaxRatios(:,:,kk), [-1, 1])
end

colormap(flipud(cbrewer('div', 'RdBu', numCol)));

set(axh(1:end-1), 'xticklabel', {}, 'yticklabel', {});

axh(end) = axes('position', [0.05, 0.05, 0.01, 0.925]);
imagesc(fliplr(linspace(-1,1,numCol))')

yLab = arrayfun(@num2str, ((1:2:numCol) - ceil(numCol/2))/floor(numCol/2), 'uniformoutput', 0);

set(axh(end), 'xtick', [], 'ytick', 1:2:numCol, 'yticklabel', yLab);

if nargin == 1
    varargout{1} = axh; 
end



end