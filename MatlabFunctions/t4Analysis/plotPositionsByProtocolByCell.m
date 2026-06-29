function varargout = plotPositionsByProtocolByCell(t4cellStruct, allSigBarTab)

% function plotPositionsByProtocolByCell(t4cellStruct)
%
% This function is designed to plot all the positions that were used in
% different protocols for the same cell on a single axis. minMaxStruct is
% used to overlay minMax data on top of singlebar resutls.
%
% INPUT
%
% t4cellStruct -            generated manually in
%                           assigningProtocols2CellsScript
%
% allSigBarTab -            generated using extractFitParameterFromSingleBarFit

% Axh is mixed up for no reason 
 
maxMinCell = getNormMinMaxSingleBarResp(allSigBarTab);

allCellNames = unique(allSigBarTab.cellName, 'rows');

cellSubStruct = struct;
numCells = length(t4cellStruct);

% orginizing the data minmax results, positive positions to be plotted
% above minmax (flicker and minmot) and negative positions (to be plotted
% below minmax results

for ii=1:numCells
    
    tempSt = t4cellStruct(ii);
    accumPos = {};
    minMovPos = {};
    counter = 0;
    secCount = 0;
    for jj=1:length(tempSt.flicker)
        counter = counter+1;
        accumPos{counter} = {['flc', num2str(jj)], tempSt.flicker(jj).pos};
    end
    
    for jj=1:length(tempSt.minMot)
        counter = counter+1;
        accumPos{counter} = {['mot', num2str(jj), 'FB', tempSt.minMot(jj).barV(1)], tempSt.minMot(jj).fbPos};
        counter = counter+1;
        accumPos{counter} = {['mot', num2str(jj), 'SB', tempSt.minMot(jj).barV(2)], tempSt.minMot(jj).sbPos};
    end
    
    for jj=1:length(tempSt.minMov)
        for kk=1:size(tempSt.minMov(jj).pos,1)
            secCount = secCount+1;
            minMovPos{secCount} = {['mov', num2str(jj)], tempSt.minMov(jj).pos(kk,:)};
        end
    end
    
    cellSubStruct(ii).maxMin = maxMinCell{ii};
    cellSubStruct(ii).posPos = accumPos;
    cellSubStruct(ii).negPos = minMovPos;
    
end


% plotting the data

posCellMaxMin = generatePositionCell(0.1, 0.975, 0.45, 0.55, -1, 0, 2);
colMapR = cbrewer('seq', 'Reds', 9);
colMapB = cbrewer('seq', 'Blues', 9);
linCols = cbrewer('qual', 'Set1', 4);
markSiz = 10;
axh = zeros(numCells, 4);

for ii=1:numCells
    
    figure('units', 'pixels', 'position', [1040, 604, 530, 645], ...
            'name', allCellNames(ii,:), 'numbertitle', 'off')
    
    relBound = floor(size(cellSubStruct(ii).maxMin,1)/2);
    
    
    % plotting max and min resp
    axh(ii,3) = axes('position', posCellMaxMin{1});
    imagesc([-relBound,relBound], [0,1], cellSubStruct(ii).maxMin(:,1)', [0,2])
    colormap(axh(ii,3), colMapR)

    axh(ii,2) = axes('position', posCellMaxMin{2});
    imagesc([-relBound,relBound], [0,1], cellSubStruct(ii).maxMin(:,2)', [0,2])
    colormap(axh(ii,2), colMapB)
    
    % plotting flicker and minMot
    axh(ii,1) = axes('position', [0.1, 0.55, 0.8750, 0.4]);
    hold on
    tempNumPos = length(cellSubStruct(ii).posPos);
    yTickLab = cell(1,tempNumPos);
    
    for jj=1:tempNumPos
        
        tempLab = cellSubStruct(ii).posPos{jj}{1};
        yTickLab{jj} = tempLab;
        
        if strcmp(tempLab(1:3), 'flc')
            linCol = 'k'; 
            markCol = [1,1,1]*0.8;
        elseif strcmp(tempLab(1:3), 'mot')
            if strcmp(tempLab(end-2), 'F')
                linCol = linCols(1,:);
            elseif strcmp(tempLab(end-2), 'S')
                linCol = linCols(2,:);
            end
            if strcmp(tempLab(end), 'B')
                markCol = 'w';
            else
                markCol = 'k';
            end
        end
        
        plot(cellSubStruct(ii).posPos{jj}{2}, ones(1,length(cellSubStruct(ii).posPos{jj}{2}))*jj, ...
             '-o', 'color', linCol, 'markerFaceColor', markCol, ...
             'markerEdgeColor', [1,1,1]*0.6, 'linewidth', 3, 'markerSize', markSiz)
    end
    
    title(allCellNames(ii,:))
    set(axh(ii,1), 'xlim', [-relBound, relBound], 'ylim', [0.5, jj+0.5], ...
        'ytick', 1:jj, 'yticklabel', yTickLab, 'xticklabel', {})
    hold off
    
    % plotting minMov
    axh(ii,4) = axes('position', [0.1, 0.05, 0.8750, 0.4]);
    hold on
    tempNumPos = length(cellSubStruct(ii).negPos);
    yTickLab = cell(1,tempNumPos);
    
    for jj=1:tempNumPos
        
        tempLab = cellSubStruct(ii).negPos{jj}{1};
        yTickLab{jj} = tempLab;
        
        linCol = [1,1,1]*0.8;
        
        plot(cellSubStruct(ii).negPos{jj}{2}, ones(1,length(cellSubStruct(ii).negPos{jj}{2}))*jj, ...
             '-o', 'color', linCol, 'markerFaceColor', 'r', ...
             'markerEdgeColor', [1,1,1]*0.6, 'linewidth', 3, 'markerSize', markSiz)
    end
    
    if isempty(jj) % in case negPos is empty
        newJJ = 2;
    else
        newJJ = jj;
    end
    set(axh(ii,4), 'xlim', [-relBound, relBound], 'ylim', [0.5, newJJ+0.5], 'ytick', 1:newJJ, 'yticklabel', yTickLab)
    hold off
    
    set(axh(ii,2:3), 'ytick', [])
    set(axh(ii,3), 'xtick', [])
    
    print2PDF(['./T4recordingSummaryAndAnalysis/assignedProtocolPlots/', allCellNames(ii,:)])
    
    

end


if nargout ==1
    varargout{1} = axh;
end


end



