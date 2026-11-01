
addpath ../supportingFunctions/

load ./optTables.mat

relIterT5 = 345; % determined by sorting for smallest error res_mov_nc+res_mov_pc

% change the optTab to explore T4 cells or diagonal T5 optimization results
[~, relParamsT5] = extract_params2(cardT5OptTab(relIterT5, :)); 


close all

w2hFac = 0.95;
figW = 17.4; %174mm 

fh = figure('units', 'centimeters', 'position', [5, 5, figW, figW*w2hFac], 'color', [1,1,1]);


%% Single Position Flashes



xSt1 = 0.05;
xEnd1 = 0.95;
ySt1 = 0.75;
yEnd1 = 0.95; 
xDif1 = 0.005; 
yDif1 = 0.02; 


frameR = 1; 
modTimeVec = (0:frameR:4000)';
startInd = 100; 
spWid = 4; 
spSD = 160; 
spVal = [0,1];
spPosVec = -5:1:5;
pColDB = [0.1451    0.1451    0.1451; ...
          0.1922    0.6392    0.3294];
      
posCell1 = generatePositionCell(xSt1, xEnd1, ySt1, yEnd1, xDif1, -yDif1, length(spPosVec)); 
axh1 = gobjects(size(posCell1));      

xxRange = [0, 2000];
yyRange = [-10, 27.5];

for ii=1:length(spPosVec)
    
	axh1(ii) = axes('position', posCell1{ii});
	hold on 
    
    for jj=1:length(spVal)

        stimInds = false(size(modTimeVec)); 
        relInds = startInd;
        tmpInds = arrayfun(@(x) find(modTimeVec == x), relInds);
        stimInds(tmpInds) = 1; 
        % change to t4_simple_wrap if using T4 parameters
        vFlash = t5_simple_wrap(relParamsT5,'spfr',{spVal(jj)} ,{spWid}, {spSD}, {spPosVec(ii)},modTimeVec,frameR, {stimInds});
        
        plot(modTimeVec, vFlash, 'linewidth', 2, 'color', pColDB(jj,:))
        
        axh1(ii).YLim = yyRange;
        axh1(ii).XLim = xxRange;
        axh1(ii).YColor = 'none'; 
        axh1(ii).XColor = 'none'; 
        
    end
    
    if ii==1 && jj==2
        line([100,100], [10,20], 'linewidth', 2, 'color', 'k')
        line([100,350], [10,10], 'linewidth', 2, 'color', 'k')
        text(200, 15, 'dark', 'color', pColDB(1,:), 'fontweight', 'bold')
        text(200, 20, 'bright', 'color', pColDB(2,:), 'fontweight', 'bold')
    end
    
    hold off
    
end



%% moving bar

yShift = 0.1;

xSt2 = xSt1;
xEnd2 = 0.45;
ySt2 = 0.35;
yEnd2 = ySt1-yShift; 
xDif2 = 0.035; 
yDif2 = 0.02;

frameR = 1; 
modTimeVec = (0:frameR:4000)';
startInd = 100; 
movWid = 2; 
movSD = 160; 
movVal = [0,1];
movPosVec = 4:-1:-4;

posCell2 = generatePositionCell(xSt2, xEnd2, ySt2, yEnd2, xDif2, -yDif2, 2); 
axh2 = gobjects(size(posCell2));  

pCol = [0.9843    0.6039    0.6000; ...
        0.8902    0.1020    0.1098; ...
        0.6510    0.8078    0.8902; ...
        0.1216    0.4706    0.7059];

xxRange2 = [0, 4000];
yyRange2 = [-10, 30];
    
    
for ii=1:length(movVal)

    stimInds = false(size(modTimeVec)); 
    relInds = (1:length(movPosVec)) * movSD + startInd;
    tmpInds = arrayfun(@(x) find(modTimeVec == x), relInds);
    stimInds(tmpInds) = 1; 
    % change to t4_simple_wrap if using T4 parameters
    vND = t5_simple_wrap(relParamsT5,'mov',{movVal(ii)} ,{movWid}, {movSD}, {movPosVec},modTimeVec,frameR, {stimInds});
    vPD = t5_simple_wrap(relParamsT5,'mov',{movVal(ii)} ,{movWid}, {movSD}, {fliplr(movPosVec)},modTimeVec,frameR, {stimInds});
    
    axh2(ii) = axes('position', posCell2{ii});
    hold on 
    plot(modTimeVec, vND, 'linewidth', 2, 'color', pCol(4,:))
    plot(modTimeVec, vPD, 'linewidth', 2, 'color', pCol(2,:))
    hold off
    
    axh2(ii).YLim = yyRange2;
    axh2(ii).XLim = xxRange2;
    axh2(ii).YColor = 'none'; 
    axh2(ii).XColor = 'none'; 
    
    if movVal(ii) == 0
        title(['\color[rgb]{0.1451, 0.1451, 0.1451}', 'Dark mov bar'])
    elseif movVal(ii) ==1
        title(['\color[rgb]{0.1922, 0.6392, 0.3294}', 'Bright mov bar'])
    end
          
    
    if ii == 1 
        line([0,250], [10,10], 'linewidth', 2, 'color', 'k')
        line([0,0], [10,20], 'linewidth', 2, 'color', 'k')
        legend('ND', 'PD', 'box', 'off')
    end


    

end



%% moving edge

xEnd3 = xEnd1;
xSt3 = xEnd1 - (xEnd2 - xSt2);
ySt3 = ySt2;
yEnd3 = yEnd2; 
xDif3 = xDif2; 
yDif3 = 0.02;

posCell3 = generatePositionCell(xSt3, xEnd3, ySt3, yEnd3, xDif3, -yDif3, 2); 
axh3 = gobjects(size(posCell3));  


frameR = 1; 
modTimeVec = (0:frameR:4000)';
startInd = 100; 
egWid = 1; 
egSD = 160; 
egVal = [0,1];
egPosVec = 4:-1:-4;

    
for ii=1:length(egVal)

    stimInds = false(size(modTimeVec)); 
    relInds = (1:length(egPosVec)) * egSD + startInd;
    tmpInds = arrayfun(@(x) find(modTimeVec == x), relInds);
    stimInds(tmpInds) = 1; 
    % change to t4_simple_wrap if using T4 parameters
    vND = t5_simple_wrap(relParamsT5,'edge',{egVal(ii)} ,{egWid}, {egSD}, {egPosVec},modTimeVec,frameR, {stimInds});
    vPD = t5_simple_wrap(relParamsT5,'edge',{egVal(ii)} ,{egWid}, {egSD}, {fliplr(egPosVec)},modTimeVec,frameR, {stimInds});
    
    axh3(ii) = axes('position', posCell3{ii});
    hold on 
    plot(modTimeVec, vND, 'linewidth', 2, 'color', pCol(4,:))
    plot(modTimeVec, vPD, 'linewidth', 2, 'color', pCol(2,:))
    hold off
    
    axh3(ii).YLim = yyRange2;
    axh3(ii).XLim = xxRange2;
    axh3(ii).YColor = 'none'; 
    axh3(ii).XColor = 'none'; 
    
    if egVal(ii) == 0
        title(['\color[rgb]{0.1451, 0.1451, 0.1451}', 'Dark edge'])
    elseif egVal(ii) ==1
        title(['\color[rgb]{0.1922, 0.6392, 0.3294}', 'Bright edge'])
    end
    
    
    if ii == 1 
        line([0,250], [10,10], 'linewidth', 2, 'color', 'k')
        line([0,0], [10,20], 'linewidth', 2, 'color', 'k')
    end


    

end


%% minimal Reverse-Phi

xEnd4 = xEnd1;
xSt4 = xSt1;
ySt4 = 0.05;
yEnd4 = ySt2 - yShift; 
xDif4 = 0.02; 
yDif4 = 0.02;

mmPosVec = -5:1; 

posCell4 = generatePositionCell(xSt4, xEnd4, ySt4, yEnd4, xDif4, -yDif4, length(mmPosVec)); 
axh4 = gobjects(size(posCell4));  


mmVal = {1,0};

mmWid = {4,4};
mmDur = {160,160};

frameR = 1; 
timeVec = (0:frameR:4000)';
startInd = 100; 


stimInds1 = false(size(timeVec)); 
stimInds2 = stimInds1;

relInds = (1:length(mmVal)) * mmDur{1} + startInd;
tmpInds = arrayfun(@(x) find(timeVec == x), relInds);
stimInds1(tmpInds(1)) = 1; 
stimInds2(tmpInds(2)) = 1; 

xxRange4 = [0, 2000];
yyRange4 = [-5, 27.5];


for ii=1:length(mmPosVec)
    
    relPosP = {mmPosVec(ii), mmPosVec(ii) + mmWid{1}};
    relPosN = fliplr(relPosP);
    % change to t4_simple_wrap if using T4 parameters
    vP = t5_simple_wrap(relParamsT5, 'mm', mmVal, mmWid, mmDur, relPosP ,timeVec ,frameR , {stimInds1, stimInds2});
    vN = t5_simple_wrap(relParamsT5, 'mm', mmVal, mmWid, mmDur, relPosN ,timeVec ,frameR , {stimInds1, stimInds2});
    
    
    axh4(ii) = axes('position', posCell4{ii});
    hold on 
    plot(timeVec, vP, 'color', pCol(2,:), 'linewidth', 2)
    plot(timeVec, vN, 'color', pCol(4,:), 'linewidth', 2)
    
    titC1 = num2str([relPosP{:}]);
    titC2 = num2str([relPosN{:}]);
    
    title({['\color[rgb]{0.8902, 0.1020, 0.1098}', titC1]; ... 
           ['\color[rgb]{0.1216, 0.4706, 0.7059}', titC2]});    
    
    axh4(ii).YLim = yyRange4;
    axh4(ii).XLim = xxRange4;
    axh4(ii).YColor = 'none'; 
    axh4(ii).XColor = 'none'; 
    
    if ii == 1 
        line([0,250], [10,10], 'linewidth', 2, 'color', 'k')
        line([0,0], [10,20], 'linewidth', 2, 'color', 'k')
    end
    
    hold off
    
end

