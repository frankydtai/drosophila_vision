%% singleBar table


stimDur = protocolStruct.gratingTable.stimDur;
appear = ones(size(stimDur)) * 20;
%appear = ones(size(stimDur)) * 12;
disappear = zeros(size(stimDur));

disFrame = [20, 21, 23, 27]+1;
%disFrame = [12, 13, 15, 19]+1;

uValStimDur = unique(stimDur);

for ii=1:length(uValStimDur)
    
    disappear(stimDur == uValStimDur(ii)) = disFrame(ii);
    
end

protocolStruct.gratingTable.appear = appear;
protocolStruct.gratingTable.disappear = disappear;


clear disFrame uValStimDur stimDur appear disappear ii

%% Minimal motion 1



fAppearInd = 20;
% fAppearInd = 12;
stepDurUnit = 0.02;
simFrame = protocolStruct.inputParams.stepDur / stepDurUnit;


relTab = protocolStruct.gratingTable;
span = abs(relTab.FBPos - relTab.SBPos);
frameFac = relTab.timeDiff/ stepDurUnit;

numStim = height(relTab);
barInds= zeros(numStim,4); % for appearance and disappearance of both bars

for ii=1:numStim
    
    if frameFac(ii) == 0
        fApp = 0;
        fDis = simFrame;
        sApp = 0;
        sDis = simFrame;
    else
        
        if span(ii) == 0
            fApp = frameFac(ii);
            sApp = fApp;
        else
            fApp = 0;    
            sApp = span(ii) * frameFac(ii);
        end
        
        fDis = fApp + frameFac(ii);
        
        sDis = sApp + frameFac(ii);
        
    end
    
    
   
    barInds(ii, :)  = [fApp, fDis, sApp, sDis] + fAppearInd;
end
    
protocolStruct.gratingTable.fAppear = barInds(:,1);
protocolStruct.gratingTable.fDisappear = barInds(:,2);
protocolStruct.gratingTable.sAppear = barInds(:,3);
protocolStruct.gratingTable.sDisappear = barInds(:,4);

clear barInds *Dis *App numStim spDiff relTab fAppearInd stepDurUnit simFrame ii frameFac



%% Minimal motion non speed corrected (dark)



fAppearInd = 20;
% fAppearInd = 12;
stepDurUnit = 0.02;
simFrame = protocolStruct.inputParams.stepDur / stepDurUnit;


relTab = protocolStruct.gratingTable;
span = abs(relTab.FBPos - relTab.SBPos);
frameFac = relTab.timeDiff/ stepDurUnit;

numStim = height(relTab);
barInds= zeros(numStim,4); % for appearance and disappearance of both bars

for ii=1:numStim
    
    if frameFac(ii) == 0
        fApp = 0;
        fDis = simFrame;
        sApp = 0;
        sDis = simFrame;
    else
        
        
        fApp = 0;
        sApp = 2 * frameFac(ii);
        
        fDis = fApp + frameFac(ii);
        
        sDis = sApp + frameFac(ii);
        
    end
    
    
   
    barInds(ii, :)  = [fApp, fDis, sApp, sDis] + fAppearInd;
end
    
protocolStruct.gratingTable.fAppear = barInds(:,1);
protocolStruct.gratingTable.fDisappear = barInds(:,2);
protocolStruct.gratingTable.sAppear = barInds(:,3);
protocolStruct.gratingTable.sDisappear = barInds(:,4);

clear barInds *Dis *App numStim spDiff relTab fAppearInd stepDurUnit simFrame ii frameFac



%%  minMot inhibition



fAppearInd = 20;
% fAppearInd = 12;
stepDurUnit = 0.02;
simFrame = protocolStruct.inputParams.stepDur / stepDurUnit;


relTab = protocolStruct.gratingTable;
span = abs(relTab.FBPos - relTab.SBPos);
frameFac = relTab.timeDiff/ stepDurUnit;
fbStat = relTab.FBStat;

numStim = height(relTab);
barInds= zeros(numStim,4); % for appearance and disappearance of both bars

for ii=1:numStim
    
    if frameFac(ii) == 0
        fApp = 0;
        fDis = simFrame;
        sApp = 0;
        sDis = simFrame;
    else
        
        if span(ii) == 0
            fApp = frameFac(ii);
            sApp = fApp;
        else
            fApp = 0;    
            sApp = span(ii) * frameFac(ii);
        end
        
        if fbStat(ii) == 0
            fDis = fApp + frameFac(ii);
        elseif fbStat(ii) == 1
            fDis = sApp + frameFac(ii);
        end
        
        sDis = sApp + frameFac(ii);
        
    end
    
    
   
    barInds(ii, :)  = [fApp, fDis, sApp, sDis] + fAppearInd;
end
    
protocolStruct.gratingTable.fAppear = barInds(:,1);
protocolStruct.gratingTable.fDisappear = barInds(:,2);
protocolStruct.gratingTable.sAppear = barInds(:,3);
protocolStruct.gratingTable.sDisappear = barInds(:,4);

clear barInds *Dis *App numStim spDiff relTab fAppearInd stepDurUnit simFrame ii frameFac fbStat


%% MinMoving Bar


fAppearInd = 12;
stepDurUnit = 0.02;



relTab = protocolStruct.gratingTable;
span = abs(relTab.startPos - relTab.stopPos);
frameFac = relTab.stepDur/ stepDurUnit;

numStim = height(relTab);
barInds= zeros(numStim,3); % for appearance and disappearance of both bars

for ii=1:numStim
    fApp = 0;    
    fDis = (span(ii)+1) * frameFac(ii);
    
    barInds(ii, 1:2)  = [fApp, fDis] + fAppearInd;
    barInds(ii,3) = frameFac(ii);
end
    
protocolStruct.gratingTable.appear = barInds(:,1);
protocolStruct.gratingTable.disappear = barInds(:,2);
protocolStruct.gratingTable.framePerStep = barInds(:,3);

clear barInds *Dis *App numStim spDiff relTab ii frameFac fAppearInd stepDurUnit

%% Flickerbar 


offApp = 20;
%offApp = 12;
stepDurUnit = 0.02;



relTab = protocolStruct.gratingTable;
frameFac = (relTab.cycDur/ stepDurUnit); %since it is full cycle (D & B) 
totTime = protocolStruct.inputParams.flickerDur;
 
numStim = height(relTab);
onInds= cell(numStim,1); 
offInds = onInds;
disInd = zeros(numStim, 1);


for ii=1:numStim
    
    tempNumStep = totTime/relTab.cycDur(ii) - 1;
    offInds{ii} = offApp:frameFac(ii):offApp+tempNumStep*frameFac(ii);
    onInds{ii} = offApp+frameFac(ii)/2:frameFac(ii):offApp+frameFac(ii)/2+tempNumStep*frameFac(ii);
    disInd(ii) = onInds{ii}(end)+frameFac(ii)/2;
    
end


protocolStruct.gratingTable.onAppear = onInds;
protocolStruct.gratingTable.offAppear = offInds;
protocolStruct.gratingTable.appear = ones(numStim,1)*offApp;
protocolStruct.gratingTable.disappear = disInd;

clear relTab numStim ii tempNumStep *Inds frameFac totTime *App stepDurUnit disInd

%% Moving bar 


fAppearInd = 20;
% fAppearInd = 12;
stepDurUnit = 0.02;



relTab = protocolStruct.gratingTable;
span = relTab.span;
ort = relTab.orient;
wid = relTab.width;
frameFac = relTab.stepDur/ stepDurUnit;

numStim = height(relTab);
barInds= zeros(numStim,4); % for appearance and disappearance of both bars

for ii=1:numStim
    fApp = 0;    
    
    relSp = span(ii);
    if rem(ort(ii),2) % to correct for diagonal movement
        relSp = 2*round(relSp/sqrt(2))+1;
    end
    
    cen = floor(relSp/2) * frameFac(ii); % since disappearing right before center is appearing in center
    fDis = relSp * frameFac(ii) + (wid(ii)-1) * frameFac(ii);
    
    barInds(ii, 1:2)  = [fApp, fDis] + fAppearInd;
    barInds(ii,3) = frameFac(ii);
    barInds(ii,4) = cen + fAppearInd;
end
    
protocolStruct.gratingTable.appear = barInds(:,1);
protocolStruct.gratingTable.disappear = barInds(:,2);
protocolStruct.gratingTable.framePerStep = barInds(:,3);
protocolStruct.gratingTable.center = barInds(:,4);


clear barInds fDis fApp numStim spDiff relTab ii frameFac relSp cen fAppearInd wid










