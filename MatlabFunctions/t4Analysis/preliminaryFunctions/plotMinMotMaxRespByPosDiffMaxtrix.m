function varargout = plotMinMotMaxRespByPosDiffMaxtrix(pStruct, barFlag, presType)

% function varargout = plotMinMotMaxRespByPosDiffMaxtrix(pStruct)
%
% This function plot the minmimal motion protocol in a matrix of reciprocal
% pairs organized by their abolute positional difference (+/-1 from all 
% positions in one column +/-2 in the other etc.). each subplot show
% maxResop and max linResp as a function of timeDiff. 
%
% INPUT 
%
% pStruct -         protocol structure from minimal motion protocol
% barFlag -         integer showing which bar data to present. 
%                   1: first bar 
%                   2: second bar
%                   3: both 
% presType -        string. which way to present the data:
%                   'raw': actual values in mV
%                   'ratio': log of data/linear prediction
%
% OUTPUT
%
% if an output is requested the axes handles are given. 


ratioThresh = 0.01; % threshold beneth which data is converted to NaN (if ratio is calculated)


minMotSt = calculateMinMotExtLinSumDiff(pStruct);
datSiz = size(minMotSt);

relPosString = arrayfun(@num2str, pStruct.inputParams.fBarPos, 'uniformoutput', 0);
relTDString = arrayfun(@num2str, pStruct.inputParams.timeDiff * 1000, 'uniformoutput', 0);


assert(ismember(presType, {'ratio'; 'raw'}), 'presType should be either raw or ratio')

switch barFlag
    case 1
        barInd = barFlag;
        barLab = {'F'}; % First
    case 2
        barInd = barFlag;
        barLab = {'S'}; % Second
    case 3
        barInd = [1,2];
        barLab = {'F'; 'S'}; 
    otherwise
        error('barFlag can only be 1,2, or 3 (both)')
end

if strcmp(presType, 'ratio')
    rawFlag = 0;
    presLab = {'R'}; %ratio
else
    rawFlag = 1;
    presLab = {'D'; 'L'}; % data ; linear
end

dirLab = {'P'; 'N'}; % positive ; negative

allCombLab = cell(1, length(barLab)*length(presLab)*length(dirLab));

count=0;
for ii=1:length(barLab)
    for jj=1:length(dirLab)
        for kk=1:length(presLab)
            count=count+1;
            allCombLab{count} = [barLab{ii}, dirLab{jj}, presLab{kk}];
        end
    end
end
            


lastDimSiz = 2*(rawFlag+1);

datMat = nan(datSiz(1)-1, datSiz(1)-1, datSiz(3), length(barInd)*lastDimSiz); 
for ii=1:datSiz(1)
    
    for jj=ii+1:datSiz(2)
        
        for kk=1:datSiz(3)
            
            for bi=1:length(barInd)
                
                tempRDPos = minMotSt(ii,jj,kk).linDiff(barInd(bi)).maxResp(2);
                tempLDPos = minMotSt(ii,jj,kk).linDiff(barInd(bi)).maxLinResp(2);
                
                if isempty(minMotSt(jj,ii,kk).linDiff) %in sim cases where upper triangle is missing
                    tempRDNeg = nan;
                    tempLDNeg = nan;
                else
                    tempRDNeg = minMotSt(jj,ii,kk).linDiff(barInd(bi)).maxResp(2);
                    tempLDNeg = minMotSt(jj,ii,kk).linDiff(barInd(bi)).maxLinResp(2);
                end
                
                lastInds = 1+(bi-1)*lastDimSiz:lastDimSiz+(bi-1)*lastDimSiz;
                
                if rawFlag
                    
                    datMat(ii,jj,kk,lastInds) = [tempRDPos, tempLDPos, tempRDNeg, tempLDNeg];
                    
                else
                    
                    datMat(ii,jj,kk,lastInds) = [tempRDPos./tempLDPos, tempRDNeg./tempLDNeg];
                    
                end
                
            end
                
        end
        
    end
    
end


% to enable log
if ~rawFlag 
    datMat(datMat < ratioThresh) = nan;
    datMat = log(datMat);
end



 
figure
posCell = generatePositionCell(0.05, 0.975, 0.05, 0.975, 0.02, 0.05, [datSiz(1)-1, datSiz(1)-1]);

totLastD = length(barInd)*lastDimSiz;
count=0;


allCol = cbrewer('qual', 'Paired', 6);

for ii=1:datSiz(1)
    
    for jj=ii+1:datSiz(2)
        count = count+1;
        axh(count) = axes('position', posCell{ii,jj-1});
        title([relPosString{ii}, 'vs' relPosString{jj}])
        hold on
        for ld=1:totLastD
            
            if allCombLab{ld}(1) == 'F'
                relPS = '--o';
            else
                relPS = '-o';
            end
            
            if allCombLab{ld}(2) == 'P'
                tempRelCol = allCol(1:2, :);
            else
                tempRelCol = allCol(5:6, :);
            end
            
            if allCombLab{ld}(3) == 'L'
                relCol = tempRelCol(1, :);
            else
                relCol = tempRelCol(2, :);
            end
            
            tempD = squeeze(datMat(ii,jj,:,ld));
            tempDSepSim = nan(1, length(tempD)+1); % forces a break between sim data and seq
            tempDSepSim(1) = tempD(1);
            tempDSepSim(3:end) = tempD(2:end);
            tempXSepSim = [1, 1.5, 2:datSiz(3)];
            
            plot(tempXSepSim, tempDSepSim, relPS, 'color', relCol, 'markerfacecolor', relCol)
            
        end
        
        hold off
        
    end
    
end


legend(axh(end), allCombLab)

% so that the line won't creep into the legend
if ~rawFlag             
    for ii=1:length(axh)
        set(gcf, 'currentaxes', axh(ii));
        line([0.5, datSiz(3)+0.5], [0,0], 'color', [1,1,1]*0.8, 'linewidth', 1)
    end
end
    


equalizeYAxes(axh)

set(axh(:), 'xlim', [0.5, datSiz(3)+0.5])

relaxhInd = 0;
for ii=1:datSiz(1)-1 
    relaxhInd = [relaxhInd, relaxhInd(end)+datSiz(1)-1-(ii-1)]; 
end
relaxhInd = relaxhInd(2:end);

set(axh(:), 'xtick', 1:datSiz(3), 'xticklabel', {})
set(axh(relaxhInd), 'xticklabel', relTDString)
set(axh(datSiz(1)+1:end), 'yticklabel', {})

if nargout ==1  
    varargout{1} = axh;
end


    
    
end