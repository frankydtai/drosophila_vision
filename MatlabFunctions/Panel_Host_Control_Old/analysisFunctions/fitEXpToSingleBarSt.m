function sigBarFitSt = fitEXpToSingleBarSt(sigBarProtSt, sigBarFitOpt)

% function sigBarFitSt = fitEXpToSingleBarSt(sigBarProtSt)
%
% this function fit uses generateAlignedSingleBarSt to generate the single
% bar response at each position and each time. It then fits an exponent to
% 3 different phases: zero to max resp; max to min; min to zero. if any of
% the phases are missing they are skipped (e.g. if there is no max resp
% then the function fits zero to min
%
% INPUT
%
% sigBarProtSt -        protocolStruct from a singleBar protocol
% sigBarFitOpt -        (optional). Structure with options for the fitting
%                       procedure
%   .useMed -           logical. if True uses median data and not mean
%
% OUTPUT
%
% TBD


fitOpt.useMed = 0;


if nargin < 2
    sigBarFitOpt = fitOpt;
else
    fNames = fieldnames(sigBarFitOpt);
    missingFields = find(~ismember(fNames, {'useMed'; 'postStimTime'}));
    for ii=1:length(missingFields)
        tempName = fNames(missingFields(ii));
        tempVal = getfield(fitOpt, tempName);
        
        sigBarFitOpt = setfield(sigBarFitOpt, tempName, tempVal);
        
    end
    
end


    



sigBSt = generateAlignedSingleBarSt(sigBarProtSt);

fOpt = fitoptions('method', 'nonlinearleastsquare', ...
                  'lower', [-1000, -1000, 0], ...
                  'upper', [1000, 1000, 1000], ...
                  'startpoint', [1, 10, 20]);

riseMod = fittype('a+b*(1-exp(-x/c))', 'coefficients', {'a', 'b', 'c'}, 'options', fOpt);
decayMod = fittype('a+b*(exp(-x/c))', 'coefficients', {'a', 'b', 'c'}, 'options', fOpt);

warning('off', 'curvefit:fit:noStartPoint')

datSiz = size(sigBSt);

gofMat = nan(datSiz(1), datSiz(2), 3); % 3 dim is 3 for rise, decay, and back to baseline

for ii=1:datSiz(1)
    
    for jj=1:datSiz(2)
        
        tempTime = sigBSt(ii,jj).subData.baseSub(:,1);
        
        if sigBarFitOpt.useMed
            tempResp = sigBSt(ii,jj).subData.baseSubMed;
        else
            tempResp = sigBSt(ii,jj).subData.baseSub(:,2);
        end 
        
        %fit parameters based on how many components are in the response
        %(rise/decay/recover)
        
        tempMaxMin = bin2dec(num2str(logical([sigBSt(ii,jj).resp.maxVal, sigBSt(ii,jj).resp.minVal])));
%          [ii,jj]
        switch tempMaxMin % since switch desn't work with vectors
            
            case 0
                
                sigBSt(ii,jj).fitType = 0;
                continue
                
            case 1
                
                sigBSt(ii,jj).fitType = 1;
                hypPInds = tempTime > sigBSt(ii,jj).resp.maxTime & tempTime < sigBSt(ii,jj).resp.minTime; 
                hypPTime = tempTime(hypPInds);
                hypPTimeZ = hypPTime - hypPTime(1); % fit won't work properly without this
                hypPResp = tempResp(hypPInds);
                
                [hypPFit, hypPGoF] = fit(hypPTimeZ, hypPResp, decayMod);
                
                gofMat(ii,jj,2) = hypPGoF.adjrsquare;
                sigBSt(ii,jj).fitResp(2).fit = hypPFit;
                sigBSt(ii,jj).fitResp(2).gof = hypPGoF;
                sigBSt(ii,jj).fitResp(2).relInds = hypPInds;
                
                backInds = tempTime > sigBSt(ii,jj).resp.minTime & tempTime < sigBSt(ii,jj).resp.postRespTime; 
                backTime = tempTime(backInds);
                backTimeZ = backTime - backTime(1); % fit won't work properly without this
                backResp = tempResp(backInds);
                
                [backFit, backGoF] = fit(backTimeZ, backResp, riseMod);
                
                gofMat(ii,jj,3) = backGoF.adjrsquare;
                sigBSt(ii,jj).fitResp(3).fit = backFit;
                sigBSt(ii,jj).fitResp(3).gof = backGoF;
                sigBSt(ii,jj).fitResp(3).relInds = backInds;
            
            case 2
                
                sigBSt(ii,jj).fitType = 2;
                % anytime maxResp is defined maxTime and preRespTime should
                % also be defined
                riseInds = tempTime > sigBSt(ii,jj).resp.preRespTime & tempTime < sigBSt(ii,jj).resp.maxTime; 
                riseTime = tempTime(riseInds);
                riseTimeZ = riseTime - riseTime(1); % fit won't work properly without this
                riseResp = tempResp(riseInds);
                
                [riseFit, riseGoF] = fit(riseTimeZ, riseResp, riseMod);
                
                gofMat(ii,jj,1) = riseGoF.adjrsquare;
                sigBSt(ii,jj).fitResp(1).fit = riseFit;
                sigBSt(ii,jj).fitResp(1).gof = riseGoF;
                sigBSt(ii,jj).fitResp(1).relInds = riseInds;
%                 if riseGoF.adjrsquare < 0.7 
%                     warning('rise adjustedRSq for position %d at time %d is %d', ii, jj, riseGoF.adjrsquare)
%                 end
                
                decayInds = tempTime > sigBSt(ii,jj).resp.maxTime & tempTime < sigBSt(ii,jj).resp.minTime; 
                
                decayTime = tempTime(decayInds);
                decayTimeZ = decayTime - decayTime(1); 
                decayResp = tempResp(decayInds);
                
                [decayFit, decayGoF] = fit(decayTimeZ, decayResp, decayMod);
                
%                 if decayGoF.adjrsquare < 0.7 
%                     warning('decay adjustedRSq for position %d at time %d is %d', ii, jj, decayGoF.adjrsquare)
%                 end
                
                gofMat(ii,jj,2) = decayGoF.adjrsquare;
                sigBSt(ii,jj).fitResp(2).fit = decayFit;
                sigBSt(ii,jj).fitResp(2).gof = decayGoF;
                sigBSt(ii,jj).fitResp(2).relInds = decayInds;
                
            case 3
                
                sigBSt(ii,jj).fitType = 3;
                riseInds = tempTime > sigBSt(ii,jj).resp.preRespTime & tempTime < sigBSt(ii,jj).resp.maxTime; 
                riseTime = tempTime(riseInds);
                riseTimeZ = riseTime - riseTime(1); % fit won't work properly without this
                riseResp = tempResp(riseInds);
                
                [riseFit, riseGoF] = fit(riseTimeZ, riseResp, riseMod);
                
                gofMat(ii,jj,1) = riseGoF.adjrsquare;
                sigBSt(ii,jj).fitResp(1).fit = riseFit;
                sigBSt(ii,jj).fitResp(1).gof = riseGoF;
                sigBSt(ii,jj).fitResp(1).relInds = riseInds;
                
                hypPInds = tempTime > sigBSt(ii,jj).resp.maxTime & tempTime < sigBSt(ii,jj).resp.minTime; 
                hypPTime = tempTime(hypPInds);
                hypPTimeZ = hypPTime - hypPTime(1); % fit won't work properly without this
                hypPResp = tempResp(hypPInds);
                
                [hypPFit, hypPGoF] = fit(hypPTimeZ, hypPResp, decayMod);
                
                gofMat(ii,jj,2) = hypPGoF.adjrsquare;
                sigBSt(ii,jj).fitResp(2).fit = hypPFit;
                sigBSt(ii,jj).fitResp(2).gof = hypPGoF;
                sigBSt(ii,jj).fitResp(2).relInds = hypPInds;
                
                backInds = tempTime > sigBSt(ii,jj).resp.minTime & tempTime < sigBSt(ii,jj).resp.postRespTime; 
                backTime = tempTime(backInds);
                backTimeZ = backTime - backTime(1); % fit won't work properly without this
                backResp = tempResp(backInds);
                
                [backFit, backGoF] = fit(backTimeZ, backResp, riseMod);
                
                gofMat(ii,jj,3) = backGoF.adjrsquare;
                sigBSt(ii,jj).fitResp(3).fit = backFit;
                sigBSt(ii,jj).fitResp(3).gof = backGoF;
                sigBSt(ii,jj).fitResp(3).relInds = backInds;
                
        end
        
    end
    
end
            
        
            
sigBarFitSt = sigBSt;

warning('on', 'curvefit:fit:noStartPoint')

posCell = generatePositionCell(0.05, 0.975, 0.1, 0.975, 0.05, -0.005, 3);

figure
axh = zeros(3,1);

for ii=1:3
    
    axh(ii) = axes('position', posCell{ii});
    imagesc(gofMat(:,:,ii), [0,1])
    hold on 
    [xx, yy] = find(isnan(gofMat(:,:,ii)));
    plot(yy,xx, 'sk','MarkerEdgeColor',[1,1,1]*0.6,'MarkerFaceColor',[1,1,1]*0.6, 'markersize', 20)
    hold off
end

axes('position', [0.25, 0.025, 0.5, 0.05])
imagesc(1:11)
set(gca, 'yticklabel', {}) 
set(gca, 'xtick', [1,6,11], 'xticklabel', {'0', '0.5', '1'})

hotCMap = flipud(colormap('hot'));

colormap(hotCMap)

end