function corrVec = extrapolateShiftedVec(shiftVec)

% function corrVec = extrapolateShiftedVec(shiftVec)
%
% This function is used internally in padRespVec to reduce the artefact
% stemming from pinning the end of the response to zero. If the response
% has an inhibitory component, pinning to zero underestimates the effect.
% Therefore, this function fits the end of the resposen with a line and
% stops the fit when values return to zero. 

estWin = 5000; % samples back from 

corrVec = shiftVec;

endInd = find(shiftVec ~= 0, 1, 'last');
hypPStartInd = find(shiftVec(1:endInd) > 0, 1, 'last');

hypPolFlag = mean(shiftVec(endInd-estWin:endInd)) < 0; % data in the end is negative
hypEndFlag = endInd < length(shiftVec) - estWin/2; % hypPol is not at the very end of the vector
hypLenFlag = endInd - hypPStartInd > estWin; % hypPol is long enough

if  hypPolFlag && hypEndFlag && hypLenFlag
    
    tempFit = fit((endInd-5000:endInd)', shiftVec(endInd-5000:endInd), 'poly1');
    
    if tempFit.p1 < 0
        fprintf('negative slope was fit for shifted vec - no change made \n')
        
%         plot(shiftVec)
%         pause
        
    else
        solX = round(-tempFit.p2/tempFit.p1);
        
        fitRange = endInd:min(solX, length(shiftVec));
            
        corrVec(fitRange) = tempFit.p1*(fitRange) + tempFit.p2;
        
    end
end





end