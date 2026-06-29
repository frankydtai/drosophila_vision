function plotPolarResultForMovBar(movBarSt)

% function plotPolarResultForMovBar(movBarSt)
%
% This function is designed to plot the polar results from calcCircResultsForMovingBar
% it uses the internal polarPlot and the specific structure of the results
%
% Note Code was taken from plotMovBarSummaryResult (plots 3 summary plot
% and this is just one_


datSiz = size(movBarSt);

allRelResp = zeros(datSiz(1), datSiz(2)-1);

for ii=1:datSiz(1)
    allRelResp(ii,:) = movBarSt(ii,9).result.respatMaxVM;
end

tempCol = cbrewer('qual', 'Paired', 8);
polOpt.type = 'both';
polOpt.color = tempCol(2:2:end, :);

polarPlot(allRelResp, polOpt) % resp mag and theta



end