function plotOptSt = makeDefaultPolarPlotOptionsStruct

% This function creates a default structure for polarPlot 

plotOptSt.type = 'line';
plotOptSt.normalize = 0;
plotOptSt.color = cbrewer('qual', 'Set1', 6);
plotOptSt.legend = 0;
plotOptSt.thetaVec = [0:-pi/4:-3*pi/4, pi:-pi/4:0.1]; 




end