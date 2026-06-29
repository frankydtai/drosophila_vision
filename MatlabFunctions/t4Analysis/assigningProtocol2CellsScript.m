

load ./T4recordingSummaryAndAnalysis/relDirNames.mat

relDirNames = relDirNames(setdiff(1:20, [1,3,4]), :);

% excluded cells 1, 3 and 4 due to very weak responses to singleBar (see
% singleBarExtAndInhibMax.pdf )



clear t4Cells

% cn=1;
% t4Cells(cn).dir = relDirNames(cn,:); % associated single bar protocol
% t4Cells(cn).singleBar(1).fileName = '/SingleBarDiagCorrProtocol20160418_15-15/protocolStruct20160418_15-15.mat';
% t4Cells(cn).singleBar(1).pos = -8:8;
% t4Cells(cn).flicker(1).fileName = '/FlickerBarDiagCorrProtocol20160418_16-08/protocolStruct20160418_16-08.mat';
% t4Cells(cn).flicker(1).pos = -5:2;
% t4Cells(cn).minMot(1).fileName = '/MinMotDiagCorrProtocol20160418_15-26/protocolStruct20160418_15-26.mat';
% t4Cells(cn).minMot(1).fbPos = -5:1;
% t4Cells(cn).minMot(1).sbPos = -5:1;
% t4Cells(cn).minMot(1).barV = 'BB';
% t4Cells(cn).minMot(2).fileName = '/MinMotDiagCorrProtocol20160418_15-44/protocolStruct20160418_15-44.mat';
% t4Cells(cn).minMot(2).fbPos = 1:4;
% t4Cells(cn).minMot(2).sbPos = 0;
% t4Cells(cn).minMot(2).barV = 'BB';
% t4Cells(cn).minMot(3).fileName = '/MinMotDiagCorrProtocol20160418_15-48/protocolStruct20160418_15-48.mat';
% t4Cells(cn).minMot(3).fbPos = -5:1;
% t4Cells(cn).minMot(3).sbPos = -5:1;
% t4Cells(cn).minMot(3).barV = 'BD';
% t4Cells(cn).movBar(1).fileName = '/MovingBarDiagCorrProtocol20160418_15-09/protocolStruct20160418_15-09.mat';
% t4Cells(cn).movEdge(1).fileName = '/MovingEdgeDiagCorrProtocol20160418_15-52/protocolStruct20160418_15-52.mat';
% t4Cells(cn).grating(1).fileName = '/GratingProtocol20160418_16-13/protocolStruct20160418_16-13.mat';
% t4Cells(cn).notes = {'flicker data is not very useful'; ...
%                      'minMot(1): noisy but still seems useful'; ...
%                      'hard to determine noisy level since it is a seperate distribution and not just outliers'; ...
%                      'considerable noise in -3 -3 40ms that ruins a row and a column'};



cn=1;
t4Cells(cn).dir = relDirNames(cn,:);
t4Cells(cn).singleBar(1).fileName = '/SingleBarDiagCorrProtocol20160420_12-10/protocolStruct20160420_12-10.mat';
t4Cells(cn).singleBar(1).pos = -5:5;
t4Cells(cn).minMot(1).fileName = '/MinMotDiagCorrProtocol20160420_12-18/protocolStruct20160420_12-18.mat';
t4Cells(cn).minMot(1).fbPos = -1:3;
t4Cells(cn).minMot(1).sbPos = -1:3;
t4Cells(cn).minMot(1).barV = 'BB';
t4Cells(cn).minMot(2).fileName = '/MinMotDiagCorrProtocol20160420_12-29/protocolStruct20160420_12-29.mat';
t4Cells(cn).minMot(2).fbPos = -4:-1;
t4Cells(cn).minMot(2).sbPos = 0;
t4Cells(cn).minMot(2).barV = 'BB';
t4Cells(cn).movBar(1).fileName = '/MovingBarDiagCorrProtocol20160420_11-58/protocolStruct20160420_11-58.mat';
t4Cells(cn).movEdge(1).fileName = '/MovingEdgeDiagCorrProtocol20160420_12-05/protocolStruct20160420_12-05.mat';
t4Cells(cn).notes = {'minMot(1): noisy but still see resp in 20ms'; ...
                     'minMot(2): too noisy to be useful'};

% cn=3;
% t4Cells(cn).dir = relDirNames(cn,:);
% t4Cells(cn).singleBar(1).fileName = '/SingleBarDiagCorrProtocol20160420_16-48/protocolStruct20160420_16-48.mat';
% t4Cells(cn).singleBar(1).pos = -6:6;
% t4Cells(cn).flicker(1).fileName = '/FlickerBarDiagCorrProtocol20160420_17-19/protocolStruct20160420_17-19.mat';
% t4Cells(cn).flicker(1).pos = -2:2;
% t4Cells(cn).minMot(1).fileName = '/MinMotDiagCorrProtocol20160420_16-55/protocolStruct20160420_16-55.mat';
% t4Cells(cn).minMot(1).fbPos = -2:2;
% t4Cells(cn).minMot(1).sbPos = -2:2;
% t4Cells(cn).minMot(1).barV = 'BB';
% t4Cells(cn).minMot(2).fileName = '/MinMotDiagCorrProtocol20160420_17-06/protocolStruct20160420_17-06.mat';
% t4Cells(cn).minMot(2).fbPos = [2,3,4,6];
% t4Cells(cn).minMot(2).sbPos = 0;
% t4Cells(cn).minMot(2).barV = 'BB';
% t4Cells(cn).minMot(3).fileName = '/MinMotDiagCorrProtocol20160420_17-10/protocolStruct20160420_17-10.mat';
% t4Cells(cn).minMot(3).fbPos = -2:2;
% t4Cells(cn).minMot(3).sbPos = -3:3;
% t4Cells(cn).minMot(3).barV = 'BD';
% t4Cells(cn).minMov(1).fileName = '/MinMovingBarDiagCorrProtocol20160420_17-32/protocolStruct20160420_17-32.mat';
% t4Cells(cn).minMov(1).pos = [-3,0; -3,1; -3,2; -3,3; -3,3; -3,4; -2,0; -2,1; -2,2; -2,3; -2,3; -2,4; 2,4; 2,6]; 
% t4Cells(cn).movBar(1).fileName = '/MovingBarDiagCorrProtocol20160420_16-42/protocolStruct20160420_16-42.mat';
% t4Cells(cn).movBar(2).fileName = '/MovingBarDiagCorrProtocol20160420_17-39/protocolStruct20160420_17-39.mat';
% t4Cells(cn).movBar(2).height = [5,9,13];
% t4Cells(cn).movBar(3).fileName = '/MovingBarDiagCorrProtocol20160420_17-44/protocolStruct20160420_17-44.mat';
% t4Cells(cn).movBar(3).barV = 'D';
% t4Cells(cn).movBar(4).fileName = '/MovingBarDiagCorrProtocol20160420_17-50/protocolStruct20160420_17-50.mat';
% t4Cells(cn).movBar(4).width = [1,2,4];
% t4Cells(cn).movEdge(1).fileName = '/MovingEdgeDiagCorrProtocol20160420_17-14/protocolStruct20160420_17-14.mat';
% t4Cells(cn).grating(1).fileName = '/GratingProtocol20160420_17-22/protocolStruct20160420_17-22.mat';
% t4Cells(cn).grating(2).fileName = '/GratingProtocol20160420_18-03/protocolStruct20160420_18-03.mat';
% t4Cells(cn).grating(2).revPhi = 1;
% t4Cells(cn).figGrd(1).fileName = '/MovingFigGrdDiagCorrProtocol20160420_17-28/protocolStruct20160420_17-28.mat';
% t4Cells(cn).figGrd(2).fileName = '/MovingFigGrdDiagCorrProtocol20160420_17-56/protocolStruct20160420_17-56.mat';
% t4Cells(cn).notes = {'receptive field touching arena edge - some stim might be missing pixels'; ...
%                      'SingleBar(1): strongest resp are about 4mV, the cell might be useless'; ...
%                      'flicker not pretty, but linFit still reasonable'; 'weak responses'; ...
%                      'minMot(1): responses are weak and some repeats are fairly noisy - dont know if useful'; ...
%                      'minMot(2); not sure useful'; ...
%                      'movBar(3): dark bar that seem to have the opposite preferance to bright (even though almost no resp)'};
% 
% 
% 
% cn=4;
% t4Cells(cn).dir = relDirNames(cn,:);
% t4Cells(cn).singleBar(1).fileName = '/SingleBarDiagCorrProtocol20160422_16-26/protocolStruct20160422_16-26.mat';
% t4Cells(cn).singleBar(1).pos = -6:6;
% t4Cells(cn).flicker(1).fileName = '/FlickerBarDiagCorrProtocol20160422_17-32/protocolStruct20160422_17-32.mat';
% t4Cells(cn).flicker(1).pos = -2:3;
% t4Cells(cn).minMot(1).fileName = '/MinMotDiagCorrProtocol20160422_16-36/protocolStruct20160422_16-36.mat';
% t4Cells(cn).minMot(1).fbPos = -2:2;
% t4Cells(cn).minMot(1).sbPos = -2:2;
% t4Cells(cn).minMot(1).barV = 'BB';
% t4Cells(cn).minMot(2).fileName = '/MinMotDiagCorrProtocol20160422_16-48/protocolStruct20160422_16-48.mat';
% t4Cells(cn).minMot(2).fbPos = 3:5;
% t4Cells(cn).minMot(2).sbPos = 0;
% t4Cells(cn).minMot(2).barV = 'BB';
% t4Cells(cn).minMot(3).fileName = '/MinMotDiagCorrProtocol20160422_17-00/protocolStruct20160422_17-00.mat';
% t4Cells(cn).minMot(3).fbPos = -4:1;
% t4Cells(cn).minMot(3).sbPos = -2:2;
% t4Cells(cn).minMot(3).barV = 'DB';
% t4Cells(cn).minMov(1).fileName = '/MinMovingBarDiagCorrProtocol20160422_17-10/protocolStruct20160422_17-10.mat';
% t4Cells(cn).minMov(1).pos = [-4,2; -2,-4; -2,0; -2,2; -2,4; 0,2; 3,5]; 
% t4Cells(cn).movBar(1).fileName = '/MovingBarDiagCorrProtocol20160422_16-14/protocolStruct20160422_16-14.mat';
% t4Cells(cn).movBar(2).fileName = '/MovingBarProtocol20160422_16-24/protocolStruct20160422_16-24.mat';
% t4Cells(cn).movBar(2).oldProt = 1;
% t4Cells(cn).movBar(3).fileName = '/MovingBarDiagCorrProtocol20160422_17-05/protocolStruct20160422_17-05.mat';
% t4Cells(cn).movBar(3).height = [5,9,13];
% t4Cells(cn).movBar(4).fileName = '/MovingBarDiagCorrProtocol20160422_17-15/protocolStruct20160422_17-15.mat';
% t4Cells(cn).movBar(4).width = [1,2,4];
% t4Cells(cn).movEdge(1).fileName = '/MovingEdgeDiagCorrProtocol20160422_16-20/protocolStruct20160422_16-20.mat';
% t4Cells(cn).grating(1).fileName = '/GratingProtocol20160422_16-49/protocolStruct20160422_16-49.mat';
% t4Cells(cn).figGrd(1).fileName = '/MovingFigGrdDiagCorrProtocol20160422_17-21/protocolStruct20160422_17-21.mat';
% t4Cells(cn).notes = {'flicker not useful'; ...
%                      'minMot(1): fast speed probably not very useful, sim is'; ...
%                      'singleBar: strongest resp is ~5mV, cell might be useless'; ...
%                      'minMot(3): supression near diagonal only in sim (40ms)'};

cn=2;
t4Cells(cn).dir = relDirNames(cn,:);
t4Cells(cn).singleBar(1).fileName = '/SingleBarDiagCorrProtocol20160426_12-48/protocolStruct20160426_12-48.mat';
t4Cells(cn).singleBar(1).pos = -5:5;
t4Cells(cn).singleBar(1).correct = 1;
t4Cells(cn).singleBar(2).fileName = '/SingleBarDiagCorrProtocol20160426_13-20/protocolStruct20160426_13-20.mat';
t4Cells(cn).singleBar(2).pos = -5:5;
t4Cells(cn).singleBar(2).barV = 0;
t4Cells(cn).flicker(1).fileName = '/FlickerBarDiagCorrProtocol20160426_13-11/protocolStruct20160426_13-11.mat';
t4Cells(cn).flicker(1).pos = -2:3;
t4Cells(cn).minMot(1).fileName = '/MinMotDiagCorrProtocol20160426_12-57/protocolStruct20160426_12-57.mat';
t4Cells(cn).minMot(1).fbPos = -1:3;
t4Cells(cn).minMot(1).sbPos = -1:3;
t4Cells(cn).minMot(1).barV = 'BB';
t4Cells(cn).minMot(2).fileName = '/MinMotDiagCorrProtocol20160426_13-08/protocolStruct20160426_13-08.mat';
t4Cells(cn).minMot(2).fbPos = -4:-2;
t4Cells(cn).minMot(2).sbPos = -1;
t4Cells(cn).minMot(2).barV = 'BB';
t4Cells(cn).minMot(3).fileName = '/MinMotDiagCorrProtocol20160426_13-24/protocolStruct20160426_13-24.mat';
t4Cells(cn).minMot(3).fbPos = -1:3;
t4Cells(cn).minMot(3).sbPos = -1:3;
t4Cells(cn).minMot(3).barV = 'DB';
t4Cells(cn).minMov(1).fileName = '/MinMovingBarDiagCorrProtocol20160426_13-30/protocolStruct20160426_13-30.mat';
t4Cells(cn).minMov(1).pos = [-5,0; -4,-2; -4,1; -1,1; -1,3; -1,5; 0,3]; 
t4Cells(cn).movBar(1).fileName = '/MovingBarDiagCorrProtocol20160426_12-42/protocolStruct20160426_12-42.mat';
t4Cells(cn).movBar(2).fileName = '/MovingBarDiagCorrProtocol20160426_13-48/protocolStruct20160426_13-48.mat';
t4Cells(cn).movBar(2).height = [5,9,13];
t4Cells(cn).movBar(3).fileName = '/MovingBarDiagCorrProtocol20160426_13-51/protocolStruct20160426_13-51.mat';
t4Cells(cn).movBar(3).width = [1,2,4];
t4Cells(cn).movEdge(1).fileName = '/MovingEdgeDiagCorrProtocol20160426_13-15/protocolStruct20160426_13-15.mat';
t4Cells(cn).grating(1).fileName = '/GratingProtocol20160426_13-35/protocolStruct20160426_13-35.mat';
t4Cells(cn).figGrd(1).fileName = '/MovingFigGrdDiagCorrProtocol20160426_13-43/protocolStruct20160426_13-43.mat';
t4Cells(cn).notes = {'flicker not useful'; ...
                     'minMot(1): clean data, resp @ 20ms might be too weak to be useful'; ...
                     'included weak ext in the protocol'; ...
                     'minMot(3): supression near diagonal only in sim (40ms)'; ...
                     'has singlebar dark for this cell but shows no resp'};


cn=3;
t4Cells(cn).dir = relDirNames(cn,:);
t4Cells(cn).singleBar(1).fileName = '/SingleBarDiagCorrProtocol20160427_12-00/protocolStruct20160427_12-00.mat';
t4Cells(cn).singleBar(1).pos = -5:5;
t4Cells(cn).singleBar(1).ort = 0;
t4Cells(cn).singleBar(1).correct = 1;
t4Cells(cn).singleBar(2).fileName = '/SingleBarDiagCorrProtocol20160427_12-08/protocolStruct20160427_12-08.mat';
t4Cells(cn).singleBar(2).pos = -6:6;
t4Cells(cn).singleBar(2).ort = 1;
t4Cells(cn).singleBar(3).fileName = '/SingleBarDiagCorrProtocol20160427_12-36/protocolStruct20160427_12-36.mat';
t4Cells(cn).singleBar(3).pos = -5:5;
t4Cells(cn).singleBar(3).barV = 0;
t4Cells(cn).singleBar(3).ort = 0;
t4Cells(cn).flicker(1).fileName = '/FlickerBarDiagCorrProtocol20160427_12-30/protocolStruct20160427_12-30.mat';
t4Cells(cn).flicker(1).pos = -4:2;
t4Cells(cn).minMot(1).fileName = '/MinMotDiagCorrProtocol20160427_12-17/protocolStruct20160427_12-17.mat';
t4Cells(cn).minMot(1).fbPos = -3:1;
t4Cells(cn).minMot(1).sbPos = -3:1;
t4Cells(cn).minMot(1).barV = 'BB';
t4Cells(cn).minMot(2).fileName = '/MinMotDiagCorrProtocol20160427_12-27/protocolStruct20160427_12-27.mat';
t4Cells(cn).minMot(2).fbPos = 2:3;
t4Cells(cn).minMot(2).sbPos = 0;
t4Cells(cn).minMot(2).barV = 'BB';
t4Cells(cn).minMot(3).fileName = '/MinMotDiagCorrProtocol20160427_12-39/protocolStruct20160427_12-39.mat';
t4Cells(cn).minMot(3).fbPos = -4:2;
t4Cells(cn).minMot(3).sbPos = -3:1;
t4Cells(cn).minMot(3).barV = 'DB';
t4Cells(cn).minMov(1).fileName = '/MinMovingBarDiagCorrProtocol20160427_12-45/protocolStruct20160427_12-45.mat';
t4Cells(cn).minMov(1).pos = [-5,1; -3,-1; -3,1; -1,1; 0,3; 0,5]; 
t4Cells(cn).movBar(1).fileName = '/MovingBarDiagCorrProtocol20160427_11-46/protocolStruct20160427_11-46.mat';
t4Cells(cn).movBar(2).fileName = '/MovingBarDiagCorrProtocol20160427_12-59/protocolStruct20160427_12-59.mat';
t4Cells(cn).movBar(2).height = [5,9,13];
t4Cells(cn).movBar(3).fileName = '/MovingBarDiagCorrProtocol20160427_13-05/protocolStruct20160427_13-05.mat';
t4Cells(cn).movBar(3).width = [1,2,4];
t4Cells(cn).movEdge(1).fileName = '/MovingEdgeDiagCorrProtocol20160427_11-52/protocolStruct20160427_11-52.mat';
t4Cells(cn).grating(1).fileName = '/GratingProtocol20160427_12-50/protocolStruct20160427_12-50.mat';
t4Cells(cn).figGrd(1).fileName = '/MovingFigGrdDiagCorrProtocol20160427_13-11/protocolStruct20160427_13-11.mat';
t4Cells(cn).notes = {'flicker: nice linComp in several positions'; 'clear responses even in 20ms'; ...
                     'leading side doesnt show resp in 40ms - same as in single bar'; ...
                     'minMot(1): shows small spikes 2-3mV'; 'weird faliure for FB response in (0,-2)'; ...
                     'other positions sometimes have one failure'; ...
                     'minMot(3): supression around diagonal, only in sim (not 40ms) - responses are clear'; ...
                     'have dark singlebar protocol for this cell but it shows no resp'};

cn=4;
t4Cells(cn).dir = relDirNames(cn,:);
t4Cells(cn).singleBar(1).fileName = '/SingleBarDiagCorrProtocol20160428_16-28/protocolStruct20160428_16-28.mat';
t4Cells(cn).singleBar(1).pos = -8:8;
t4Cells(cn).flicker(1).fileName = '/FlickerBarDiagCorrProtocol20160428_16-54/protocolStruct20160428_16-54.mat';
t4Cells(cn).flicker(1).pos = -3:3;
t4Cells(cn).minMot(1).fileName = '/MinMotDiagCorrProtocol20160428_16-40/protocolStruct20160428_16-40.mat';
t4Cells(cn).minMot(1).fbPos = -1:3;
t4Cells(cn).minMot(1).sbPos = -1:3;
t4Cells(cn).minMot(1).barV = 'BB';
t4Cells(cn).minMot(2).fileName = '/MinMotDiagCorrProtocol20160428_16-51/protocolStruct20160428_16-51.mat';
t4Cells(cn).minMot(2).fbPos = -4:-2;
t4Cells(cn).minMot(2).sbPos = 0;
t4Cells(cn).minMot(2).barV = 'BB';
t4Cells(cn).minMot(3).fileName = '/MinMotDiagCorrProtocol20160428_17-06/protocolStruct20160428_17-06.mat';
t4Cells(cn).minMot(3).fbPos = -1:4;
t4Cells(cn).minMot(3).sbPos = -1:2;
t4Cells(cn).minMot(3).barV = 'DB';
% repeated to add FB remain
t4Cells(cn).minMot(4).fileName = '/MinMotDiagCorrProtocol20160428_17-11/protocolStruct20160428_17-11.mat';
t4Cells(cn).minMot(4).fbPos = -4:-2;
t4Cells(cn).minMot(4).sbPos = 0;
t4Cells(cn).minMot(4).barV = 'BB';
t4Cells(cn).minMov(1).fileName = '/MinMovingBarDiagCorrProtocol20160428_17-00/protocolStruct20160428_17-00.mat';
t4Cells(cn).minMov(1).pos = [-5,3; -3,3; -1,1; 0,3; 0,5; 0,7]; 
t4Cells(cn).movBar(1).fileName = '/MovingBarDiagCorrProtocol20160428_16-17/protocolStruct20160428_16-17.mat';
% the 3 2 order in movBar is not a mistake 
t4Cells(cn).movBar(3).fileName = '/MovingBarDiagCorrProtocol20160428_17-32/protocolStruct20160428_17-32.mat'; 
t4Cells(cn).movBar(3).height = [5,9,13];
t4Cells(cn).movBar(2).fileName = '/MovingBarDiagCorrProtocol20160428_17-21/protocolStruct20160428_17-21.mat';
t4Cells(cn).movBar(2).width = [1,2,4];
t4Cells(cn).movEdge(1).fileName = '/MovingEdgeDiagCorrProtocol20160428_16-23/protocolStruct20160428_16-23.mat';
t4Cells(cn).grating(1).fileName = '/GratingProtocol20160428_17-12/protocolStruct20160428_17-12.mat';
t4Cells(cn).figGrd(1).fileName = '/MovingFigGrdDiagCorrProtocol20160428_17-27/protocolStruct20160428_17-27.mat';
t4Cells(cn).notes = {'flicker: perfect linComp in all positions'; 'missing flicker positions in leading side'; ...
                     'clear phase locked responses even in 20ms'; 'leading pos3 weak resp in 40ms 0 - same as in single bar'; ...
                     'pos2 resp more strongly even though in single bar it is weak'; ...
                     'minMot(1): pos 3 in 80ms has no resp (all 3 reps)'; ...
                     'minMot(3): speed corr, 40ms for sim and speed, nice supression in sim but not 40'; ...
                     'movBar(3): very noisy - not useful'};

cn=5;
t4Cells(cn).dir = relDirNames(cn,:);
t4Cells(cn).singleBar(1).fileName = '/SingleBarDiagCorrProtocol20160502_17-40/protocolStruct20160502_17-40.mat';
t4Cells(cn).singleBar(1).pos = -6:6;
t4Cells(cn).flicker(1).fileName = '/FlickerBarDiagCorrProtocol20160502_18-05/protocolStruct20160502_18-05.mat';
t4Cells(cn).flicker(1).pos = -3:3;
t4Cells(cn).minMot(1).fileName = '/MinMotDiagCorrProtocol20160502_17-50/protocolStruct20160502_17-50.mat';
t4Cells(cn).minMot(1).fbPos = -2:2;
t4Cells(cn).minMot(1).sbPos = -2:2;
t4Cells(cn).minMot(1).barV = 'BB';
t4Cells(cn).minMot(2).fileName = '/MinMotDiagCorrProtocol20160502_18-01/protocolStruct20160502_18-01.mat';
t4Cells(cn).minMot(2).fbPos = -5:-3;
t4Cells(cn).minMot(2).sbPos = -1;
t4Cells(cn).minMot(2).barV = 'BB';
t4Cells(cn).minMot(3).fileName = '/MinMotDiagCorrProtocol20160502_18-27/protocolStruct20160502_18-27.mat';
t4Cells(cn).minMot(3).fbPos = -1:4;
t4Cells(cn).minMot(3).sbPos = -2:2;
t4Cells(cn).minMot(3).barV = 'DB';
t4Cells(cn).minMov(1).fileName = '/MinMovingBarDiagCorrProtocol20160502_18-12/protocolStruct20160502_18-12.mat';
t4Cells(cn).minMov(1).pos = [-5,-3; -5,2; -3,0; -2,1; -2,2; -2,5]; 
t4Cells(cn).movBar(1).fileName = '/MovingBarDiagCorrProtocol20160502_17-30/protocolStruct20160502_17-30.mat';
t4Cells(cn).movBar(2).fileName = '/MovingBarDiagCorrProtocol20160502_17-36/protocolStruct20160502_17-36.mat';
t4Cells(cn).movBar(2).width = [1,2,4];
t4Cells(cn).movEdge(1).fileName = '/MovingEdgeDiagCorrProtocol20160502_18-37/protocolStruct20160502_18-37.mat';
t4Cells(cn).grating(1).fileName = '/GratingProtocol20160502_18-16/protocolStruct20160502_18-16.mat';
t4Cells(cn).figGrd(1).fileName = '/MovingFigGrdDiagCorrProtocol20160502_18-33/protocolStruct20160502_18-33.mat';
t4Cells(cn).notes = {'flicker: clear resp in 40ms, but some good fits'; ...
                     'minMot(3): noisy and missing some positions (code bug - not sure useful'; ...
                     'sim is mainly linear, but resp is weak, 80ms shows somw sub singlebar - speed corr'};


cn=6;
t4Cells(cn).dir = relDirNames(cn,:);
t4Cells(cn).singleBar(1).fileName = '/SingleBarDiagCorrProtocol20160504_17-24/protocolStruct20160504_17-24.mat';
t4Cells(cn).singleBar(1).pos = -6:6;
t4Cells(cn).singleBar(1).ort = 3; 
t4Cells(cn).singleBar(2).fileName = '/SingleBarDiagCorrProtocol20160504_17-48/protocolStruct20160504_17-48.mat';
t4Cells(cn).singleBar(2).pos = -5:5;
t4Cells(cn).singleBar(2).ort = 2;
t4Cells(cn).singleBar(2).correct = 1;
t4Cells(cn).flicker(1).fileName = '/FlickerBarDiagCorrProtocol20160504_18-13/protocolStruct20160504_18-13.mat';
t4Cells(cn).flicker(1).pos = -3:4;
t4Cells(cn).minMot(1).fileName = '/MinMotDiagCorrProtocol20160504_17-58/protocolStruct20160504_17-58.mat';
t4Cells(cn).minMot(1).fbPos = -1:3;
t4Cells(cn).minMot(1).sbPos = -1:3;
t4Cells(cn).minMot(1).barV = 'BB';
t4Cells(cn).minMot(2).fileName = '/MinMotDiagCorrProtocol20160504_18-08/protocolStruct20160504_18-08.mat';
t4Cells(cn).minMot(2).fbPos = [-5:-3, 0];
t4Cells(cn).minMot(2).sbPos = 0;
t4Cells(cn).minMot(2).barV = 'BB';
t4Cells(cn).minMot(3).fileName = '/MinMotDiagCorrProtocol20160504_18-22/protocolStruct20160504_18-22.mat';
t4Cells(cn).minMot(3).fbPos = -1:4;
t4Cells(cn).minMot(3).sbPos = -2:2;
t4Cells(cn).minMot(3).barV = 'DB';
t4Cells(cn).minMov(1).fileName = '/MinMovingBarDiagCorrProtocol20160504_18-18/protocolStruct20160504_18-18.mat';
t4Cells(cn).minMov(1).pos = [-5,-3; -5,0; -1,1; 0,4; 0,6]; 
t4Cells(cn).movBar(1).fileName = '/MovingBarDiagCorrProtocol20160504_17-44/protocolStruct20160504_17-44.mat';
t4Cells(cn).movBar(1).stepDur = [0.04, 0.08];
t4Cells(cn).movBar(2).fileName = '/MovingBarDiagCorrProtocol20160504_17-55/protocolStruct20160504_17-55.mat';
t4Cells(cn).movBar(2).stepDur = [0.02, 0.16];
t4Cells(cn).movBar(3).fileName = '/protocolStruct_combMovBar1And2.mat';
t4Cells(cn).notes = {'flicker: good fits at slower flickers'; ...
                     'pos 3,4 on leading side show no phase lock and weak resp in 40ms'; ...
                     'their respopnses in singlebar are obvious'; ...
                     'minMot(1): in several repeats smeared spikes can be detected'; ...
                     'minMot(3) - too noisy to be useful, and missing combinations'};
% there are 2 more movingbar protocols and a movingEdge protocol for this
% cell by they are not centered 



cn=7;
t4Cells(cn).dir = relDirNames(cn,:);
t4Cells(cn).singleBar(1).fileName = '/SingleBarDiagCorrProtocol20160505_14-28/protocolStruct20160505_14-28.mat';
t4Cells(cn).singleBar(1).pos = -6:6;
t4Cells(cn).flicker(1).fileName = '/FlickerBarDiagCorrProtocol20160505_14-58/protocolStruct20160505_14-58.mat';
t4Cells(cn).flicker(1).pos = -3:2;
t4Cells(cn).minMot(1).fileName = '/MinMotDiagCorrProtocol20160505_14-38/protocolStruct20160505_14-38.mat';
t4Cells(cn).minMot(1).fbPos = -3:1;
t4Cells(cn).minMot(1).sbPos = -3:1;
t4Cells(cn).minMot(1).barV = 'BB';
t4Cells(cn).minMot(2).fileName = '/MinMotDiagCorrProtocol20160505_14-49/protocolStruct20160505_14-49.mat';
t4Cells(cn).minMot(2).fbPos = [-6:-4, -1];
t4Cells(cn).minMot(2).sbPos = -1;
t4Cells(cn).minMot(2).barV = 'BB';
t4Cells(cn).minMov(1).fileName = '/MinMovingBarDiagCorrProtocol20160505_14-53/protocolStruct20160505_14-53.mat';
t4Cells(cn).minMov(1).pos = [-6,5; -5,-3; -5,-1; -2,0; -2,3; -2,5]; 
t4Cells(cn).movBar(1).fileName = '/MovingBarDiagCorrProtocol20160505_14-15/protocolStruct20160505_14-15.mat';
t4Cells(cn).movBar(2).fileName = '/MovingBarProtocol20160505_14-21/protocolStruct20160505_14-21.mat';
t4Cells(cn).movBar(2).oldProt = 1;
t4Cells(cn).movEdge(1).fileName = '/MovingEdgeDiagCorrProtocol20160505_14-23/protocolStruct20160505_14-23.mat';
t4Cells(cn).notes = {'flicker: good fits at slower flickers'; 'had to remove last 5 stims due to noise'; ...
                     'pos2 loses phase locking @40ms but there is a weak resp in single bar'; ...
                     'minMot(2): nice resp, control match signlebar very nicely'};


cn=8;
t4Cells(cn).dir = relDirNames(cn,:);
t4Cells(cn).singleBar(1).fileName = '/SingleBarDiagCorrProtocol20160510_11-41/protocolStruct20160510_11-41.mat';
t4Cells(cn).singleBar(1).pos = -8:8;
t4Cells(cn).singleBar(1).correct = 1;
t4Cells(cn).singleBar(2).fileName = '/SingleBarDiagCorrProtocol20160510_12-19/protocolStruct20160510_12-19.mat';
t4Cells(cn).singleBar(2).pos = -6:6;
t4Cells(cn).singleBar(2).barV = 'D';
t4Cells(cn).flicker(1).fileName = '/FlickerBarDiagCorrProtocol20160510_12-10/protocolStruct20160510_12-10.mat';
t4Cells(cn).flicker(1).pos = -6:2:2;
t4Cells(cn).minMot(1).fileName = '/MinMotDiagCorrProtocol20160510_11-53/protocolStruct20160510_11-53.mat';
t4Cells(cn).minMot(1).fbPos = -4:0;
t4Cells(cn).minMot(1).sbPos = -4:0;
t4Cells(cn).minMot(1).barV = 'BB';
t4Cells(cn).minMot(2).fileName = '/MinMotDiagCorrProtocol20160510_12-04/protocolStruct20160510_12-04.mat';
t4Cells(cn).minMot(2).fbPos = [0, 2:5];
t4Cells(cn).minMot(2).sbPos = 0;
t4Cells(cn).minMot(2).barV = 'BB';
t4Cells(cn).minMot(3).fileName = '/MinMotDiagCorrProtocol20160510_12-22/protocolStruct20160510_12-22.mat';
t4Cells(cn).minMot(3).fbPos = -8:2:0;
t4Cells(cn).minMot(3).sbPos = -6:2:0;
t4Cells(cn).minMot(3).barV = 'DB';
t4Cells(cn).minMov(1).fileName = '/MinMovingBarDiagCorrProtocol20160510_12-14/protocolStruct20160510_12-14.mat';
t4Cells(cn).minMov(1).pos = [-8,0; -8,6; -6,0; -4,0; 0,5; 2,5]; 
t4Cells(cn).movBar(1).fileName = '/MovingBarDiagCorrProtocol20160510_11-36/protocolStruct20160510_11-36.mat';
t4Cells(cn).movBar(2).fileName = '/MovingBarDiagCorrProtocol20160510_12-38/protocolStruct20160510_12-38.mat';
t4Cells(cn).movBar(2).width = [1,2,4];
t4Cells(cn).movBar(3).fileName = '/MovingBarDiagCorrProtocol20160510_12-51/protocolStruct20160510_12-51.mat';
t4Cells(cn).movBar(3).height = [5,9,13];
t4Cells(cn).movEdge(1).fileName = '/MovingEdgeDiagCorrProtocol20160510_12-25/protocolStruct20160510_12-25.mat';
t4Cells(cn).grating(1).fileName = '/GratingProtocol20160510_12-30/protocolStruct20160510_12-30.mat';
t4Cells(cn).figGrd(1).fileName = '/MovingFigGrdDiagCorrProtocol20160510_12-43/protocolStruct20160510_12-43.mat';
t4Cells(cn).figGrd(2).fileName = '/MovingFigGrdDiagCorrProtocol20160510_12-46/protocolStruct20160510_12-46.mat';
t4Cells(cn).figGrd(3).fileName = '/MovingFigGrdDiagCorrProtocol20160510_12-55/protocolStruct20160510_12-55.mat';
t4Cells(cn).notes = {'flicker: good fits at slower flickers'; 'beautiful example for leading edge'; ...
                     'pos -4 and -6 both evoke resp in single bar 40ms but not in flicker'; ...
                     'also, pos -2 show no entrainment in 20ms but pos 0 and 2 do - rare'; ...
                     'minMot(2): control resp very clean (similar to singleBar'; ...
                     'minMot(3): some comb are missing due to bug in code, plus datais sup when compared to singleBar resp'; ...
                     'possibly not useful - though one position shows complete inhibition'};


cn=9;
t4Cells(cn).dir = relDirNames(cn,:);
t4Cells(cn).singleBar(1).fileName = '/SingleBarDiagCorrProtocol20160516_11-53/protocolStruct20160516_11-53.mat';
t4Cells(cn).singleBar(1).pos = -8:8;
t4Cells(cn).flicker(1).fileName = '/FlickerBarDiagCorrProtocol20160516_12-56/protocolStruct20160516_12-56.mat';
t4Cells(cn).flicker(1).pos = [-2:2, 4];
t4Cells(cn).flicker(2).fileName = '/FlickerBarDiagCorrProtocol20160516_13-01/protocolStruct20160516_13-01.mat';
t4Cells(cn).flicker(2).pos = [-4, -3, 0, 3, 5]; %2 separate protocols
t4Cells(cn).minMot(1).fileName = '/MinMotDiagCorrProtocol20160516_12-05/protocolStruct20160516_12-05.mat';
t4Cells(cn).minMot(1).fbPos = -1:3;
t4Cells(cn).minMot(1).sbPos = -1:3;
t4Cells(cn).minMot(1).barV = 'BB';
t4Cells(cn).minMot(2).fileName = '/MinMotDiagCorrProtocol20160516_12-15/protocolStruct20160516_12-15.mat';
t4Cells(cn).minMot(2).fbPos = [-5:-3, -1];
t4Cells(cn).minMot(2).sbPos = -1;
t4Cells(cn).minMot(2).barV = 'BB';
t4Cells(cn).minMot(3).fileName = '/MinMotDiagCorrProtocol20160516_12-19/protocolStruct20160516_12-19.mat';
t4Cells(cn).minMot(3).fbPos = [1,2,4,5];
t4Cells(cn).minMot(3).sbPos = [1,2,4,5];
t4Cells(cn).minMot(3).barV = 'BB';
t4Cells(cn).minMot(4).fileName = '/MinMotDiagCorrProtocol20160516_12-32/protocolStruct20160516_12-32.mat';
t4Cells(cn).minMot(4).fbPos = 1:5;
t4Cells(cn).minMot(4).sbPos = -1:3;
t4Cells(cn).minMot(4).barV = 'DB';
t4Cells(cn).minMov(1).fileName = '/MinMovingBarDiagCorrProtocol20160516_12-26/protocolStruct20160516_12-26.mat';
t4Cells(cn).minMov(1).pos = [-7,-1; -7,7; -5,-3; -5,-1; -1,3; -1,5];
t4Cells(cn).minMov(2).fileName = '/MinMovingBarDiagCorrProtocol20160516_12-51/protocolStruct20160516_12-51.mat';
t4Cells(cn).minMov(2).pos = [-5,3; -1,1; -1,3; -1,7; 2,4];
t4Cells(cn).movBar(1).fileName = '/MovingBarDiagCorrProtocol20160516_11-42/protocolStruct20160516_11-42.mat';
t4Cells(cn).movBar(2).fileName = '/MovingBarDiagCorrProtocol20160516_12-36/protocolStruct20160516_12-36.mat';
t4Cells(cn).movBar(2).width = [1,2,4];
t4Cells(cn).movBar(3).fileName = '/MovingBarDiagCorrProtocol20160516_12-46/protocolStruct20160516_12-46.mat';
t4Cells(cn).movBar(3).height = [5,9,13];
t4Cells(cn).movEdge(1).fileName = '/MovingEdgeDiagCorrProtocol20160516_11-48/protocolStruct20160516_11-48.mat';
t4Cells(cn).figGrd(1).fileName = '/MovingFigGrdDiagCorrProtocol20160516_12-41/protocolStruct20160516_12-41.mat';
t4Cells(cn).notes = {'flicker: good fits at slower flickers for both protocols'; ...
                     'pos 3 and 5 in 2nd flicker are most obvious'; ...
                     'minMot(3): some positions are missing due to a bug in the code, stim is non motion corrected'; ...
                     'response supression around the diagonal is still apparent (possibly both in sim and 40ms)'};


cn=10;
t4Cells(cn).dir = relDirNames(cn,:);
t4Cells(cn).singleBar(1).fileName = '/SingleBarDiagCorrProtocol20160517_11-45/protocolStruct20160517_11-45.mat';
t4Cells(cn).singleBar(1).pos = -5:5;
t4Cells(cn).minMot(1).fileName = '/MinMotDiagCorrProtocol20160517_11-54/protocolStruct20160517_11-54.mat';
t4Cells(cn).minMot(1).fbPos = -2:2;
t4Cells(cn).minMot(1).sbPos = -2:2;
t4Cells(cn).minMot(1).barV = 'BB';
t4Cells(cn).minMot(2).fileName = '/MinMotDiagCorrProtocol20160517_12-04/protocolStruct20160517_12-04.mat'; % data only to stim 73
t4Cells(cn).minMot(2).fbPos = [-4:-2, 0];
t4Cells(cn).minMot(2).sbPos = 0;
t4Cells(cn).minMot(2).barV = 'BB';
t4Cells(cn).minMot(3).fileName = '/MinMotDiagCorrProtocol20160517_12-06/protocolStruct20160517_12-06.mat';
t4Cells(cn).minMot(3).fbPos = [-4:-2, 0]; % some noise at the end of 2 so repeated 
t4Cells(cn).minMot(3).sbPos = 0;
t4Cells(cn).minMot(3).barV = 'BB';
t4Cells(cn).minMot(4).fileName = '/protocolStruct_combMinMot2And3.mat';
t4Cells(cn).minMot(4).fbPos = [-4:-2, 0]; % some noise at the end of 2 so repeated 
t4Cells(cn).minMot(4).sbPos = 0;
t4Cells(cn).minMot(4).barV = 'BB';
t4Cells(cn).minMov(1).fileName = '/MinMovingBarDiagCorrProtocol20160517_12-11/protocolStruct20160517_12-11.mat';
t4Cells(cn).minMov(1).pos = [-5,0; -5,5; -3,0; -1,1; 0,3; 0,5]; 
t4Cells(cn).movBar(1).fileName = '/MovingBarDiagCorrProtocol20160517_11-34/protocolStruct20160517_11-34.mat';
t4Cells(cn).movEdge(1).fileName = '/MovingEdgeDiagCorrProtocol20160517_11-40/protocolStruct20160517_11-40.mat';
t4Cells(cn).notes = {'minMot(1): spikes can be seen in some of the repeats'; ...
                     'solid responses even at 20ms; great data'; ...
                     'minMot(2)&(3): 2 was noisy so was repeated, they are combined into one protocolStruct in the parent directory'; ...
                     'FileName: protocolStruct_combMinMot2And3.mat (now minMot4), results are very clean'};



cn=11;
t4Cells(cn).dir = relDirNames(cn,:);
t4Cells(cn).singleBar(1).fileName = '/SingleBarDiagCorrProtocol20160518_17-57/protocolStruct20160518_17-57.mat';
t4Cells(cn).singleBar(1).pos = -5:5;
t4Cells(cn).flicker(1).fileName = '/FlickerBarDiagCorrProtocol20160518_18-23/protocolStruct20160518_18-23.mat';
t4Cells(cn).flicker(1).pos = -4:2;
t4Cells(cn).minMot(1).fileName = '/MinMotDiagCorrProtocol20160518_18-04/protocolStruct20160518_18-04.mat';
t4Cells(cn).minMot(1).fbPos = -2:2;
t4Cells(cn).minMot(1).sbPos = -2:2;
t4Cells(cn).minMot(1).barV = 'BB';
t4Cells(cn).minMot(2).fileName = '/MinMotDiagCorrProtocol20160518_18-15/protocolStruct20160518_18-15.mat';
t4Cells(cn).minMot(2).fbPos = [-5:-3, -1];
t4Cells(cn).minMot(2).sbPos = -1;
t4Cells(cn).minMot(2).barV = 'BB';
t4Cells(cn).minMot(3).fileName = '/MinMotDiagCorrProtocol20160518_18-58/protocolStruct20160518_18-58.mat';
t4Cells(cn).minMot(3).fbPos = -1:3;
t4Cells(cn).minMot(3).sbPos = -2:1;
t4Cells(cn).minMot(3).barV = 'DB';
t4Cells(cn).minMov(1).fileName = '/MinMovingBarDiagCorrProtocol20160518_18-18/protocolStruct20160518_18-18.mat';
t4Cells(cn).minMov(1).pos = [-5,-3; -5,0; -5,4; -2,0; 0,2; 0,4]; 
t4Cells(cn).minMov(2).fileName = '/MinMovingBarDiagCorrProtocol20160518_18-48/protocolStruct20160518_18-48.mat';
t4Cells(cn).minMov(2).pos = [-3,0; -2,1; -1,2; 0,3];
t4Cells(cn).movBar(1).fileName = '/MovingBarDiagCorrProtocol20160518_17-46/protocolStruct20160518_17-46.mat';
t4Cells(cn).movBar(2).fileName = '/MovingBarDiagCorrProtocol20160518_18-42/protocolStruct20160518_18-42.mat';
t4Cells(cn).movBar(2).width = [1,2,4];
t4Cells(cn).movBar(3).fileName = '/MovingBarDiagCorrProtocol20160518_18-45/protocolStruct20160518_18-45.mat';
t4Cells(cn).movBar(3).height = [5,9,13];
t4Cells(cn).movEdge(1).fileName = '/MovingEdgeDiagCorrProtocol20160518_17-52/protocolStruct20160518_17-52.mat';
t4Cells(cn).grating(1).fileName = '/GratingProtocol20160518_18-34/protocolStruct20160518_18-34.mat';
t4Cells(cn).grating(2).fileName = '/GratingProtocol20160518_18-53/protocolStruct20160518_18-53.mat';
t4Cells(cn).grating(2).revPhi = 1;
t4Cells(cn).figGrd(1).fileName = '/MovingFigGrdDiagCorrProtocol20160518_18-28/protocolStruct20160518_18-28.mat';
t4Cells(cn).notes = {'flicker: good fits at slower flickers though some useful also in 40ms'; ...
                     'leading pos 1, 2 show weaker entrainment in 40ms (but also lagging -4), perhaps not far enough from center'; ...
                     'clear resp in 20ms in center (even pos 1)'; ...
                     'minMot(1): clear resp also in 20ms'; ...
                     'minMot(2): clean resp and clear effect of FBStat=1'; ...
                     'minMot(3): not speed corrected, one pos delay as constant (bug)';
                     'data is beatuful: no effect in 80ms and strong suppresion around the diagonal - both sides'};



cn=12;
t4Cells(cn).dir = relDirNames(cn,:);
t4Cells(cn).singleBar(1).fileName = '/SingleBarDiagCorrProtocol20160520_15-47/protocolStruct20160520_15-47.mat';
t4Cells(cn).singleBar(1).pos = -5:5;
t4Cells(cn).singleBar(1).correct = 1;
t4Cells(cn).singleBar(2).fileName = '/SingleBarDiagCorrProtocol20160520_15-59/protocolStruct20160520_15-59.mat';
t4Cells(cn).singleBar(2).pos = -8:-4; 
t4Cells(cn).flicker(1).fileName = '/FlickerBarDiagCorrProtocol20160520_16-16/protocolStruct20160520_16-16.mat';
t4Cells(cn).flicker(1).pos = -4:2;
t4Cells(cn).minMot(1).fileName = '/MinMotDiagCorrProtocol20160520_16-03/protocolStruct20160520_16-03.mat';
t4Cells(cn).minMot(1).fbPos = -3:1;
t4Cells(cn).minMot(1).sbPos = -3:1;
t4Cells(cn).minMot(1).barV = 'BB';
t4Cells(cn).minMot(2).fileName = '/MinMotDiagCorrProtocol20160520_16-13/protocolStruct20160520_16-13.mat';
t4Cells(cn).minMot(2).fbPos = [0, 2:4];
t4Cells(cn).minMot(2).sbPos = 0;
t4Cells(cn).minMot(2).barV = 'BB';
t4Cells(cn).minMot(3).fileName = '/MinMotDiagCorrProtocol20160520_16-44/protocolStruct20160520_16-44.mat';
t4Cells(cn).minMot(3).fbPos = [-5,-4,-1,0]; 
t4Cells(cn).minMot(3).sbPos = [-5,-4,-1,0];
t4Cells(cn).minMot(3).barV = 'BB';
t4Cells(cn).minMot(4).fileName = '/MinMotDiagCorrProtocol20160520_16-52/protocolStruct20160520_16-52.mat';
t4Cells(cn).minMot(4).fbPos = -1:4; 
t4Cells(cn).minMot(4).sbPos = -3:1;
t4Cells(cn).minMot(4).barV = 'DB';
t4Cells(cn).minMov(1).fileName = '/MinMovingBarDiagCorrProtocol20160520_16-21/protocolStruct20160520_16-21.mat';
t4Cells(cn).minMov(1).pos = [-5,5; -4,-2; -4,0; -1,1; -1,3; 2,4]; 
t4Cells(cn).movBar(1).fileName = '/MovingBarDiagCorrProtocol20160520_15-41/protocolStruct20160520_15-41.mat';
t4Cells(cn).movBar(2).fileName = '/MovingBarDiagCorrProtocol20160520_16-32/protocolStruct20160520_16-32.mat';
t4Cells(cn).movBar(2).width = [1,2,4];
t4Cells(cn).movBar(3).fileName = '/MovingBarDiagCorrProtocol20160520_16-57/protocolStruct20160520_16-57.mat';
t4Cells(cn).movBar(3).height = [5,9,13];
t4Cells(cn).movEdge(1).fileName = '/MovingEdgeDiagCorrProtocol20160520_15-54/protocolStruct20160520_15-54.mat';
t4Cells(cn).grating(1).fileName = '/GratingProtocol20160520_16-35/protocolStruct20160520_16-35.mat';
t4Cells(cn).grating(2).fileName = '/GratingProtocol20160520_17-02/protocolStruct20160520_17-02.mat';
t4Cells(cn).grating(2).revPhi = 1;
t4Cells(cn).figGrd(1).fileName = '/MovingFigGrdDiagCorrProtocol20160520_16-26/protocolStruct20160520_16-26.mat';
t4Cells(cn).notes = {'flicker: good fits at slower flickers'; 'pos -3,-4 show weak resp in singlebar 40ms but none in flicker'; ...
                     'entrainment is clear in center positions'; ...
                     'minMot(1): clear resp even at 20ms'; ...
                     'singleBar(2): added more positions on the edge of excitatory side - not that useful'; ...
                     'minMot(2): responses to control bar are sub-singlebar across all speeds (maybe useless)'; ...
                     'plus position 2 gives a much stronger response than expected for first bar'; ...
                     'minMot(4): dark bar shuts off the response completely when presented close in space and sim in time'};

cn=13;
t4Cells(cn).dir = relDirNames(cn,:);
t4Cells(cn).singleBar(1).fileName = '/SingleBarDiagCorrProtocol20160523_12-08/protocolStruct20160523_12-08.mat';
t4Cells(cn).singleBar(1).pos = -8:8;
t4Cells(cn).flicker(1).fileName = '/FlickerBarDiagCorrProtocol20160523_12-34/protocolStruct20160523_12-34.mat';
t4Cells(cn).flicker(1).pos = -5:0;
t4Cells(cn).minMot(1).fileName = '/MinMotDiagCorrProtocol20160523_12-18/protocolStruct20160523_12-18.mat';
t4Cells(cn).minMot(1).fbPos = -5:-1;
t4Cells(cn).minMot(1).sbPos = -5:-1;
t4Cells(cn).minMot(1).barV = 'BB';
t4Cells(cn).minMot(2).fileName = '/MinMotDiagCorrProtocol20160523_12-29/protocolStruct20160523_12-29.mat';
t4Cells(cn).minMot(2).fbPos = [-1, 1:4];
t4Cells(cn).minMot(2).sbPos = -1;
t4Cells(cn).minMot(2).barV = 'BB';
t4Cells(cn).minMot(3).fileName = '/MinMotDiagCorrProtocol20160523_12-45/protocolStruct20160523_12-45.mat';
t4Cells(cn).minMot(3).fbPos = -6:-1;
t4Cells(cn).minMot(3).sbPos = -4:-1;
t4Cells(cn).minMot(3).barV = 'DB';
t4Cells(cn).minMov(1).fileName = '/MinMovingBarDiagCorrProtocol20160523_12-39/protocolStruct20160523_12-39.mat';
t4Cells(cn).minMov(1).pos = [-8,-0; -8,5; -5,0; -4,-1; -3,0; -1,4; 1,4];
t4Cells(cn).minMov(2).fileName = '/MinMovingBarDiagCorrProtocol20160523_13-14/protocolStruct20160523_13-14.mat';
t4Cells(cn).minMov(2).pos = [-6,-3; -4,-1; -2,1; 0,3];
t4Cells(cn).movBar(1).fileName = '/MovingBarDiagCorrProtocol20160523_11-57/protocolStruct20160523_11-57.mat';
t4Cells(cn).movBar(2).fileName = '/MovingBarDiagCorrProtocol20160523_12-54/protocolStruct20160523_12-54.mat';
t4Cells(cn).movBar(2).width = [1,2,4];
t4Cells(cn).movBar(3).fileName = '/MovingBarDiagCorrProtocol20160523_13-06/protocolStruct20160523_13-06.mat';
t4Cells(cn).movBar(3).height = [5,9,13];
t4Cells(cn).movEdge(1).fileName = '/MovingEdgeDiagCorrProtocol20160523_12-03/protocolStruct20160523_12-03.mat';
t4Cells(cn).grating(1).fileName = '/GratingProtocol20160523_12-58/protocolStruct20160523_12-58.mat';
t4Cells(cn).grating(2).fileName = '/GratingProtocol20160523_13-10/protocolStruct20160523_13-10.mat';
t4Cells(cn).grating(2).revPhi = 1;
t4Cells(cn).figGrd(1).fileName = '/MovingFigGrdDiagCorrProtocol20160523_12-50/protocolStruct20160523_12-50.mat';
t4Cells(cn).notes = {'flicker: only slowest looks good'; '80 responds, but seems weaker than it can be'; ...
                     'minMot(1): responses are a bit weak and flitered, still relatively good agreement'; ...
                     'minMot(2): resp a bit sub compared to singlebar, still see stronger sublinearity at low speeds'; ...
                     'minMot(3): dark bar completely abolishes resp close to bright (on both side) in sim'};

cn=14;
t4Cells(cn).dir = relDirNames(cn,:);
t4Cells(cn).singleBar(1).fileName = '/SingleBarDiagCorrProtocol20160525_11-31/protocolStruct20160525_11-31.mat';
t4Cells(cn).singleBar(1).pos = -8:8;
t4Cells(cn).flicker(1).fileName = '/FlickerBarDiagCorrProtocol20160525_12-02/protocolStruct20160525_12-02.mat';
t4Cells(cn).flicker(1).pos = -6:2:4;
t4Cells(cn).minMot(1).fileName = '/MinMotDiagCorrProtocol20160525_11-42/protocolStruct20160525_11-42.mat';
t4Cells(cn).minMot(1).fbPos = -3:1;
t4Cells(cn).minMot(1).sbPos = -3:1;
t4Cells(cn).minMot(1).barV = 'BB';
t4Cells(cn).minMot(2).fileName = '/MinMotDiagCorrProtocol20160525_11-52/protocolStruct20160525_11-52.mat';
t4Cells(cn).minMot(2).fbPos = [0, 2:5];
t4Cells(cn).minMot(2).sbPos = 0;
t4Cells(cn).minMot(2).barV = 'BB';
t4Cells(cn).minMot(3).fileName = '/MinMotDiagCorrProtocol20160525_12-05/protocolStruct20160525_12-05.mat';
t4Cells(cn).minMot(3).fbPos = -6:-3;
t4Cells(cn).minMot(3).sbPos = -6:-3;
t4Cells(cn).minMot(3).barV = 'BB';
t4Cells(cn).minMov(1).fileName = '/MinMovingBarDiagCorrProtocol20160525_11-57/protocolStruct20160525_11-57.mat';
t4Cells(cn).minMov(1).pos = [-7,1; -7,5; -5,1; -3,1; 1,5; 3,5];
t4Cells(cn).minMov(2).fileName = '/MinMovingBarDiagCorrProtocol20160525_12-17/protocolStruct20160525_12-17.mat';
t4Cells(cn).minMov(2).pos = [-6,-2; -6,4; -4,0; -2,2; 0,4];
t4Cells(cn).movBar(1).fileName = '/MovingBarDiagCorrProtocol20160525_11-26/protocolStruct20160525_11-26.mat';
t4Cells(cn).movEdge(1).fileName = '/MovingEdgeDiagCorrProtocol20160525_12-12/protocolStruct20160525_12-12.mat';
t4Cells(cn).notes = {'flicker: nice resp even in 40ms'; ...
                     'pos -4 is a bit of an abberation - resp is entrained and stringer than center'; ...
                     'minMot(1): almost no resp at 20ms - though combinations show something'; ...
                     'minMot(3): good data at all times'; ...
                     'minMot(2): comtrol looks good, esp. at lower speeds'; ...
                     'minMov(2): noisy but still may have useful data'};



cn=15;
t4Cells(cn).dir = relDirNames(cn,:);
t4Cells(cn).singleBar(1).fileName = '/SingleBarDiagCorrProtocol20160525_16-27/protocolStruct20160525_16-27.mat';
t4Cells(cn).singleBar(1).pos = -6:6;
t4Cells(cn).minMot(1).fileName = '/MinMotDiagCorrProtocol20160525_16-35/protocolStruct20160525_16-35.mat';
t4Cells(cn).minMot(1).fbPos = -2:2;
t4Cells(cn).minMot(1).sbPos = -2:2;
t4Cells(cn).minMot(1).barV = 'BB';
t4Cells(cn).movBar(1).fileName = '/MovingBarDiagCorrProtocol20160525_16-21/protocolStruct20160525_16-21.mat';
t4Cells(cn).movEdge(1).fileName = '/MovingEdgeDiagCorrProtocol20160525_12-12/protocolStruct20160525_12-12.mat';
t4Cells(cn).notes = {'minMot(1): nice resp even in 20ms'}; 


cn=16;
t4Cells(cn).dir = relDirNames(cn,:);
t4Cells(cn).singleBar(1).fileName = '/SingleBarDiagCorrProtocol20160526_16-01/protocolStruct20160526_16-01.mat';
t4Cells(cn).singleBar(1).pos = -8:8;
t4Cells(cn).flicker(1).fileName = '/FlickerBarDiagCorrProtocol20160526_16-32/protocolStruct20160526_16-32.mat';
t4Cells(cn).flicker(1).pos = -3:3;
t4Cells(cn).minMot(1).fileName = '/MinMotDiagCorrProtocol20160526_16-12/protocolStruct20160526_16-12.mat';
t4Cells(cn).minMot(1).fbPos = -2:2;
t4Cells(cn).minMot(1).sbPos = -2:2;
t4Cells(cn).minMot(1).barV = 'BB';
t4Cells(cn).minMot(2).fileName = '/MinMotDiagCorrProtocol20160526_16-23/protocolStruct20160526_16-23.mat';
t4Cells(cn).minMot(2).fbPos = [1, 3,4,5];
t4Cells(cn).minMot(2).sbPos = 1;
t4Cells(cn).minMot(2).barV = 'BB';
t4Cells(cn).minMov(1).fileName = '/MinMovingBarDiagCorrProtocol20160526_16-27/protocolStruct20160526_16-27.mat';
t4Cells(cn).minMov(1).pos = [-3,-1; -3,0; -3,5; -1,2; 0,2; 0,3; 3,5];
t4Cells(cn).movBar(1).fileName = '/MovingBarDiagCorrProtocol20160526_15-42/protocolStruct20160526_15-42.mat';
t4Cells(cn).movBar(2).fileName = '/MovingBarDiagCorrProtocol20160526_15-54/protocolStruct20160526_15-54.mat';
t4Cells(cn).movBar(2).height = [5,9,13];
t4Cells(cn).movEdge(1).fileName = '/MovingEdgeDiagCorrProtocol20160526_15-47/protocolStruct20160526_15-47.mat';
t4Cells(cn).notes = {'flicker: only slowest looks good and even that is weak'; ...
                     'still, no resp in pos -3 to -1 in flicker 40ms'; ...
                     'minMot(1): nice data, resp in 20 are weak but noticable'; ...
                     'possible includes weak ext zone and strong ext zone'; ...
                     'minMot(2): beatiful data'};

cn=17;
t4Cells(cn).dir = relDirNames(cn,:);
t4Cells(cn).singleBar(1).fileName = '/SingleBarDiagCorrProtocol20160527_11-37/protocolStruct20160527_11-37.mat';
t4Cells(cn).singleBar(1).pos = -6:6;
t4Cells(cn).singleBar(1).correct = 1;
% second was done half way through to make sure there was no movement
t4Cells(cn).singleBar(2).fileName = '/SingleBarDiagCorrProtocol20160527_12-24/protocolStruct20160527_12-24.mat';
t4Cells(cn).singleBar(2).pos = -4:4; 
t4Cells(cn).flicker(1).fileName = '/FlickerBarDiagCorrProtocol20160527_12-03/protocolStruct20160527_12-03.mat';
t4Cells(cn).flicker(1).pos = -2:3;
t4Cells(cn).minMot(1).fileName = '/MinMotDiagCorrProtocol20160527_11-45/protocolStruct20160527_11-45.mat';
t4Cells(cn).minMot(1).fbPos = -1:3;
t4Cells(cn).minMot(1).sbPos = -1:3;
t4Cells(cn).minMot(1).barV = 'BB';
t4Cells(cn).minMot(2).fileName = '/MinMotDiagCorrProtocol20160527_11-56/protocolStruct20160527_11-56.mat';
t4Cells(cn).minMot(2).fbPos = -3:0;
t4Cells(cn).minMot(2).sbPos = 0;
t4Cells(cn).minMot(2).barV = 'BB';
t4Cells(cn).minMov(1).fileName = '/MinMovingBarDiagCorrProtocol20160527_12-00/protocolStruct20160527_12-00.mat';
t4Cells(cn).minMov(1).pos = [-3,-1; -3,4; -1,2; 0,2; 1,4; 2,4];
t4Cells(cn).movBar(1).fileName = '/MovingBarDiagCorrProtocol20160527_11-32/protocolStruct20160527_11-32.mat';
t4Cells(cn).movBar(2).fileName = '/MovingBarDiagCorrProtocol20160527_12-28/protocolStruct20160527_12-28.mat';
t4Cells(cn).movBar(2).width = [1,2,4];
t4Cells(cn).movEdge(1).fileName = '/MovingEdgeDiagCorrProtocol20160527_12-34/protocolStruct20160527_12-34.mat';
t4Cells(cn).figGrd(1).fileName = '/MovingFigGrdDiagCorrProtocol20160527_12-09/protocolStruct20160527_12-09.mat';
t4Cells(cn).notes = {'flicker: nice response even in 20ms'; 'useful fits also in 40'; ...
                     'weaker entrainment in leading pos 2&3 - might be quantifiable'; ...
                     'minMot(1): a bit niosy but still good data. responses even at 20ms, esp. for combinations'; ...
                     'minMot(2): control stim is always sub singlebar, not sure useful'};

                 
%  protInds lists

relProtInds.movBar = {1; 1; ...
                      1; 1; 1; 3; 1; ...
                      1; 1; 1; 1; 1; ...
                      1; 1; 1; 1; 1};

                  
relProtInds.singleBar = {1; 1; ...
                         1; 1; 1; 2; 1; ...
                         1; 1; 1; 1; 1; ...
                         1; 1; 1; 1; 1};
                        
                        
relProtInds.minMotE = {1; 1; 1; 1; 1; ...
                       1; 1; 1; [1,3]; 1; ...
                       1; [1,3]; 1; [1,3]; 1; ...
                       1; 1}; 
                   
                   
relProtInds.minMotI = {2; 2; 2; [2,4]; 2; ...
                       2; 2; 2; 2; 4; ... % 4 since it is the combined file
                       2; 2; 2; 2; nan; ...
                       2; 2};
                   
relProtInds.minMotD = {nan; 3; 3; 3; 3; ...
                       3; nan; 3; 4; nan; ...
                       3; 4; 3; nan; nan; ...
                       nan; nan};




clear cn relDirNames
