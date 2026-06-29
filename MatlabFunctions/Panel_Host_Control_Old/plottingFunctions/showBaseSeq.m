function showBaseSeq(baseSeqMatrix, gsLevel)

% Basic function to show baseGratingSeq
% INPUT 
% baseSeqMatrix - the output from generateGratingBaseSeq


if nargin < 2
    gsLevel = 3;
end

maxval = 2^gsLevel-1;

% can change if needed
times = 1; % how many times to repeat the whole sequnce
freq = 60; % at what freqeuncy

clf
figure(gcf)
set(gcf, 'position', [ 540   600   970   300])
cmap = zeros(maxval,3);
cmap(:,2) = linspace(0,1,maxval);
colormap(cmap)

% to mimic arena with 3 missing panels on each side
baseSeqMatrix(1:8, [1:15, 81:96], :) = 0;
baseSeqMatrix(8:16, [1:7, 89:96], :) = 0;



imh = image('CData', baseSeqMatrix(:,:,1), 'CDataMapping', 'direct');
% draws red lines for panels

set(gca,'xtick', [], 'ytick', [], 'clim', [0,1], ...
    'xlim', [0, 97], 'ylim', [0, 33], 'color', 'none', ...
    'position', [0.05, 0.1, 0.9, 0.8])
%set(imh, 'erasemode', 'none')
axis equal off

for kk=1:times
    for ii=1:size(baseSeqMatrix, 3)
        set(imh,'CData',flipud(baseSeqMatrix(:,:,ii)),'CDataMapping', 'direct')
        drawnow
        pause(1/freq);
    end
end



end