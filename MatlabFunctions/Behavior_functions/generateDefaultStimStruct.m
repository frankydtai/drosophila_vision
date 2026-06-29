function stimStruct = generateDefaultStimStruct(arenaSize)

if nargin < 1
    arenaSize = [32, 96];
end

stimStruct.barSt = generateDefaultBarSt(arenaSize);
stimStruct.halfAmp = 10; 
stimStruct.center = [ceil(arenaSize(1)*2/3), floor(arenaSize(2)/2)];    
stimStruct.background = 'uniform';
stimStruct.bkgdProp = [1,1,1];
stimStruct.barSize = [(8:2:16)', ones(5,1) *8];
stimStruct.barStable = 1;
stimStruct.expand = 'number';



end