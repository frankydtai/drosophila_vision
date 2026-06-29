function defaultBarSt = generateDefaultBarSt(arenaSize)

if nargin < 1
    arenaSize = [32, 96];
end


defaultBarSt.center = [floor(arenaSize(1)/2), ceil(arenaSize(2)/3)];
defaultBarSt.size = [5, 5];
defaultBarSt.proportion = [1, 0, 0];
defaultBarSt.range = [0, 7];
defaultBarSt.arrangement = 'random';
defaultBarSt.seed = [];


end