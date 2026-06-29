function arenaChecker = generateCheckboard(arenaSize)

% generates an 8X8 checkrboard background

padX1 =[];

for ii=0:2:(arenaSize(1)/8-1)
    padX1 = [padX1, (1:8) + ii*8];    
end

padX2 = setdiff(1:arenaSize(1), padX1);

padY1 = [];
for ii=0:2:(arenaSize(2)/8-1)
    padY1 = [padY1, (1:8) + ii*8];
end
padY2 = setdiff(1:arenaSize(2), padY1);

[indX1, indY1] = meshgrid(padX1, padY1);
[indX2, indY2] = meshgrid(padX2, padY2);

arenaChecker = zeros(arenaSize);
arenaChecker(indX1, indY1) = 1;
arenaChecker(indX2, indY2) = 1;



end