function randInds = randomizeMatrixInds2(matSize, randDim)

% function randInds = randomizeMatrixInds2(matSize, randDim)
%
% Same as randomizeMatrixInds only for 3D matrices 
%
% INPUT
%
% matSize -     1X3 vector describing the size of the 3D matrix to be randomized
% randDim -     binary vector tagging which dimension to randomize. Should be same length as matSize 
%
% OUTPUT
% randInds -    (MXN)X2 vector of indices into the required matrix (x,y
%               coordinates of the given matrix size

% couldn't think of a more elegent way of doing this (that would still
% allow to randomize independently)

if size(matSize) ~= 3
    error('matSize should be 3D')
end

if length(matSize) ~= length(randDim)
    error('matSize and randDim should be of equal length')
end

if ~prod(ismember((randDim), [0,1])) %verifies that only 0,1 are the input
    error('randDim should be a logical array')
end

randInds = zeros([prod(matSize), 3]);

nonRandMat = randInds;
[xx, yy, zz] = ind2sub(matSize, 1:prod(matSize));
nonRandMat = [xx', yy', zz'];


if isequal(randDim, [0,0,0]) % does not randomize
        
        randInds = nonRandMat;
        
elseif isequal(randDim, [1,0,0])
        xx=[];
        for ii=1:prod(matSize(2:3))
            xx = [xx, randperm(matSize(1))];
        end
        
        randInds(:,1) = xx;
        randInds(:,2:3) = nonRandMat(:,2:3);
        
elseif isequal(randDim, [0,1,0])
        
        tempyy=[];
        for ii=1:matSize(3)
            tempyy = [tempyy, randperm(matSize(2))];
        end
        yy = reshape(repmat(tempyy, matSize(1),1), [], 1);
        randInds(:,2) = yy;
        randInds(:,[1,3]) = nonRandMat(:,[1,3]);
        

elseif isequal(randDim, [0,0,1])
        
        tempzz = randperm(matSize(3));
        
        zz = reshape(repmat(tempzz, prod(matSize(1:2)),1), [], 1);
        randInds(:,3) = zz;
        randInds(:,[1,2]) = nonRandMat(:,[1,2]);

% NOTE!!        
% When 2 dimension are random the non random dimension goes through all of
% their permutations before changing value <change if needed>
elseif isequal(randDim, [1,1,0]) 
        
        augInds = [];
        for ii=1:matSize(3)
            tempInds = find(nonRandMat(:,3) == ii);
            permMetaInds = randperm(length(tempInds));
            augInds = [augInds, tempInds(permMetaInds)'];
        end
        
        [xx,yy,zz] = ind2sub(matSize, augInds);
        randInds = [xx',yy',zz'];
        
elseif isequal(randDim, [1,0,1])
        
        augInds = [];
        for ii=1:matSize(2)
            tempInds = find(nonRandMat(:,2) == ii);
            permMetaInds = randperm(length(tempInds));
            augInds = [augInds, tempInds(permMetaInds)'];
        end
        
        [xx,yy,zz] = ind2sub(matSize, augInds);
        randInds = [xx',yy',zz'];
   
        
        
elseif isequal(randDim, [0,1,1])
        
        augInds = [];
        for ii=1:matSize(1)
            tempInds = find(nonRandMat(:,1) == ii);
            permMetaInds = randperm(length(tempInds));
            augInds = [augInds, tempInds(permMetaInds)'];
        end
        
        [xx,yy,zz] = ind2sub(matSize, augInds);
        randInds = [xx',yy',zz'];
    
        
        
elseif isequal(randDim, [1,1,1])
        
        tempInds = randperm(prod(matSize));
        
        [xx,yy,zz] = ind2sub(matSize, tempInds);
        randInds = [xx',yy',zz'];
        secShuff = randperm(size(randInds, 1));
        randInds = randInds(secShuff, :);
end
        
            
        


end