
function indsCell = divideSquareToCols(sqDim, sqOri)

% function indsCell = divideSquareToCols(sqDim, sqOri)
%
% this function divides a rotated square to consequtive diagonal columnns
% based on the given orientation. 
% square center is [0,0] and indices can later be applied to any size of
% odd numbered square matrix.
% deals also with non diagonal squares for ease of use
%
% INPUT
%
% sqDim -       dimension of the square. if rotated by a multiple of 45
%               will be taken as the diagonal otherwise, it is the side  
% sqOri -       orientation of the square that conforms with the rest of
%               the functions (0:7 multiples of 45 degrees)
%
% OUTPUT
%
% indsCell -    [x,y] indices for each column in the square. order is based
%               on direction of motion in rest of functions, meaning:
%               sqOri 0: first is left-most
%               sqOri 1: first is top-left
%               sqOri 2: first is top-most
%               sqOri 3: first is top-right and so on 
% indsCell is a cell since for the diagonals columns are of uneven length
 

assert(ismember(sqOri, 0:7), 'square orientation should be between 0-3')
assert(round(sqDim)==sqDim, 'sqDim should be an integer')

halfSqDim = floor(sqDim/2);
assert(halfSqDim < sqDim/2, 'sqDim must be an odd number')

indsCell = cell(1, sqDim);

flipInd = 0;
if sqOri > 3
    sqOri = sqOri - 4;
    flipInd = 1;
end

switch sqOri
    
    case 2
        
        firstCol = repmat([-halfSqDim, 0], sqDim, 1) + ([0,1]' * (-halfSqDim:halfSqDim))';
        indsCell{1} = firstCol;
        
        for ii=1:sqDim-1
            indsCell{ii+1} = firstCol + repmat([ii,0], sqDim, 1);
        end
        
        indsCell = fliplr(indsCell);% so positions would conform w/ orientation as it has been used
        
    case 0
        
        firstCol = repmat([0, halfSqDim], sqDim, 1) + ([1, 0]' * (halfSqDim:-1:-halfSqDim))';
        indsCell{1} = firstCol;
        
        for ii=1:sqDim-1
            indsCell{ii+1} = firstCol - repmat([0,ii], sqDim, 1);
        end
        indsCell = fliplr(indsCell);% so positions would conform w/ orientation as it has been used
        
    case 1
        
        indsCell{1} = [(-halfSqDim:0)', (0:halfSqDim)'];
        
        for ii=1:sqDim-1
            
            relCol = indsCell{ii};
            if rem(ii,2) > 0
                newCol = relCol + repmat([1, 0], halfSqDim+1, 1);
                indsCell{ii+1} = newCol(1:end-1,:);
            else
                newColR = relCol + repmat([1, 0], halfSqDim, 1);
                newColD = relCol + repmat([0, -1], halfSqDim, 1);
                indsCell{ii+1} = union(newColR, newColD, 'rows');
            end
            
        end
        indsCell = fliplr(indsCell); % so positions would conform w/ orientation as it has been used 
        
    case 3
        
       indsCell{1} = [(0:halfSqDim)', (halfSqDim:-1:0)']; 
        
        for ii=1:sqDim-1
            
            relCol = indsCell{ii};
            if rem(ii,2) > 0
                newCol = relCol + repmat([-1, 0], halfSqDim+1, 1);
                indsCell{ii+1} = newCol(2:end,:);
            else
                newColL = relCol + repmat([-1, 0], halfSqDim, 1);
                newColD = relCol + repmat([0, -1], halfSqDim, 1);
                indsCell{ii+1} = union(newColL, newColD, 'rows');
            end
            
        end    
         
end

% in case orientation was bigger than 3, flips order
if flipInd
   indsCell = fliplr(cellfun(@flipud, indsCell, 'uniformoutput', 0)); % filps up-down also for completion sake
end


end

            