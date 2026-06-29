function maskPosCell = makeLinearTrajectoriesGrid(inputStruct, errorFlag)

% function maskPosCell = makeLinearTrajectoriesGrid(inputStruct)
%
% This function generates a cell array of trajectories to be used as mask
% position for linearly moving objects. If part of a trajectory are outside
% of the arena the function can either warn and crop trajectory or abort.
%
% INPUT
%
% inputStruct -         Structure containing the following fields
% .center -             1X2 vector. Point around which trajectories will revolve
% .direction -          1XN vector (0-7). Specified the same way as orientation (45 degree steps 0 being left to right)
% .length -             Non-negative Integer. Applied to all direction 
% .numTraj -            Non-negative Integer. Number of trajectories to generate in
%                       every direction. If even no trajectory will actually pass through
%                       center. Applied to directions  
% .stepSize -           Non-negative Integer. Determines displacement size between
%                       trajectories of the same directions (Always perpendicular to the
%                       direction of motion. 
% errorFlag -           (optional) Logical. If True function will abort when a trajectory is out
%                       of the arena. Default is False where there will be a warning and trajectory will be
%                       cropped
% OUTPUT
% maskPosCell -         Cell array of mask positions. 
%
% NOTE! Functio assumes a [96,32], pixel arena. 
% All units should be in arena pixels and will be rounded
% Therefore, it is advisable to use odd number for numTraj and lenghts
% (reduces weird rounding errors)

arenaSize = [96, 32];

assert(isfield(inputStruct, 'center'), 'inputStruct must have a center field')
cent = reshape(inputStruct.center, 1, []);
assert(length(cent) == 2, 'Center should be a 1X2 vector') 
assert(cent(1) <= arenaSize(1) && cent(2) <= arenaSize(2), 'Center should be the X Y position in arena coordinates')

assert(isfield(inputStruct, 'direction'), 'inputStruct must have a direction field')
direcs = inputStruct.direction;
assert(prod(ismember(direcs, 0:7)) == 1, 'Directions should be given as a 1XN vector of integers between 0 to 7')

assert(isfield(inputStruct, 'length'), 'inputStruct must have a length field')
len = round(inputStruct.length);
assert(len > 0, 'Length should be a non negative integer')

assert(isfield(inputStruct, 'numTraj'), 'inputStruct must have a numTraj field')
numTraj = round(inputStruct.numTraj);
assert(numTraj > 0, 'number of trajectories should be a non negative integer')

assert(isfield(inputStruct, 'stepSize'), 'inputStruct must have a stepSize field')
stepSiz = round(inputStruct.stepSize);
assert(stepSiz > 0, 'step size should be a non negative integer')

if nargin < 2
    errorFlag = 0;
end

trajDisp = floor(numTraj/2);
maskPosCell = cell(length(direcs), 1);


for ii=1:length(direcs)
    switch direcs(ii)
        case 0 % left right movement
            displacment = [0, stepSiz];
            parTraj.startPos = cent - floor([len/2, 0]);
            parTraj.endPos = cent + floor([len/2, 0]);
        case 4 % right left movement
            displacment = [0, stepSiz];
            parTraj.startPos = cent + floor([len/2, 0]);
            parTraj.endPos = cent - floor([len/2, 0]);
        case 2
            displacment = [stepSiz, 0];
            parTraj.startPos = cent + floor([0, len/2]);
            parTraj.endPos = cent - floor([0, len/2]);
        case 6
            displacment = [stepSiz, 0];
            parTraj.startPos = cent - floor([0, len/2]);
            parTraj.endPos = cent + floor([0, len/2]);
        case 1
            displacment = ceil([stepSiz, stepSiz]/sqrt(2));
            parTraj.startPos = cent + [floor(-len/(2*sqrt(2))), ceil(len/(2*sqrt(2)))];
            parTraj.endPos = cent - [floor(-len/(2*sqrt(2))), ceil(len/(2*sqrt(2)))];
        case 5
            displacment = ceil([stepSiz, stepSiz]/sqrt(2));
            parTraj.startPos = cent - [floor(-len/(2*sqrt(2))), ceil(len/(2*sqrt(2)))];
            parTraj.endPos = cent + [floor(-len/(2*sqrt(2))), ceil(len/(2*sqrt(2)))];
        case 3
            displacment = [ceil(stepSiz/sqrt(2)), floor(-stepSiz/sqrt(2))];
            parTraj.startPos = cent + ceil([len/(2*sqrt(2)), len/(2*sqrt(2))]);
            parTraj.endPos = cent - ceil([len/(2*sqrt(2)), len/(2*sqrt(2))]);
        case 7
            displacment = [ceil(stepSiz/sqrt(2)), floor(-stepSiz/sqrt(2))];
            parTraj.startPos = cent - ceil([len/(2*sqrt(2)), len/(2*sqrt(2))]);
            parTraj.endPos = cent + ceil([len/(2*sqrt(2)), len/(2*sqrt(2))]);
    end
    
    parTraj.numSteps = len;
    baseTraj = makeLinearTrajectory(parTraj);
    
    tempTrajCell = cell(2*trajDisp +1, 1);
    counter = 0;
    
    for jj = -trajDisp:trajDisp
        counter = counter+1;
        tempTrajCell{counter} = baseTraj + jj* repmat(displacment, len, 1);
    end
    
    if logical(rem(numTraj, 2))
        maskPosCell{ii} = tempTrajCell;
    else
        tempTrajCell = cellfun(@(x) x - repmat(floor(displacment/2), len, 1), tempTrajCell, 'uniformoutput', 0);
        maskPosCell{ii} = tempTrajCell(2:end);
    end
    
end
    
maskPosCell = vertcat(maskPosCell{:});

% checking if coordinates are out of range
for ii=1:length(maskPosCell)
    tempTraj = maskPosCell{ii};
    xCheck = ~((tempTraj(:,1) > arenaSize(1)) + (tempTraj(:,1) < 1));
    yCheck = ~((tempTraj(:,2) > arenaSize(2)) + (tempTraj(:,2) < 1));
    tempTraj = tempTraj(logical(xCheck.*yCheck), :);
    if size(tempTraj, 1) < size(maskPosCell{ii}, 1) % if some of the trajectory is out of bounds
        if errorFlag
            error('Coordinates are out of range')
        else
            relDirec = ceil(ii/numTraj);
            warning('Trajectory for direction %d is out of range', relDirec)
            maskPosCell{ii} = tempTraj;
        end
    end
end

nonEmptyInd = ~cellfun(@isempty, maskPosCell);

if sum(nonEmptyInd) < length(maskPosCell)
    warning('Empty trajectory/ies removed')
    maskPosCell = maskPosCell(nonEmptyInd);
end

% Remove coordinates that repeat themselves (esp. in diagonal trajectories)

for ii=1:length(maskPosCell)
    relInd = find(maskPosCell{ii}(end, :) - maskPosCell{ii}(1, :), 1, 'first');
    firstVal = maskPosCell{ii}(1,relInd);
    tempCell = unique(maskPosCell{ii}, 'rows');
    if tempCell(1,relInd) == firstVal % since unique also sorts
        maskPosCell{ii} = tempCell;
    else
        maskPosCell{ii} = flipud(tempCell);
    end
end
        
    
    
    
    
    
    
    
    





end