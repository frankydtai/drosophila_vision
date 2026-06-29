function gtSeq = generateGratingBaseSeq(gtSeqStruct, gratingFrameFunH)

% This function uses generateGratingFrame to construct a matrix of Grating
% Frames based on the parameters given in the gtSeqStruct (which include
% all the parameters for gtStruct (see createDefaultGratingStruct) The function 
% creates a sequence from the values that are vectors and not scalars. 
% If there is more than one, and they are of the same length, the function 
% will change them together. Any other combination is an error
%
% INPUT
% gtSeqStruct -         structure with the same fields as in generateGratingFrame
%                       or generateConcentricGratingFrame. One or more of the fields should be a
%                       vector so that a sequence will be generated from it
% gratingFrameFunH -    function handle to generateFrame functions (for now
%                       just ConcentricGrating and Grating)
% gsLevel -             (optional) gray scale level for the frames 
%
% OUTPUT
% 
% gtSeq -               225X225XN matrix of frames specified by the input. 
%                       N is the length of the field which is a vector 


% same size as in generateGratingFrame and generateBaseMask
matSiz = 225;

fieldLen = structfun(@length, gtSeqStruct);
fieldInd = find(fieldLen > 1);


% checks that the is at least one variable that changes
if isempty(fieldInd)
    warning('gratingGeneration:singleFrame', 'No field is a vector, generating a single frame')
    fieldInd = 1;
elseif length(fieldInd) > 1 % checks that variables that change together are of the same length
    testEqLen = fieldLen(fieldInd);
    compLen = unique(testEqLen);
    if length(compLen) > 1
        error('Vectors are not of same length, function aborted')
    end
end

fnames = fieldnames(gtSeqStruct);
vecLen = fieldLen(fieldInd(1));

% expands the structure to contain vectors of same length where fields were
% scalars (by just repeating those values)

for ii=1:length(fnames)
    if fieldLen(ii) == 1
        currVal = getfield(gtSeqStruct, fnames{ii});
        for jj=1:vecLen
            gtSeqStruct = setfield(gtSeqStruct, {jj}, fnames{ii}, currVal);
        end
    else
        currVal = getfield(gtSeqStruct, fnames{ii});
        for jj=1:vecLen
            gtSeqStruct = setfield(gtSeqStruct, {jj}, fnames{ii}, currVal(jj));
        end
    end
end


gtSeq = zeros(matSiz,matSiz, vecLen);

for ii=1:vecLen
    gtSeq(:,:,ii) = feval(gratingFrameFunH, gtSeqStruct(ii));
end







end

