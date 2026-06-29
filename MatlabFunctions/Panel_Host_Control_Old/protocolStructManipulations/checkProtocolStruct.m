function outStruct = checkProtocolStruct(protStruct)


% function outStruct = checkProtocolStruct(protStruct);
%
% This function check that the protcol structure has all the required
% fields to construct a proper protocol (see also createProtocol). 
% If non-crucial fields are missing the function fill them based on other
% inputs (fill interleave based on the lengths of stimSeqCell, masksMat and ortientations) or based on defaults (randomizations)
%
% OUTPUT is a the same structure with additional fields if necessary
% (.interleave and or .randomize and its parts)


outStruct = protStruct;

arenaSize = [96,32];  % in pixels, in spatial coordinates 
possibleOrientations = 0:7;
baseImSize = [225, 225]; % size for original stimSeq and masks



    % FIELD 1 - gratingStruct
gratingNames = {'widthON'; 'widthOFF'; 'position'; 'barAtPos'; 'valsONSt'; 'valsONEnd'; 'valsOFFSt'; 'valsOFFEnd'}; 
gratingIndsNames = {'fVal'; 'sVal'; 'ori'; 'phase'; 'sqDim'; 'gsLevel'};
gratingConcenNames = {'widthON'; 'widthOFF'; 'position'; 'barAtPos'; 'valsONSt'; 'valsONEnd'; 'valsOFFSt'; 'valsOFFEnd'; 'type'};    
grating4BarNames = {'width1'; 'width2'; 'width3'; 'width4';'position'; 'barAtPos'; ...
                'vals1St'; 'vals1End'; 'vals2St'; 'vals2End'; 'vals3St'; 'vals3End'; 'vals4St'; 'vals4End'};  
randomDotNames = {'dotSize'; 'valON'; 'valOFF'; 'propON'; 'propOFF'; 'gsLevel'; 'bkgdVal'; 'relFrameSize'; 'rngSeed'}; 
generateBarFrameByIndsNames = {'wid'; 'ori'; 'pos'; 'val'; 'sqDim'; 'gsLevel'; 'bkgdVal'; 'matSize'};
generate2BarsFrameByIndsNames = {'fWid'; 'sWid'; 'fPos'; 'sPos'; 'fVal'; 'sVal'; 'sqDim'; 'ori'; 'gsLevel'; 'bkgdVal'; 'matSize'};
funcStr = func2str(protStruct.funcHand);

switch funcStr(9:11)
    case 'Con'
        relnames = gratingConcenNames;
    case 'Gra'
        if strcmp(funcStr(end-3:end), 'Inds')
            relnames = gratingIndsNames;
        else
            relnames = gratingNames;
        end
    case '4Ba'
        relnames = grating4BarNames;
    case 'Ran'
        relnames = randomDotNames;
    case '2Ba'
        relnames = generate2BarsFrameByIndsNames;
    case 'Bar'
        relnames = generateBarFrameByIndsNames;
    otherwise
        relnames = [];
end
        
        

if isfield(protStruct, 'gratingStruct')
    gtfnames = fieldnames(protStruct.gratingStruct);
    if ~isempty(relnames)
        fieldsPresent = cellfun(@(x) ismember(x, relnames), gtfnames);
        assert(sum(fieldsPresent) == length(relnames), 'Missing fields in Grating structure')
    else
        warning('do not have names for this frame generating function: gratingStruct was not checked')
    end
else
    error('gratingStruct field is missing!')
end
    

    % FIELD 2 - masksStruct

relnames = {'type', 'radius'};
if isfield(protStruct, 'masksStruct')
    mknames = fieldnames(protStruct.masksStruct);
    fieldsPresent = cellfun(@(x) ismember(x, relnames), mknames);
    assert(sum(fieldsPresent) == length(relnames), 'Missing fields in Grating structure')
else
    error('masksStruct is missing!')
end


    % FIELD 3 - orientations
if isfield(protStruct, 'orientations')

    % checking proper values
    checkVals = ismember(protStruct.orientations, possibleOrientations);
    assert(sum(checkVals) == length(checkVals), 'Orientations should have values between %d and %d', min(possibleOrientations), max(possibleOrientations)); 
else
    error('orientations field is missing!')
end



    % FIELD 4 - mask positions
if isfield(protStruct, 'maskPositions')
    if iscell(protStruct.maskPositions)
        mpLen = cellfun(@(x) size(x, 2), protStruct.maskPositions);
        assert(unique(mpLen) == 2, 'mask position cell array should be of NX2 matrices')
    else
        assert(size(protStruct.maskPositions,3) == 1, 'mask position matrix should be an NX2 matrix')
        assert(size(protStruct.maskPositions, 2) == 2, 'mask position matrix should be an NX2 matrix')
        for ii=1:2
            assert(max(protStruct.maskPositions(:,ii)) <= arenaSize(ii), ...
                'Values of mask positions should not exceed %d', arenaSize(ii)) 
        end

        assert(min(min(protStruct.maskPositions)) >= 1, 'Values of mask positions should not be lower than 1')
        posRound = arrayfun(@(x) x == round(x), protStruct.maskPositions);
        assert(prod(posRound(:)) == 1, 'Values of mask positions should be round numbers')
    end
else
    error('masksPosition field is missing!')
end


    % FIELD 5 - repeats


if isfield(protStruct, 'repeats')
    
    assert(protStruct.repeats == round(protStruct.repeats), 'Repeat values should be round numbers')
else
    outStruct = setfield(protStruct, repeats, 3);
    fprintf('added field %s with value %d', 'repeats', 3)
end

    %FIELD 6 - interleave
    
if isfield(protStruct, 'interleave')
    intVal = protStruct.interleave;
    assert(ismember(intVal, 0:4), 'interleave should be between 0-4')
    
    % checking that if interleave is negetive it is possible to generate a
    % protocol
    relSizes = [length(protStruct.gratingStruct), length(protStruct.masksStruct), length(protStruct.orientations)];
    switch intVal
        
        case 0
            
            sumDiff = sum(diff(relSizes)); % all of same length
            numOnes = sum(relSizes == 1); % at least 2 have just one component
            twoId = 0;
            if numOnes == 1
                twoId = diff(relSizes(~(relSizes == 1))); % in case one is of length 1 the other 2 should be identical
            end
            assert(sumDiff == 0 || numOnes == 2 || twoId == 0, ...
                'interleave cannot be 0 if numbers of stimSeqs, masks and orientations are not identical (if 2 of 3 are the same they should be one or the third one should)')
            % NEED TO UPDATE THE RIGHT INTERLEAVE VALUES
%         case 1
%             assert(relSizes(1) == relSizes(2), 'interleave 1 requires the same number of stimSeqs and masks')
    end
else
    outStruct = setfield(protStruct, 'interleave', 1);
    fprintf('added field %s with value %d', 'interleave', 1)
end



    % FIELD 7 - randomize
    
randSubfields = {'gratingSeq'; 'masks'; 'orientations'; 'maskPositions'};
    
if isfield(protStruct, 'randomize')    
    randSt = protStruct.randomize;
    
    for ii=1:length(randSubfields)
        if isfield(randSt, randSubfields{ii})
            testVal = getfield(randSt, randSubfields{ii});
            assert(ismember(testVal, [0,1]), 'Randomize values should be logicals')
        else
            
            randSt = setfield(randSt, randSubfields{ii}, 1);
            fprintf('Added %s to randomize structure with value %d\n', randSubfields{ii}, 1)
        end
    end
    protStruct = setfield(protStruct, 'randomize', randSt); % adds missing fields if needed
else
    randSt = struct('gratingSeq', 1, 'masks', 1, 'orientations', 1, 'maskPositions', 1);
    outStruct = setfield(protStruct, 'randomize', randSt); 
    fprintf('Added entire randomize structure\n')
end


    % Field 8 - generalFrequency
    
if isfield(protStruct, 'generalFrequency')
    assert(protStruct.generalFrequency > 0, 'Frequency should be a positive number (Hz)')
end

if isfield(protStruct, 'freqCorrFlag')
   assert(ismember(protStruct.freqCorrFlag, [0,1]), 'freqCorrFlag should be logical')
end


end



