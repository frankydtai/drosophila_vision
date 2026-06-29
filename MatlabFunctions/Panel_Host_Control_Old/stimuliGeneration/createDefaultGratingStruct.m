function gtStruct = createDefaultGratingStruct

% This function generates a default grating structure with all the required
% fields

gtStruct.widthON = 4;       % width of bar with lum increaments
gtStruct.widthOFF = 4;      % width of bar with lum decrements
gtStruct.position = 0;      % relative to the center of the matrix that is used to generate the mask+stim
                            % designates the left side of the ON bar before
                            % rotation in pixels
gtStruct.barAtPos = 1;      % Which bar to position in the specified place (1 for 'ON' or 0 for 'OFF')
gtStruct.valsONSt = 1;      % beginning and end of the ON bar, if different will linearly interpolate
gtStruct.valsONEnd = 1;
gtStruct.valsOFFSt = 0;       % same for OFF. Value are between 0 and 1 and will be adjusted according 
gtStruct.valsOFFEnd = 0;
                            % to the GS level used (non integer values will be ROUNDED)
                            % first value will be the left-side value of
                            % the bar before rotation
                            
                            
end