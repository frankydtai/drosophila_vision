function varargout = plotBoxandJitterDots(plotAxH, cellData, inptSt)

% function varargout = plotBoxandJitterDots(axh, cellData, inptSt)
%
% this function is designed to plot an alternative to boxplot. 
% it plots a box with quartiles and overlays it with the actual points
% jittered
%
% INPUT
%
% plotAxH -             axes handles for the plot
% cellData -            cellarray. Each cell will be represented as a box with
%                       jittered dots on it. 
%                       
% inptSt -              (optional). input structure with different parameters
%                       for the plot. Given as fields in the structure
% .jitRange -           0-1. range between positions in which dots would be
%                       jittered (default 0.25)
% .boxWid -             single number. default is the same as jitter
% .boxEdge/FillCol -    Color for the boxes (either one, that will be applied
%                       to all, or the same as the number of cells. 
% .dotEdge/FillCol -    same as above for the dots. 
% .medLineCol -         color for the median line
% .medLineWid -         width of med line
% .dotMarker -          symbol for dot marker. 
% .dotSize -            obvious
% .boxLineWid -       width for the box line
% .positions -          positions to plot the data on (1XnumG)
%
% NOTE! is several values are given the should be given as a cell array
% except when givining dotSize, medLineWid or positions (can be given as
% arrays)




assert(iscell(cellData), 'input should be a cell array')

numG = length(cellData); 

% Default input structure

defSt.jitRange = 0.25; 
defSt.boxWid = defSt.jitRange; 
defSt.boxEdgeCol = repmat({'r'}, 1, numG);
defSt.boxFillCol = repmat({'r'}, 1, numG);
defSt.dotEdgeCol = repmat({'k'}, 1, numG);
defSt.dotFillCol = repmat({'k'}, 1, numG);
defSt.medLineCol = repmat({'k'}, 1, numG);
defSt.dotMarker = repmat({'o'}, 1, numG); 
defSt.dotSize = repmat(4, 1, numG);
defSt.boxLineWid = 1; 
defSt.medLineWid = repmat(4, 1, numG);
defSt.positions = 1:numG; 


% taking inptSt parameters
if nargin == 3
    
    relFnames = fieldnames(defSt); 
    currFNames = fieldnames(inptSt); 
    
    assert(all(ismember(currFNames, relFnames)), 'Field names in input structure are not correct')
    
    for ii=1:length(currFNames)
        
        tempF = getfield(inptSt, currFNames{ii});
        
        if strcmp(currFNames{ii}, 'jitRange') || strcmp(currFNames{ii}, 'boxWid') || strcmp(currFNames{ii}, 'boxLineWid')
            assert(length(tempF) == 1, 'jitter/boxWid/boxLineWid should be only one value')
            defSt = setfield(defSt, currFNames{ii}, tempF);
        elseif strcmp(currFNames{ii}(end-2:end), 'Col')
            
            if ischar(tempF) % color given as string
                assert(length(tempF) == 1 || strcmp(tempF, 'none'), 'field content should be either a single value or cell array the same length as input cell')
                
                defSt = setfield(defSt, currFNames{ii}, repmat({tempF}, 1, numG));
                
            elseif iscell(tempF) % color is given as a cell array of strings
                assert(length(tempF) ==numG, 'field content should be either a single value or cell array the same length as input cell')
                defSt = setfield(defSt, currFNames{ii}, tempF);
                
            else %color given as NX3 matrix
                
                assert(size(tempF, 2) == 3, 'color matrix should be a NX3 matrix')
                
                if size(tempF,1) == 1
                    defSt = setfield(defSt, currFNames{ii}, repmat({tempF}, 1, numG));
                elseif size(tempF,1) == numG
                    tempC = cell(1,numG); 
                    for jj=1:numG
                        tempC{jj} = tempF(jj,:); 
                    end
                    defSt = setfield(defSt, currFNames{ii}, tempC);
                end
            end
            
        elseif strcmp(currFNames{ii}(end-2:end), 'ker') % if it is marker type (only char or cell array of char
            
            if ischar(tempF) % marker given as string
                assert(length(tempF) == 1, 'field content should be either a single value or cell array the same length as input cell')
                defSt = setfield(defSt, currFNames{ii}, repmat({tempF}, 1, numG));
            else iscell(tempF) % marker is given as a cell array of strings
                assert(length(tempF) ==numG, 'field content should be either a single value or cell array the same length as input cell')
                defSt = setfield(defSt, currFNames{ii}, tempF);
            end
            
        elseif strcmp(currFNames{ii}, 'positions')
            assert(length(tempF) == numG, ...
                  'positions should be the same length as input cell')
            defSt = setfield(defSt, currFNames{ii}, tempF);
        else
            assert(length(tempF) == 1 || length(tempF) == numG, ...
                  'field content should be either a single value or cell array the same length as input cell')
            if length(tempF) == 1
                defSt = setfield(defSt, currFNames{ii}, repmat(tempF, 1, numG));
            else
                defSt = setfield(defSt, currFNames{ii}, tempF);
            end
            
        end
        
    end
    
end


qCD = cellfun(@(x) quantile(x, [0.25, 0.5, 0.75]), cellData, 'uniformoutput', 0); 

hold(plotAxH, 'on')

jitRng = defSt.jitRange;
boxWid = defSt.boxWid;
boxLW = defSt.boxLineWid;

for ii=1:numG
    
    dotDat = cellData{ii};
    qData = qCD{ii}; 
    relPos = defSt.positions(ii);
    
    xxP = [relPos-boxWid/2, relPos+boxWid/2, relPos+boxWid/2, relPos-boxWid/2]; 
    yyP = [qData(1), qData(1), qData(3), qData(3)];
    
    patch(plotAxH, xxP, yyP, defSt.boxFillCol{ii}, 'edgeColor', defSt.boxEdgeCol{ii}, 'linewidth', boxLW)
    
    line(xxP(1:2), [qData(2), qData(2)], 'linewidth', defSt.medLineWid(ii), 'color', defSt.medLineCol{ii})
    
    xxD = (rand(1, length(dotDat)) - 0.5) * jitRng + relPos; 
    
    plot(plotAxH, xxD, dotDat, 'marker', defSt.dotMarker{ii}, 'markerSize', defSt.dotSize(ii), ...
                               'markerfacecolor', defSt.dotFillCol{ii}, 'markeredgecolor', defSt.dotEdgeCol{ii}, ...
                               'color', 'none')
                           
                           
end
    

hold(plotAxH, 'off')


if nargout == 1
    varargout{1} = plotAxH;
end








end





