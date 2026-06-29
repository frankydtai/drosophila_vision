function fixedIm = fixImAfterRot(stimIm, relVal, direction)

% This function is to be used after the minimal motion stimulus is rotated
% 45 degrees. It finds the cases where individual rows still exists and
% corrects the rest of the image with that.
%
% INPUT
% stimIm -      image that was rotated 45 degrees and is distorted (before
%               mask is applied)
% relVal -      value to correct
% direction -   1 or -1 to indicate direction of diagonal

bkgdVal = 7;
corrPix = 20; % will correst only 20 pix above and below diagonal
imSiz = size(stimIm);
midI = ceil(imSiz(1)/2);

midLine = stimIm(:, midI);

inds = find(midLine == relVal);

indsDiff = diff(inds);

fixedIm = stimIm;

if length(unique(indsDiff)) > 1 % if it is 1 then all values are the same and therefore all line are width one

   indsDiff = vertcat(0, indsDiff);
   tempDiff = find(diff(indsDiff) == 0);  % proper values of skipped pixels between lines
   skipSize = indsDiff(tempDiff(1)+1);
   %inds = inds(2:end); % to match diff indices
   yCorrInds = abs(sort((midI-corrPix:midI+corrPix)*direction));

   relRange = tempDiff(1)+1:tempDiff(end);
   checkV = 0;
   for ii=1:length(relRange)-1 % since I already know the first is correct
       checkV = checkV+indsDiff(relRange(ii));

       if rem(checkV, skipSize) == 0 || indsDiff(relRange(ii)) == skipSize
           checkV = 0;
           continue
       elseif indsDiff(relRange(ii)) < skipSize && indsDiff(relRange(ii)+1) == 1
           xCorrInds = inds(relRange(ii))-corrPix:inds(relRange(ii))+corrPix;
           linInd = sub2ind(imSiz, xCorrInds, yCorrInds);
           fixedIm(linInd) = bkgdVal;
       elseif indsDiff(relRange(ii)) < skipSize && indsDiff(relRange(ii)) == 1
           xCorrInds = inds(relRange(ii))-corrPix:inds(relRange(ii))+corrPix;
           linInd = sub2ind(imSiz, xCorrInds, yCorrInds);
           fixedIm(linInd) = bkgdVal;
       end
   end
end


end
