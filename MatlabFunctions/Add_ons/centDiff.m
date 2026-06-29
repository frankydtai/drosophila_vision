function Y = centDiff(X, stepSiz)

%Taken from Michael
% function Y = cent_diff(X, h)
% computes the numerical derivative of X by using the central difference
% formula. For the derivatives to be meaningful, need to use the step size
% h.

if nargin < 2
    stepSiz=1;
    warning('stepSize value was not given: derivative not in right units')
end


Y = zeros(size(X));

Y(2:end-1) = (X(3:end) - X(1:end-2))/(2*stepSiz);

% just repeat the first and end point, so that Y is length of X
Y(1) = Y(2);
Y(end) = Y(end-1);


end


