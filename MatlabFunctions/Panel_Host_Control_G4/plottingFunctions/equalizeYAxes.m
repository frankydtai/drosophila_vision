function equalizeYAxes(axesHand)

% function equalizeYAxes(axesHand)
%
% this function finds the ylim of all axes from the handles given and
% equalizes them

yyLim = get(axesHand(:), 'ylim');
yyLim = vertcat(yyLim{:});

set(axesHand(:), 'ylim', [min(yyLim(:,1)), max(yyLim(:,2))])




end