function dumpFrameCB(obj, event, patVec, patVecLen)

ind = obj.TasksExecuted;

temp = obj.UserData;
temp = temp+1;
%temp{2} = [temp{2}, toc(temp{1})];
obj.UserData = temp;


Panel_tcp_com('dump_frame', [patVecLen, temp, 0, 48, 3, 0, patVec(:, ind)']);

%fprintf('frame %d presented %d\n', timesRan, toc(obj.UserData))


end