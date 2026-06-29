function allFrameCB(obj, event)

timesRan = obj.TasksExecuted;
tog = rem(timesRan, 2);

if tog
    Panel_com('all_on')
else
    Panel_com('all_off')
end

temp = obj.UserData;
temp{2} = [temp{2}, toc(temp{1})];
obj.UserData = temp;

end
    
    