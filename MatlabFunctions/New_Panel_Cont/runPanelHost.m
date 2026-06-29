

% This script is used to run the new panel controller from within matlab


system('C:\Users\gruntmane\Documents\Panel Host\Panel Host.exe &');

pause(3);

global tcpHandle

tcpHandle =  init_tcp;
Panel_tcp_com('all_off');


disp('tcp conn established')


