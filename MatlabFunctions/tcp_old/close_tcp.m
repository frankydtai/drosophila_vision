function close_tcp()
%% closes TCP connection fo

global tcpName;

fclose(tcpName)
delete(tcpName)
clear tcpName


end
