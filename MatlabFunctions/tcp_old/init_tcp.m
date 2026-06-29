function tcpHandle = init_tcp()

global tcpNumber tcpName;
tcpNumber = 62222;


tcpName                     = tcpip('localhost', tcpNumber, 'NetworkRole', 'client');
tcpName.OutputBufferSize    = 1164;
fopen(tcpName)

tcpHandle = tcpName;

end
