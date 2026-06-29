function varargout = Panel_tcp_log(command)

% function out = Panel_tcp_log(command)
%
% This function was seperated from Panel_tcp_com since log command now give
% outputs (look at release note for PanelHost v.1.0.0.178)
%
% INPUT 
% command - string. 'start' or 'stop' to start and stop log
%
% OUTPUT 
%
% if asked for gives the file name created

global tcpName

switch lower(command)
    
    case 'start'
        send_tcp( char([1 hex2dec('41')]));
    case 'stop'
        send_tcp( char([1 hex2dec('40')]));
    otherwise
        error('invalid command name, for Panel_tcp_log valid commands are "start" and "stop"')
end

while tcpName.BytesAvailable < 1
    pause(0.001) 
end

outBytes = fread(tcpName, tcpName.BytesAvailable);

% verify that everything was read
if outBytes(1)+1 < length(outBytes)
    assert(tcpName.BytesAvailable ~= 0, 'Missing Bytes - Aborted')
    outBytes = vertcat(outBytes, fread(tcpName, tcpName.BytesAvailable));
end

assert(outBytes(2) == 0, 'Problem with log file opening/closing, aborted function') 

if nargout == 1
    varargout{1} = char(outBytes(4:end))'; 
end

end