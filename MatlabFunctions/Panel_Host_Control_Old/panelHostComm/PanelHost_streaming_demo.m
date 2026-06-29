

%% This script is used to run the new panel controller from within matlab

system('C:\Users\reiserm\Documents\Panel Host\Panel Host.exe &');
pause(4);


global tcpNumber tcpName;
init_tcp;

Panel_tcp_com('all_off');

disp('tcp conn established')

tic
for k = 1:10
    Panel_tcp_com('dump_frame', [1152, 1, 0, 48, 1, 0, 255*mod(k,2)*ones(1,1152)]);
    pause(0.5)
end
toc 


% then stream something interesting
pattern.row_compression = 0; % kind of a hack, but then can use old code for process_panel_map
pattern.Panel_map = [12 8 4 11 7 3 10 6 2  9 5 1; 24 20 16 23 19 15 22 18 14 21 17 13; 36 32 28 35 31 27 34 30 26 33 29 25; 48 44 40 47 43 39 46 42 38 45 41 37];
BitMapIndex = process_panel_map(pattern);

%%%

%% white noise version 1

random_fraction = 0.1; % the fraction of pixels to set as random
number_pixels = 16 * 48 % using a 2x2 'pixel'
number_random = round(number_pixels*random_fraction)

tic
for k = 1:50
    k
    temp_frame = 1*ones(number_pixels,1);
    rand_index = randperm(number_pixels);
    temp_frame(rand_index(1:number_random)) = 7;
    Pat_frame = Pattern_Repeater(reshape(temp_frame, 16, 48),2);
    
    pat_vector = make_frame_vector_GS3(Pat_frame, BitMapIndex);
    Panel_tcp_com('dump_frame', [1152, k, 0, 48, 3, 0, pat_vector']);
    pause(0.1)
end
toc



%% white noise version 2, ON and OFF pixels

random_fraction = 0.1; % the fraction of pixels to set as random, ON or OFF (so total number of rand pixles will be 2x this quantity)

number_pixels = 16 * 48; % using a 2x2 'pixel'
number_random = round(number_pixels*random_fraction);

tic
for k = 1:50
    k
    temp_frame = 3*ones(number_pixels,1);
    rand_index = randperm(number_pixels);
    temp_frame(rand_index(1:number_random)) = 7; % the ON pixels
    temp_frame(rand_index((number_random+1) : (2*number_random))) = 0; % the OFF pixels
    Pat_frame = Pattern_Repeater(reshape(temp_frame, 16, 48),2);
    
    pat_vector = make_frame_vector_GS3(Pat_frame, BitMapIndex);
    Panel_tcp_com('dump_frame', [1152, k, 0, 48, 3, 0, pat_vector']);
    pause(0.5)
end
toc


%% white noise version 3, ON and OFF pixels, but only within a window

random_fraction = 0.1; % the fraction of pixels to set as random, ON or OFF (so total number of rand pixles will be 2x this quantity)
stim_window = [49 72 13 24]; % defined as [XMIN XMAX YMIN YMAX]
background_intensity = 1;
% now make 'global masks' this approach is inefficient, but should be good
% enough. Can always optimize later to only make patterns within the window
% instead of masking everythign outside of the window. 
BG_Mask = ones(32,96)*background_intensity;
BG_Mask(stim_window(3):stim_window(4), stim_window(1):stim_window(2)) = 0;
Sel_Mask = BG_Mask ==0 ;

number_pixels = 16 * 48; % using a 2x2 'pixel'
number_random = round(number_pixels*random_fraction);

tic
for k = 1:2000
    k
    temp_frame = 3*ones(number_pixels,1);
    rand_index = randperm(number_pixels);
    temp_frame(rand_index(1:number_random)) = 7; % the ON pixels
    temp_frame(rand_index((number_random+1) : (2*number_random))) = 0; % the OFF pixels
    Pat_frame = BG_Mask + Sel_Mask.*Pattern_Repeater(reshape(temp_frame, 16, 48),2);
    
    pat_vector = make_frame_vector_GS3(Pat_frame, BitMapIndex);
    Panel_tcp_com('dump_frame', [1152, k, k, 48, 3, 0, pat_vector']);
    pause(0.1)
end
toc


