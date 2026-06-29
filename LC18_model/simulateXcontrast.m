function sim = simulateXcontrast(stim, param, varargin)
% stim - visual stimulus, xyt array, already gaussian spatial sampled and temporally low-pass filtered 
% param - tunable parameters


if nargin < 3
    fs = 1000; % Hz, data sampling frequency
else
    fs = varargin{1};
end

% central visual column
ctr = struct;
ctr.r = ceil( size( stim(1).pr_mov ,1) / 2);
ctr.c = ceil( size( stim(1).pr_mov, 2) / 2);

for k = 1:length(stim)
    % block 1: photoreceptor processing
    b1_pr = stim(k).pr_mov;
    
	%------------------------------------------------------    
    % block 2: contrast
    % this block follows T2/T3 model from Tanaka, Clark 2020 Curr. Biol.
    b2_fc = param.hp_fc; % Hz, cutoff frequency
    [b, a] = butter(1, b2_fc/(fs/2), 'high'); % compute coefficients for filter
    b2_tf = filter( b, a, b1_pr, [], 3 ); % apply filter, along time dimension (3)
    
    % split into ON and OFF channels
    b2_on = b2_tf;
    b2_on( b2_on < 0 ) = 0;
    
    b2_off = -b2_tf;
    b2_off( b2_off < 0 ) = 0;
    
    
    % apply spatial filter with difference of gaussians    
    sigma = 4.5 .* [1, 3]; % degrees, stdev of center vs surround
    w = [1, 3.5]; % weights of center (positive) vs surround (negative)
    
    b2_on = DoG(b2_on, sigma, w);
    b2_off = param.ratio_off2on .* DoG(b2_off, sigma, w);
    

    % adapt ON or OFF separately
    [b, a] = butter(1, param.on_adapt_fc/(fs/2), 'low'); % compute coefficients for filter
    b2_on_adapted = b2_on ./ (1 + param.on_adapt_scale .* filter(b, a, b2_on, [], 3 ));
    
    [b, a] = butter(1, param.off_adapt_fc/(fs/2), 'low'); % compute coefficients for filter
    b2_off_adapted = b2_off ./ (1 + param.off_adapt_scale .* filter(b, a, b2_off, [], 3 ));
    
	%------------------------------------------------------
    % block 3: generate combinations of ON,OFF signals
    
    % run signals through linear gain/saturation function to ad hoc
    % diversify contrast response for each signal
    b3_e_on = saturate(b2_on_adapted, param.e_on_sat, param.e_on_gain);
    b3_i_on = saturate(b2_on_adapted, param.i_on_sat, param.i_on_gain);
    
    b3_e_off = saturate(b2_off_adapted, param.e_off_sat, param.e_off_gain);
    b3_i_off = saturate(b2_off_adapted, param.i_off_sat, param.i_off_gain);
    
    % low pass filtered signals
    [b, a] = butter(1, param.e_on_lp/(fs/2), 'low'); % compute coefficients for filter
    b3_e_on = filter(b, a, b3_e_on, [], 3 );
    
    [b, a] = butter(1, param.i_on_lp/(fs/2), 'low'); % compute coefficients for filter
    b3_i_on = filter(b, a, b3_i_on, [], 3 );
    
    [b, a] = butter(1, param.e_off_lp/(fs/2), 'low'); % compute coefficients for filter
    b3_e_off = filter(b, a, b3_e_off, [], 3 );
    
    [b, a] = butter(1, param.i_off_lp/(fs/2), 'low'); % compute coefficients for filter
    b3_i_off = filter(b, a, b3_i_off, [], 3 );
       
    
    % key step - cross contrast polarity interaction
    % currently, on and off signals are all strictly positive
    % divide signals to simulate downscaling of amplitudes without changing
    % signs, so use similar functional form as adaptation
    
    % ON (inh) is attenuated by OFF (exc)
    [b, a] = butter(1, param.off_xinh_fc/(fs/2), 'low'); % compute coefficients for filter
    b3_i_on_xinh = b3_i_on ./ (1 + param.off_xinh_scale .* filter(b, a, b3_e_off, [], 3 ));
    
    % OFF (inh) is attenuated by ON (exc)
    [b, a] = butter(1, param.on_xinh_fc/(fs/2), 'low'); % compute coefficients for filter
    b3_i_off_xinh = b3_i_off ./ (1 + param.on_xinh_scale .* filter(b, a, b3_e_on, [], 3 ));
      
    %------------------------------------------------------
    % block 4: E - I for ON and OFF channels
    
    b4_on = b3_e_on - b3_i_on_xinh;
    b4_on( b4_on < 0 ) = 0;
    
    b4_off = b3_e_off - b3_i_off_xinh;
    b4_off( b4_off < 0 ) = 0;
    
    
    b4_full = (1 - param.w_off) .* b4_on + param.w_off .* b4_off;   
    
    %------------------------------------------------------
    % LC18
       
    % spatial pooling step, simple gaussian
    sigma = 4.5 .* [param.lc18_fwhm, 0]; % degrees, stdev of center vs surround
    w = [1, 0]; % weights of center (positive) vs surround (negative)
    
    tmp = DoG(b4_full, sigma, w);
    lc_out = tmp(ctr.r, ctr.c, :); % readout from the center neuron only
    
    %------------------------------
    % gcamp readout

    ca = param.lc_scale .* lc_out;    
    ca( ca < 0 ) = 0; % remove negative resp
    
    ca_fc = 0.4; % Hz, gcamp6f has tau off ~400ms, which is 0.4 Hz
    [b, a] = butter(1, ca_fc/(fs/2), 'low'); % compute coefficients for filter
    ca = filter( b, a, ca, 0, 3);
    

    %------------------------------------------------------
    % generate additional outputs for paper figure
    % pull out intermediate signals from central visual column
    % add signs to indicate exc vs inhib
    
    % ON_faster, exc
    on_fast_single = b3_e_on(ctr.r, ctr.c, :);
    on_fast_ca = filter( b, a, on_fast_single, 0, 3);

    % ON_slower, inh
    on_slow_single = -b3_i_on_xinh(ctr.r, ctr.c, :);
        
    % OFF_faster, exc
    off_fast_single = b3_e_off(ctr.r, ctr.c, :);
    off_fast_ca = filter( b, a, off_fast_single, 0, 3);    
        
    % OFF_slower, inh
    off_slow_single = -b3_i_off_xinh(ctr.r, ctr.c, :);    
    
    
    % single visual column voltage output
    column_out = b4_full(ctr.r, ctr.c, :);
    column_out_ca = filter( b, a, column_out, 0, 3);    

    
    %------------------------------------------------------
    % save output signals for plotting
    
    sim(k).pr = b1_pr;
    sim(k).lc = lc_out;
    sim(k).ca = ca;   
    
    % other misc signals
    sim(k).on_fast = on_fast_single;
    sim(k).on_slow = on_slow_single;
    
    sim(k).off_fast = off_fast_single;
    sim(k).off_slow = off_slow_single;
    
    sim(k).on_fast_ca = on_fast_ca;
    sim(k).off_fast_ca = off_fast_ca;
    
    sim(k).column_out = column_out;
    sim(k).column_out_ca = column_out_ca;
    
end



end


function fxyt = DoG(xyt, sigma, w)

ns = 4.5; % deg, implicit spacing of neighboring points in input matrix

max_val = 2*ns * ceil(max(sigma)/ns); % sample twice the size of gaussian fwhm
x_azi_list = -max_val:ns:max_val;
y_ele_list = -max_val:ns:max_val;

[x_azi_grid, y_ele_grid] = meshgrid(x_azi_list, y_ele_list);

Kgauss_c = 1/(2*pi*sigma(1)^2) .* exp( -(x_azi_grid.^2 + y_ele_grid.^2) ./ (2*sigma(1)^2) );

if sigma(2) > 0
    Kgauss_s = 1/(2*pi*sigma(2)^2) .* exp( -(x_azi_grid.^2 + y_ele_grid.^2) ./ (2*sigma(2)^2) ) ;
else % no inhibitory surround
    Kgauss_s = 0;
end


Kgauss = w(1).*Kgauss_c - w(2).*Kgauss_s;
Kgauss = Kgauss ./ sqrt( sum( Kgauss(:).^2 ) * ns^2 ); % L2 norm


% pad input xyt matrix by replicating values at edges
dimX = size(xyt, 1); dimY = size(xyt, 2);
padded_xyt = [ ones(size(xyt)).*xyt(1,1,:), repmat(xyt(1,:,:), dimX,1,1), ones(size(xyt)).*xyt(1,end,:);
    repmat(xyt(:,1,:), 1,dimY,1), xyt, repmat(xyt(:,end,:), 1,dimY,1);
    ones(size(xyt)).*xyt(end,1,:), repmat(xyt(end,:,:), dimX,1,1), ones(size(xyt)).*xyt(end,end,:)];

fxyt = convn(padded_xyt, Kgauss, 'same') .* ns^2; % must rescale amplitude so it is independent of spatial sampling freq

% extract the unpadded part, should be same size as the input xyt
fxyt = fxyt( (1+dimX):(2*dimX), ...
    (1+dimY):(2*dimY), ...
    : );

fxyt( fxyt < 0 ) = 0; % rectify

end


function out = saturate(x, x_sat, gain)
% x - input vector
% x_sat - input value that should saturate
% gain - linear gain factor for output

% implicitly fix x=0, out=0 
out = gain.*x;

% abrupt flat saturation
out( x >= x_sat ) = gain*x_sat;

end



