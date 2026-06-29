function plot_figures


param_fname = 'lc18_model_param';
load( strcat(param_fname, '.mat'), 'x' ); % load optimized parameters

vstim_dir = './vstim/';

x_orig = x; % make a copy of optimized param

param = convX2param(x);

fun = @simulateXcontrast;




% -------------------------------------------------------------------------
% Fig3J-M
% dark bar width tuning - stimuli used in experiments


tmp = load( strcat(vstim_dir, 'Fig3_barTuning_experiment.mat') ); % load visual stimuli
fieldn = fieldnames(tmp);
stim = tmp.( fieldn{1} );

vstim2plot = [stim(:).barcode];
outfile_name = 'F3_barWidthTuning_experiment';
signals2analyze = {'ca'};

export_signals(fun, vstim2plot, stim, param, param_fname, outfile_name, signals2analyze);

plot_xRng = [-2,5];
plot_yRng = [-0.5, 1.5];
plot_each_protocol(fun, vstim2plot, stim, param, param_fname, outfile_name, plot_xRng, plot_yRng);


plot_xRng = [0.25, 1.25];
plot_yRng = [-0.015, 0.015];
plot_sum_yRng = [-0.003, 0.001];
plot_intermediate_signals(fun, vstim2plot, stim, param, param_fname, outfile_name, plot_xRng, plot_yRng, plot_sum_yRng);


% remove crossover inhibition and replot bar width tuning
outfile_name = 'F3_barWidthTuning_experiment_noXinh';
x_mod = x_orig;
x_mod(20) = 0;
x_mod(22) = 0;

param_mod = convX2param(x_mod);

export_signals(fun, vstim2plot, stim, param_mod, param_fname, outfile_name, signals2analyze);

plot_xRng = [-2,5];
plot_yRng = [-0.5, 1.5];
plot_each_protocol(fun, vstim2plot, stim, param_mod, param_fname, outfile_name, plot_xRng, plot_yRng);


plot_xRng = [0.25, 1.25];
plot_yRng = [-0.015, 0.015];
plot_sum_yRng = [-0.003, 0.001];
plot_intermediate_signals(fun, vstim2plot, stim, param_mod, param_fname, outfile_name, plot_xRng, plot_yRng, plot_sum_yRng);



% -------------------------------------------------------------------------
% Fig3N, S5B-D,F
% dark bar width tuning - stimuli for modeling, dense sampling


tmp = load( strcat(vstim_dir, 'Fig3_barTuning_model.mat') ); % load visual stimuli
fieldn = fieldnames(tmp);
stim = tmp.( fieldn{1} );

vstim2plot = [stim(:).barcode];
outfile_name = 'F3_barWidthTuning_model';
signals2analyze = {'ca', 'on_fast', 'off_fast', 'column_out', 'on_fast_ca', 'off_fast_ca', 'column_out_ca'};

export_signals(fun, vstim2plot, stim, param, param_fname, outfile_name, signals2analyze);

plot_xRng = [-2,5];
plot_yRng = [-0.5, 1.5];
plot_each_protocol(fun, vstim2plot, stim, param, param_fname, outfile_name, plot_xRng, plot_yRng);


% remove contrast saturation
outfile_name = 'F3_barWidthTuning_model_noSat';
x_mod = x_orig;
x_mod(7) = Inf;

param_mod = convX2param(x_mod);

export_signals(fun, vstim2plot, stim, param_mod, param_fname, outfile_name, signals2analyze);

% remove crossover inhibition
outfile_name = 'F3_barWidthTuning_model_noXinh';
x_mod = x_orig;
x_mod(20) = 0;
x_mod(22) = 0;

param_mod = convX2param(x_mod);

export_signals(fun, vstim2plot, stim, param_mod, param_fname, outfile_name, signals2analyze);


% -------------------------------------------------------------------------
% Fig S5B-D,F
% bright bar width tuning - stimuli for modeling, dense sampling


tmp = load( strcat(vstim_dir, 'Fig3_barTuning(bright)_model.mat') ); % load visual stimuli
fieldn = fieldnames(tmp);
stim = tmp.( fieldn{1} );

vstim2plot = [stim(:).barcode];
outfile_name = 'F3_widthTuning_brightBar_model';
signals2analyze = {'ca', 'on_fast', 'off_fast', 'column_out', 'on_fast_ca', 'off_fast_ca', 'column_out_ca'};

export_signals(fun, vstim2plot, stim, param, param_fname, outfile_name, signals2analyze);

plot_xRng = [-2,5];
plot_yRng = [-0.5, 1.5];
plot_each_protocol(fun, vstim2plot, stim, param, param_fname, outfile_name, plot_xRng, plot_yRng);


% remove contrast saturation
outfile_name = 'F3_widthTuning_brightBar_model_noSat';
x_mod = x_orig;
x_mod(7) = Inf;

param_mod = convX2param(x_mod);

export_signals(fun, vstim2plot, stim, param_mod, param_fname, outfile_name, signals2analyze);

% remove crossover inhibition
outfile_name = 'F3_widthTuning_brightBar_model_noXinh';
x_mod = x_orig;
x_mod(20) = 0;
x_mod(22) = 0;

param_mod = convX2param(x_mod);

export_signals(fun, vstim2plot, stim, param_mod, param_fname, outfile_name, signals2analyze);




% -------------------------------------------------------------------------
% Fig. 3O, S5E
% contrast tuning (4.5deg square)

tmp = load( strcat(vstim_dir, 'Fig3_smallSquareContrastTuning_model.mat') ); % load visual stimuli
fieldn = fieldnames(tmp);
stim = tmp.( fieldn{1} );

vstim2plot = 1:length(stim);
outfile_name = 'contrastTuning_smallSquare';
signals2analyze = {'ca'};

export_signals(fun, vstim2plot, stim, param, param_fname, outfile_name, signals2analyze);


% remove contrast saturation
outfile_name = 'contrastTuning_smallSquare_noSat';
x_mod = x_orig;
x_mod(7) = Inf;

param_mod = convX2param(x_mod);

export_signals(fun, vstim2plot, stim, param_mod, param_fname, outfile_name, signals2analyze);



% -------------------------------------------------------------------------
% Fig S5H
% ON-OFF square flicker


tmp = load( strcat(vstim_dir, 'Fig3_squareOnOffFlicker.mat') ); % load visual stimuli
fieldn = fieldnames(tmp);
stim = tmp.( fieldn{1} );

vstim2plot = [stim(:).barcode];
outfile_name = 'F3_squareOnOffFlicker';
signals2analyze = {'ca'};

export_signals(fun, vstim2plot, stim, param, param_fname, outfile_name, signals2analyze);

plot_xRng = [-0.5,1.5];
plot_yRng = [-0.5, 10.5];
plot_each_protocol(fun, vstim2plot, stim, param, param_fname, outfile_name, plot_xRng, plot_yRng);



% -------------------------------------------------------------------------
% Fig S5I
% temporal tuning - flicker stimuli


tmp = load( strcat(vstim_dir, 'FigS5_temporalTuning_flicker.mat') ); % load visual stimuli
stim = tmp.( 'flicker_stim_pr_mov' );

vstim2plot = 1:length(stim);
outfile_name = 'temporalTuning_flicker';
signals2analyze = {'ca'};

export_signals(fun, vstim2plot, stim, param, param_fname, outfile_name, signals2analyze);

% -------------------------------------------------------------------------
% Fig S5I
% temporal tuning - motion stimuli


tmp = load( strcat(vstim_dir, 'FigS5_temporalTuning_motion.mat') ); % load visual stimuli
fieldn = fieldnames(tmp);
stim = tmp.( fieldn{1} );

vstim2plot = 1:length(stim);
outfile_name = 'temporalTuning_motion';
signals2analyze = {'ca'};

export_signals(fun, vstim2plot, stim, param, param_fname, outfile_name, signals2analyze);




% -------------------------------------------------------------------------
% Fig S5G, J
% size x speed interaction simulation


tmp = load( strcat(vstim_dir, '2DsizeTuning_40dps.mat') ); % load visual stimuli
fieldn = fieldnames(tmp);
stim = tmp.( fieldn{1} );

vstim2plot = [stim(:).barcode];
outfile_name = '2Dsize_40dps';
signals2analyze = {'ca'};

export_signals(fun, vstim2plot, stim, param, param_fname, outfile_name, signals2analyze);




tmp = load( strcat(vstim_dir, '2DsizeTuning_80dps.mat') ); % load visual stimuli
fieldn = fieldnames(tmp);
stim = tmp.( fieldn{1} );

vstim2plot = [stim(:).barcode];
outfile_name = '2Dsize_80dps';
signals2analyze = {'ca'};

export_signals(fun, vstim2plot, stim, param, param_fname, outfile_name, signals2analyze);




tmp = load( strcat(vstim_dir, '2DsizeTuning_100dps.mat') ); % load visual stimuli
fieldn = fieldnames(tmp);
stim = tmp.( fieldn{1} );

vstim2plot = [stim(:).barcode];
outfile_name = '2Dsize_100dps';
signals2analyze = {'ca'};

export_signals(fun, vstim2plot, stim, param, param_fname, outfile_name, signals2analyze);






tmp = load( strcat(vstim_dir, '2DsizeTuning_200dps.mat') ); % load visual stimuli
fieldn = fieldnames(tmp);
stim = tmp.( fieldn{1} );

vstim2plot = [stim(:).barcode];
outfile_name = '2Dsize_200dps';
signals2analyze = {'ca'};

export_signals(fun, vstim2plot, stim, param, param_fname, outfile_name, signals2analyze);





tmp = load( strcat(vstim_dir, '2DsizeTuning_400dps.mat') ); % load visual stimuli
fieldn = fieldnames(tmp);
stim = tmp.( fieldn{1} );

vstim2plot = [stim(:).barcode];
outfile_name = '2Dsize_400dps';
signals2analyze = {'ca'};

export_signals(fun, vstim2plot, stim, param, param_fname, outfile_name, signals2analyze);




% -------------------------------------------------------------------------
% Fig S5K
% speed tuning - 4.5deg square


tmp = load( strcat(vstim_dir, 'FigS5_speedTuning.mat') ); % load visual stimuli
fieldn = fieldnames(tmp);
stim = tmp.( fieldn{1} );

vstim2plot = 1:length(stim);
outfile_name = 'speedTuning';
signals2analyze = {'ca', 'on_fast', 'off_fast', 'column_out', 'on_fast_ca', 'off_fast_ca', 'column_out_ca'};

export_signals(fun, vstim2plot, stim, param, param_fname, outfile_name, signals2analyze);

plot_xRng = [-2,2];
plot_yRng = [-0.2, 1.2];
plot_each_protocol(fun, vstim2plot, stim, param, param_fname, outfile_name, plot_xRng, plot_yRng);






end



function plot_each_protocol(fun, vstim2plot, stim, param, param_fname, outfile_name, plot_xRng, plot_yRng)

vstim_list = [stim(:).barcode];
stim_idx = zeros( size(vstim2plot) );
for k = 1:length(vstim2plot)
    stim_idx(k) = find(vstim2plot(k) == vstim_list);
end

stim = stim( stim_idx );

sim = feval(fun, stim, param); % simulate only ones that will be plotted



ctr.r = ceil( size(stim(1).pr_mov,1) / 2);
ctr.c = ceil( size(stim(1).pr_mov,2) / 2);

%%
% plot options
epsfig = hgexport('factorystyle');
epsfig.Format = 'eps';
set(0, 'DefaultFigurePosition', [0 0 100 200]);
set(0, 'DefaultFigureColor', 'white');
set(0, 'DefaultFigurePaperPositionMode','auto');


%%
% plot individual protocols as separate eps

plot_dir = strcat('./Plots/', param_fname, '/', outfile_name, '/');
if exist(plot_dir, 'dir') == 0
    mkdir( plot_dir );
end

for k = 1:length(sim)
    % plot calcium output
    figure;
    
    t = stim(k).t;
    
    subplot(2,1,1);
    plot(t, squeeze( sim(k).ca ));
    xlim(plot_xRng); ylim(plot_yRng);
    set(gca, 'box', 'off');
    title( strcat('protocol: ', num2str(stim(k).barcode)));
    
    subplot(2,1,2);
    plot(t, squeeze( stim(k).pr_mov(ctr.r,ctr.c,:) ));
    xlim(plot_xRng);
    set(gca, 'box', 'off');
    
    fname = strcat(plot_dir, 'prot_', num2str(stim(k).barcode), '.eps');
    hgexport(gcf, fname,epsfig,'Format','eps')
    close
end


end




function plot_intermediate_signals(fun, vstim2plot, stim, param, param_fname, outfile_name, plot_xRng, plot_yRng, plot_sum_yRng)

% similar plot function as plot_each_protocol
% but plot the intermediate signals instead of calcium output

vstim_list = [stim(:).barcode];
stim_idx = zeros( size(vstim2plot) );
for k = 1:length(vstim2plot)
    stim_idx(k) = find(vstim2plot(k) == vstim_list);
end

stim = stim( stim_idx );

sim = feval(fun, stim, param); % simulate only ones that will be plotted


ctr.r = ceil( size(stim(1).pr_mov,1) / 2);
ctr.c = ceil( size(stim(1).pr_mov,2) / 2);

%%
% plot options
epsfig = hgexport('factorystyle');
epsfig.Format = 'eps';
set(0, 'DefaultFigurePosition', [0 0 100 200]);
set(0, 'DefaultFigureColor', 'white');
set(0, 'DefaultFigurePaperPositionMode','auto');


% define some custom colors for the on/off signals
% requires matlab 2019 to specify custom hex color in plot fn
cc_off_fast = '#7B3294';
cc_off_slow = '#C2A5CF';
cc_off_sum = '#9F6CB2';

cc_on_fast = '#008837';
cc_on_slow = '#A6DBA0';
cc_on_sum = '#53B26C';


%% plot intermediate ON and OFF signals
% plot individual protocols as separate eps


plot_dir = strcat('./Plots/', param_fname, '/', outfile_name, '/intermediate_signals/');
if exist(plot_dir, 'dir') == 0
    mkdir( plot_dir );
end


% for each protocol
for k = 1:length(sim)
    % plot intermediate raw signal output, not gcamp filtered
    
    t = stim(k).t;

    on_output = squeeze( sim(k).on_fast + sim(k).on_slow );
    off_output = squeeze( sim(k).off_fast + sim(k).off_slow );
    
    relu_on_output = on_output;
    relu_on_output( relu_on_output < 0 ) = 0; % rectified output for ON channel
 
    relu_off_output = off_output;
    relu_off_output( relu_off_output < 0 ) = 0; % rectified output for ON channel
    
    summed_output = (1 - param.w_off) .* relu_on_output + param.w_off .* relu_off_output; % weighted sum, nearly symmetric weight ~ 0.54   
    
    
    % plot A: ON signals 
    figure;    
    
    subplot(2,1,1);
    plot(t, squeeze( sim(k).on_fast ), 'Color', cc_on_fast ); hold on;
    plot(t, squeeze( sim(k).on_slow ), 'Color', cc_on_slow ); hold off;
    xlim(plot_xRng); ylim(plot_yRng);
    set(gca, 'box', 'off');
    title( strcat('p: ', num2str(stim(k).barcode), ' ON'));
    
    subplot(2,1,2);
    plot(t, squeeze( stim(k).pr_mov(ctr.r,ctr.c,:) ));
    xlim(plot_xRng);
    ylim([-0.5, 0.5]);    
    set(gca, 'box', 'off');
    
    fname = strcat(plot_dir, 'prot_', num2str(stim(k).barcode), '_ON.eps');
    hgexport(gcf, fname,epsfig,'Format','eps')
    close
    
    
    % plot B: OFF signals
    figure;
        
    subplot(2,1,1);
    plot(t, squeeze( sim(k).off_fast ), 'Color', cc_off_fast ); hold on;
    plot(t, squeeze( sim(k).off_slow ), 'Color', cc_off_slow ); hold off;
    xlim(plot_xRng); ylim(plot_yRng);
    set(gca, 'box', 'off');
    title( strcat('p: ', num2str(stim(k).barcode), ' OFF'));
    
    subplot(2,1,2);
    plot(t, squeeze( stim(k).pr_mov(ctr.r,ctr.c,:) ));
    xlim(plot_xRng);
    ylim([-0.5, 0.5]);
    set(gca, 'box', 'off');
    
    fname = strcat(plot_dir, 'prot_', num2str(stim(k).barcode), '_OFF.eps');
    hgexport(gcf, fname,epsfig,'Format','eps')
    close
    
    
    % plot C: ON and OFF final outputs overlaid
    figure;
    
        
    subplot(2,1,1);
    plot(t, off_output, 'Color', cc_off_sum ); hold on;
    plot(t, on_output, 'Color', cc_on_sum ); hold off;
    xlim(plot_xRng); ylim( 2*plot_sum_yRng);
    set(gca, 'box', 'off');
    title( strcat('p: ', num2str(stim(k).barcode), 'sum'));
    
    subplot(2,1,2);
    plot(t, summed_output, 'k' );
    xlim(plot_xRng); ylim(plot_sum_yRng);
    set(gca, 'box', 'off');
    
    
    fname = strcat(plot_dir, 'prot_', num2str(stim(k).barcode), '_summed.eps');
    hgexport(gcf, fname,epsfig,'Format','eps')
    close
    
end

end


function export_signals(fun, vstim2plot, stim, param, param_fname, outfile_name, signals2analyze)

vstim_list = [stim(:).barcode];
stim_idx = zeros( size(vstim2plot) );
for k = 1:length(vstim2plot)
    stim_idx(k) = find(vstim2plot(k) == vstim_list);
end

stim = stim( stim_idx );

sim = feval(fun, stim, param); % simulate only ones that will be plotted


sim_barcode = [stim(:).barcode]';

sim_peaks = zeros( length( sim_barcode ), length(signals2analyze) );

for k = 1:length(sim)
    for s = 1:length(signals2analyze)
        sim_peaks(k, s) = max( abs( squeeze( sim(k).( signals2analyze{s}  ) ) ) ); % need to take max, simulation multiplied "inh" signals w/ negative sign already 
    end    
end


% pull out some stimulus info to also save to table export
stim_subfields = {'int', 'dim', 'speed', 'flicker_param', 'int_seq'};
extracted_fields = {}; % store valid field names
extracted_info = {}; % store stim info

cnt = 0;

for k = 1:length(stim_subfields)
    if isfield( stim, stim_subfields{k} )
        % do switch cases
        
        switch stim_subfields{k}
            case 'int'
                cnt = cnt + 1;
                
                extracted_fields{cnt} = 'bar_intensity';
                extracted_info{cnt} = [stim.int]';
            case 'dim'
                bar_dim = cat(1, stim.dim);
                bar_w = bar_dim(:, 1);
                bar_h = bar_dim(:, 2);
                
                cnt = cnt + 1;
                extracted_fields{cnt} = 'bar_width';
                extracted_info{cnt} = bar_w;
                
                cnt = cnt + 1;
                extracted_fields{cnt} = 'bar_height';
                extracted_info{cnt} = bar_h;
            case 'speed'
                cnt = cnt + 1;
                
                extracted_fields{cnt} = 'bar_speed';
                extracted_info{cnt} = [stim.speed]';
            case 'int_seq'
                intensity_sequence = cat(1, stim.int_seq);
                intensity_A = intensity_sequence(:, 1);
                intensity_B = intensity_sequence(:, 2);
                
                cnt = cnt + 1;
                extracted_fields{cnt} = 'bar_intensity_A';
                extracted_info{cnt} = intensity_A;
                
                cnt = cnt + 1;
                extracted_fields{cnt} = 'bar_intensity_B';
                extracted_info{cnt} = intensity_B;
            case 'flicker_param'
                flicker_param = cat(1, stim.flicker_param);
                num_cycles = flicker_param(:, 1);
                flash_duration = flicker_param(:, 2); % in seconds
                
                cnt = cnt + 1;
                extracted_fields{cnt} = 'number_cycles';
                extracted_info{cnt} = num_cycles;
                
                cnt = cnt + 1;
                extracted_fields{cnt} = 'flash_duration';
                extracted_info{cnt} = flash_duration;
                
            otherwise
                warning('unknown stimulus subfield');
        end
    end
end



plot_dir = strcat('./Plots/', param_fname, '/', outfile_name, '/');
if exist(plot_dir, 'dir') == 0
    mkdir( plot_dir );
end

    
fname = strcat(plot_dir, outfile_name, '_signal_peaks.csv');
 

concat_mat = cat(2, sim_barcode, cell2mat(extracted_info), sim_peaks);
tmp_table = array2table(concat_mat);
tmp_table.Properties.VariableNames = cat(2, {'prot_barcode'}, extracted_fields ,signals2analyze);

writetable(tmp_table, fname, 'WriteRowNames',true)


end