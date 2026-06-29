clear all
close all


% plot options
epsfig = hgexport('factorystyle');
epsfig.Format = 'eps';
set(0, 'DefaultFigurePosition', [0 0 300 200]);
set(0, 'DefaultFigureColor', 'white');

contour_colormap = colormap_magma(100);

input_dir = {'2Dsize_40dps', '2Dsize_80dps', '2Dsize_100dps', '2Dsize_200dps', '2Dsize_400dps'};
file_suffix = '_signal_peaks.csv';


%% import data

data = {};
max_ca = -1;

for k = 1:length(input_dir)
    input_file = strcat('../', input_dir{k}, '/', input_dir{k}, file_suffix);
    data{k} = readtable([input_file]); 
    
    tmp_ca = table2array( data{k}(:, 'ca') );
    
    max_ca = max( cat(1, max_ca, tmp_ca ) );
end

max_ca = max_ca * 1.2; % scale color slightly higher than the max value


%% reshape data and plot as heatmap


for k = 1:length(data)
    h_list = table2array( data{k}(:, 'bar_height') );
    h_grid = reshape(h_list, [7,7]);
    
    w_list = table2array( data{k}(:, 'bar_width') );
    w_grid = reshape(w_list, [7,7]);
    
    ca_list = table2array( data{k}(:, 'ca') );
    ca_grid = reshape(ca_list, [7,7]);
    
    ca_grid_norm = ca_grid ./ max_ca;
    
    
    
    figure;
    contourf(w_grid, h_grid, ca_grid_norm, 'LineStyle', 'none');
    xlabel('Object width (°)');
    ylabel('Object height (°)');
    colormap(contour_colormap);
    caxis([0, 1]);

    axis equal
    
    view([0,90]);
    set(gca, 'XScale', 'log');
    set(gca, 'YScale', 'log');
    
    
    ax = gca;
    ax.XTick = [2, 4.5, 9, 15, 30, 45];
    ax.YTick = [2, 4.5, 9, 15, 30, 45];
    
    set(gca,'TickDir','out');
    
    fname = strcat( input_dir{k}, '.eps' );
    hgexport(gcf, fname,epsfig,'Format','eps')
    
    colorbar
    fname = strcat( input_dir{k}, '_withScalebar.eps');
    hgexport(gcf, fname,epsfig,'Format','eps')
    
    
    close

end





