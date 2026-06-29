function param = convX2param(x)

param = struct;

param.hp_fc = x(1);

param.ratio_off2on = x(2);

param.on_adapt_fc = x(3);
param.on_adapt_scale = x(4);
param.off_adapt_fc = x(5);
param.off_adapt_scale = x(6);

param.e_on_sat = x(7);
param.e_on_gain = x(8);

param.i_on_sat = x(9);
param.i_on_gain = x(10);

param.e_off_sat = x(11);
param.e_off_gain = x(12);

param.i_off_sat = x(13);
param.i_off_gain = x(14);

param.e_on_lp = x(15);
param.i_on_lp = x(16);
param.e_off_lp = x(17);
param.i_off_lp = x(18);

param.on_xinh_fc = x(19);
param.on_xinh_scale = x(20);
param.off_xinh_fc = x(21);
param.off_xinh_scale = x(22);

param.w_off = x(23);

param.lc_scale = x(24);

param.lc18_fwhm = x(25);
