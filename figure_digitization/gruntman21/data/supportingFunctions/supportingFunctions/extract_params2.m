function [param_str,params] = extract_params2(T)
%function takes in a many variable table (ony one row) and returns those
%variables that are parameters of the model. 

params_str = {};
params = [];

T_list = T.Properties.VariableNames;

p_list = T_list(endsWith(T_list, '_p'));
          
counter = 1;
for i = 1:length(p_list)
    var_idx = strcmp(T_list,p_list{i});
    if ~any(var_idx)
        continue
    end
    param_str{counter} = T_list{var_idx};
    params = [params,T{:,var_idx}];
    counter = counter+1;
end
        