
dataname = 'all_data.csv';
data = load(dataname);

network_output_name = 'rbf_best_python.csv';
network_output = load(network_output_name);
network_output = network_output.';


% measurements Z_k = Z(t) + v(t)
alpha_mtrue = data(1,:); % measured angle of attack
beta_mtrue = data(2,:);  % measured angle of sideslip
alpha_m = network_output(:,1).';
beta_m = network_output(:,2).';
Cmtrue = data(3,:);
Cm = network_output(:,3);    % measured velocity