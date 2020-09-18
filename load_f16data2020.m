
dataname = 'all_data.csv';
data = load(dataname);

% measurements Z_k = Z(t) + v(t)
alpha_m = data(1,:); % measured angle of attack
beta_m = data(2,:);  % measured angle of sideslip
Cm = data(3,:);    % measured velocity