
dataH = load('H_system.mat');
dataK0 = load('K0_system.mat');

sys = dataH.sys_min;
K0 = dataK0.K0;

I = ss([], [], [], eye(2), Ts);
CLS = minreal(sys / (I + K0*sys));

Ts = 0.032;
D_NOM = 4.0;
files = dir('C:\Users\Imad-Eddine-FERHAT\Desktop\Lidar_data\*.csv');


for i = 1:length(files)

    fullFileName = fullfile(files(i).folder, files(i).name);
    T = readtable(fullFileName);

    X = T{:,:};
    X(isinf(X)) = NaN;
    X = fillmissing(X, 'previous');

    T{:,:} = X;

    delta_v = T.('speed_km_h') - 3;
    delta_delta = T.('steering_angle_rad');

    eps_ym30 = -(T.('lidar_150') - D_NOM);
    eps_y0 = -(T.('lidar_180') - D_NOM);
    eps_yp30 = -(T.('lidar_210') - D_NOM);
    eps = [eps_ym30, eps_y0, eps_yp30];

    tk0 = (0:size(eps,1)-1)' * Ts;
    u_K0 = lsim(K0, eps, tk0);

    y_qv = delta_v - u_K0(:,1);
    y_qdelta = delta_delta - u_K0(:,2);
    y_q = [y_qv, y_qdelta];

    tcls = (0:size(y_q,1)-1)' * Ts;
    y_b = lsim(CLS, y_q, tcls);

    u_q = y_b + eps;

    data = [u_q, y_q];

    output = array2table([u_q, y_q], ...
    'VariableNames', {'u_q_m30', 'u_q_0', 'u_q_p30', 'y_q_v', 'y_q_delta'});

    outName = fullfile(files(i).folder, ['uq_yq_' files(i).name]);

    writetable(output, outName);

end
