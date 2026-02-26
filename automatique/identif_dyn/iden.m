na = 2;
nb = 2;
nc = 2;
nk = 1;

% ecart par rapport aux donnees non perturbees
n_ech = 25;
cmd_len = 2;
lidar_len = 0;

%openExample('ident/EstimateARMAXModelExample')

X = zeros(n_ech,cmd_len+lidar_len);
X(:,1:2) = 2;

for row = 1:n_ech
    X(row,3:2+lidar_len) = 5;
end

X

load sdata2 tt2

%X = x(:).^(n:-1:0)

sys = armax(X,[na nb nc nk]);

