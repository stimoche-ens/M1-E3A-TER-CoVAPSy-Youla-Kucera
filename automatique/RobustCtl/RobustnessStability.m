clear; clc; close all;

Ts = 0.032;

%% ========== Angle -30 ==========
a_m30 = [0.9798748095196319, 0.009887793616690812, -0.054727352291114736, 0.0416740506458685];
bv_m30 = [0.06267114858515915, -0.010686849668911902, -0.1545204919517112, 0.0997761820509041];
bd_m30 = [-1.327492842167979, 1.0818560719438495, 0.9427613537291105, -0.6981330991554017];

den_m30 = [1, -a_m30];
numv_m30 = [0, bv_m30];
numd_m30 = [0, bd_m30];

Gv_m30 = tf(numv_m30, den_m30, Ts, 'Variable', 'z^-1');
Gd_m30 = tf(numd_m30, den_m30, Ts, 'Variable', 'z^-1');

G_m30 = [Gv_m30 Gd_m30];   % 1 output, 2 inputs
sys_m30 = ss(G_m30);

%% ========== Angle 0 ==========
a_0 = [0.9896811947147172, 0.02088693855160128, 0.013301571881347765, -0.03947176828777346];
bv_0 = [0.11544011686595934, -0.21114958404451617, 0.24122972267420098, -0.1502557979462919];
bd_0 = [-0.1245289071729435, 0.3364004121740889, -0.14164617414038372, -0.08851161106885697];

den_0 = [1, -a_0];
numv_0 = [0, bv_0];
numd_0 = [0, bd_0];

Gv_0 = tf(numv_0, den_0, Ts, 'Variable', 'z^-1');
Gd_0 = tf(numd_0, den_0, Ts, 'Variable', 'z^-1');

G_0 = [Gv_0 Gd_0];
sys_0 = ss(G_0);

%% ========== Angle +30 ==========
a_p30 = [0.9932978232209756, 0.010313733022527177, 0.018613456193800216, -0.029815067952104986];
bv_p30 = [0.008819717273354661, 0.0037502826532860344, 0.002926811990912702, -0.019106861251709923];
bd_p30 = [0.4608263259628151, -0.24057750623300656, -0.5418587928824419, 0.29881853558934685];

den_p30 = [1, -a_p30];
numv_p30 = [0, bv_p30];
numd_p30 = [0, bd_p30];

Gv_p30 = tf(numv_p30, den_p30, Ts, 'Variable', 'z^-1');
Gd_p30 = tf(numd_p30, den_p30, Ts, 'Variable', 'z^-1');

G_p30 = [Gv_p30 Gd_p30];
sys_p30 = ss(G_p30);

%% ========== Global 3 outputs / 2 inputs ==========
G = [G_m30;
     G_0;
     G_p30];

sys = ss(G);

disp('Global discrete-time state-space model:')
sys

%% Poles / stability check
disp('Poles of global system:')
disp(pole(sys))

%% Minimal realization
sys_min = minreal(sys);
save('H_system.mat', 'sys_min');

%% Controllability / observability
A = sys_min.A;
B = sys_min.B;
C = sys_min.C;
D = sys_min.D;

n = size(A,1);
rank_ctrb = rank(ctrb(A,B));
rank_obsv = rank(obsv(A,C));

fprintf('Number of states        = %d\n', n);
fprintf('Controllability rank    = %d\n', rank_ctrb);
fprintf('Observability rank      = %d\n', rank_obsv);

%% ========== Hinf controller synthesis ==========
n = size(A,1);   % number of states
m = size(B,2);   % number of control inputs
p = size(C,1);   % number of measured outputs

% Disturbance input matrix
B1 = B;     % simplest starting choice

% Control input matrix
B2 = B;

% State and control penalties
Q = eye(n);
R = eye(m);

% Build C1 and D12 for z = [sqrt(Q)x; sqrt(R)u]
C1 = C;
D12 = zeros(p,m);

% Measurement equation
C2 = C;
D21 = zeros(p, size(B1,2));      % no direct disturbance-to-measurement term
D22 = D;

% Generalized plant P with inputs [w; u] and outputs [z; y]
P = ss(A, [-B1 -B2], ...
       [C1; C2], ...
       [zeros(size(C1,1), size(B1,2)) D12;
        D21                         D22], Ts);

% H-infinity synthesis
nmeas = p;   % y dimension
ncont = m;   % u dimension

[K0, CL, gamma] = hinfsyn(P, nmeas, ncont);
disp('The synthesised controller:')
K0
save('K0_system.mat', 'K0');
%% ========== Checking the results ==========
isstable(CL)
eig(A)
pole(K0)

Tcl = lft(P,K0);   % if dimewhy did we use lft here?nsions are set appropriately
isstable(Tcl)
pole(Tcl)

order(K0)

disp('Achieved Hinf gamma:')
disp(gamma)
norm(K0)
%% ========== Loop transfer / sensitivity functions ==========
Gyu = sys_min;   % plant from u to y

L  = Gyu*K0;
S  = feedback(eye(size(Gyu,1)), L);   % S = inv(I + L)
T  = feedback(L, eye(size(Gyu,1)));   % T = L*inv(I + L)
KS = K0*S;                            % KS = K*inv(I + L)

%% Stability check
disp('Closed-loop poles from T:')
pole(T)

%% Singular value plots
figure;
sigma(S);
grid on;
title('Sensitivity function S');

figure;
sigma(KS);
grid on;
title('Control sensitivity KS');

figure;
sigma(T);
grid on;
title('Complementary sensitivity T');

figure;
sigma(S, KS, T);
grid on;
legend('S','KS','T');
title('Singular values of S, KS, and T');