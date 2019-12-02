function Feature = VMD_TF(signal)
% some sample parameters for VMD
alpha = length(signal);        % moderate bandwidth constraint
tau = 0;            % noise-tolerance (no strict fidelity enforcement)
K = 4;              % 4 modes
DC = 0;             % no DC part imposed
init = 1;           % initialize omegas uniformly
tol = 1e-6;

[u, u_hat, omega] = VMD(signal, alpha, tau, K, DC, init, tol);
% signal  - the time domain signal (1D) to be decomposed
% alpha   - the balancing parameter of the data-fidelity constraint
% tau     - time-step of the dual ascent ( pick 0 for noise-slack )
% K       - the number of modes to be recovered
% DC      - true if the first mode is put and kept at DC (0-freq)
% init    - 0 = all omegas start at 0
%                    1 = all omegas start uniformly distributed
%                    2 = all omegas initialized randomly
% tol     - tolerance of convergence criterion; typically around 1e-6
%
% Output:
% u       - the collection of decomposed modes
% u_hat   - spectra of the modes
% omega   - estimated mode center-frequencies
Feature1 = [];
Feature2 = [];
for i=1:K
    testdata=u(i,:);
    n=length(testdata);
    N=2^nextpow2(n);
    Y=fft(testdata,N)/N*2;
    A=abs(Y);
    A = A(1:1:N/2);
    energy = sum(A.*A); 
    Feature1 = [Feature1 energy ];
    AA = 0.5*(A.*A).^2;
    entropy = PermEn(AA,3);
    Feature2 = [Feature2 entropy];
end
Feature = [Feature1 Feature2];
end