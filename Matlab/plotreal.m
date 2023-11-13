function plotreal(f,z)
% semilogx(f,real(z));
loglog(f,real(z));
ylabel("Z'/\Omega");
xlabel("f/Hz");
end