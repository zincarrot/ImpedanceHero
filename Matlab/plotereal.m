function plotereal(f,z)
e=1./(1i*2*pi.*f.*z);
loglog(f,real(e));
ylabel("\epsilon'");
xlabel("f/Hz");
end