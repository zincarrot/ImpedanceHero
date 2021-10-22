function plotymag(f,z)
y=1./z;
loglog(f,abs(y));
ylabel("|Y|/\Omega");
xlabel("f/Hz");
end