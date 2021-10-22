function plotmag(f,z)
loglog(f,abs(z));
ylabel("|Z|/\Omega");
xlabel("f/Hz");
end