function z=sconnect(z1,z2,s)
z=(z1.^s+z2.^s).^(1/s);
end