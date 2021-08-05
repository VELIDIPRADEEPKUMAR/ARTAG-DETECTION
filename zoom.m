function [Y] = zoom(I,S)

[R,C] = size(I);
I1 = zeros(R*S,C*S);
for i = 1:R
    
    for j = 1:C
        r1 = S*i - S + 1;
        c1 = S*j - S + 1;
        
        I1(r1,c1) = I(i,j);
        
    end
    
end

P = ones(S,S);            

for i = 1:R
    
    for j = 1:C
        
        r1 = S*i - S + 1;
        c1 = S*j - S + 1;
        
        Z = I1(r1:r1+S-1,c1:c1+S-1);
        F = ifft2(fft2(Z).*fft2(P));
        
        I1(r1:r1+S-1,c1:c1+S-1) = F; 
        
    end
    
end

Y = I1;

end

