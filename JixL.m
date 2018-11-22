invJ = rand(3);
L = rand(8,3,8);
B = zeros(24,1);
for i=0:7
    for k=0:2
        for j=0:7
%             B(j+8*k+1) = 0.0;
            Bind = j+8*k+1
            for l=0:2
                %                 B(j+8*k+1) = B(j+8*k+1) + invJ(k+3*l+1)*L(8*l+3*j+24*i+1);
%                 Bind = j+8*k+1
                Jind = k+3*l+1
                Lind = j+8*l+24*i+1
            end
        end
    end
end

            // Gradient matrix B
            for (k=0;k<3;k++) {
                for (j=0;j<8;j++) {
                    B[j+8*k] = 0.0;
                    for (l=0;l<3;l++) {
                        B[j+8*k] += invJ[k+3*l]*L[j+8*l+24*i];
                    }
                }
            }