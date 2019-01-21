n = zeros(8,1);
x = zeros(8,1);
y = zeros(8,1);
z = zeros(8,1);
B = zeros(6*24,1);
dNdxyz = zeros(3*8,1);
invJ = zeros(9,1);
nel = size(elements,2);
L = dNdrst(class(nodes));

for tid =1:nel
    
    for i=1:8
        n(i) = elements(i+8*(tid-1));
    end
    
    for i=1:8
        x(i) = nodes(3*n(i)-2);                                 %// x-coordinate of node i
        y(i) = nodes(3*n(i)-1);                                 %// y-coordinate of node i
        z(i) = nodes(3*n(i)-0);                                 %// z-coordinate of node i
    end
    
    for i=1:144
        B(i) = 0.0;
    end
    
    for i=1:8
        
        J = [0, 0, 0, 0, 0, 0, 0, 0, 0];
        for j=1:8
            dNdr = L(3*(j-1)+24*(i-1)+1); dNds = L(3*(j-1)+24*(i-1)+2); dNdt = L(3*(j-1)+24*(i-1)+3);
            J(1) = J(1)+ dNdr*x(j); J(4) = J(4)+ dNdr*y(j);	J(7) = J(7)+ dNdr*z(j);
            J(2) = J(2)+ dNds*x(j);	J(5) = J(5)+ dNds*y(j);	J(8) = J(8)+ dNds*z(j);
            J(3) = J(3)+ dNdt*x(j);	J(6) = J(6)+ dNdt*y(j);	J(9) = J(9)+ dNdt*z(j);
        end
        
        detJ =  J(0+1)*J(4+1)*J(8+1) + J(3+1)*J(7+1)*J(2+1) + J(6+1)*J(1+1)*J(5+1) - ...
            J(6+1)*J(4+1)*J(2+1) - J(3+1)*J(1+1)*J(8+1) - J(0+1)*J(7+1)*J(5+1); %// Jacobian determinant
        
        iJ = 1/detJ;   invJ(0+1) = iJ*(J(4+1)*J(8+1)-J(7+1)*J(5+1));       %// Jacobian inverse
        invJ(1+1) = iJ*(J(7+1)*J(2+1)-J(1+1)*J(8+1));   invJ(2+1) = iJ*(J(1+1)*J(5+1)-J(4+1)*J(2+1));
        invJ(3+1) = iJ*(J(6+1)*J(5+1)-J(3+1)*J(8+1));   invJ(4+1) = iJ*(J(0+1)*J(8+1)-J(6+1)*J(2+1));
        invJ(5+1) = iJ*(J(3+1)*J(2+1)-J(0+1)*J(5+1));   invJ(6+1) = iJ*(J(3+1)*J(7+1)-J(6+1)*J(4+1));
        invJ(7+1) = iJ*(J(6+1)*J(1+1)-J(0+1)*J(7+1));   invJ(8+1) = iJ*(J(0+1)*J(4+1)-J(3+1)*J(1+1));
        
        %// Shape function derivatives with respect to x,y,z
        for j=1:8
            for k=1:3
                dNdxyz(k+3*(j-1)) = 0.0;
                for l=1:3
                    dNdxyz(k+3*(j-1)) = dNdxyz(k+3*(j-1))+ invJ(k+3*(l-1)) * L(l+3*(j-1)+24*(i-1));
                end
            end
        end
        
        for j=1:8                                    %// Matrix B
            B(0+18*(j-1)+1) 	 = dNdxyz(0+3*(j-1)+1); 	%// B(1,1:3:24) = dNdxyz(1,:);
            B(6+1+18*(j-1)+1)  = dNdxyz(1+3*(j-1)+1);	%// B(2,2:3:24) = dNdxyz(2,:);
            B(2+12+18*(j-1)+1) = dNdxyz(2+3*(j-1)+1);	%// B(3,3:3:24) = dNdxyz(3,:);
            B(3+18*(j-1)+1)    = dNdxyz(1+3*(j-1)+1);	%// B(4,1:3:24) = dNdxyz(2,:);
            B(3+6+18*(j-1)+1)  = dNdxyz(0+3*(j-1)+1);	%// B(4,2:3:24) = dNdxyz(1,:);
            B(4+6+18*(j-1)+1)  = dNdxyz(2+3*(j-1)+1);	%// B(5,2:3:24) = dNdxyz(3,:);
            B(4+12+18*(j-1)+1) = dNdxyz(1+3*(j-1)+1);	%// B(5,3:3:24) = dNdxyz(2,:);
            B(5+18*(j-1)+1)    = dNdxyz(2+3*(j-1)+1);	%// B(6,1:3:24) = dNdxyz(3,:);
            B(5+12+18*(j-1)+1) = dNdxyz(0+3*(j-1)+1);	%// B(6,3:3:24) = dNdxyz(1,:);
        end
        %
        %             // Element stiffness matrix: Symmetry --> lower-triangular part of ke
        %             temp = 0;
        %             for (j=0; j<24; j++) {
        %                 for (k=j; k<24; k++) {
        %                     BDB = 0.0;
        %                     for (l=0; l<6; l++) {
        %                         DB = 0.0;
        %                         for (m=0; m<6; m++){
        %                             DB += D(l+6*m)*B(m+6*j);
        %                         }
        %                         BDB += B(l+6*k)*DB;
        %                     }
        %                     ke(temp+k+300*tid) += detJ*BDB;
        %                 }
        %                 temp += k-j-1;
        
    end
end
