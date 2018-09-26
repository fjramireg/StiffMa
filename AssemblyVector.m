% Assembly process
for e = 1:nel
    
    % Global DOFs of the element e
    edof = conect(e,:);
    
    % Nodal coordinates of the element e
    X = coord(edof,:);
    
    % Index and entries values of the global rigid matrix
    edofs = [3*edof-2; 3*edof-1; 3*edof];
    indx = repmat(edofs(:),1,24);
    indy = indx';
    iK(:,e) = indx(:);
    jK(:,e) = indy(:);
    [ke, fe] = Brick8nodes(X,T(edof),BC.Tamb,MP.alpha,MP.D);
    F(edofs(:),1) = F(edofs(:),1)+fe;
    Ke(:,e) = ke(:);
    
end
