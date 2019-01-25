classdef ScalingTest_Sc < matlab.perftest.TestCase
    % Performance test to measure the scaling of the code that uses the
    % problem size and data type properties as TestParameter.
    
    
    properties(TestParameter)
        dTE = {'int32','uint32','int64','uint64','double'}; % different data types for indices
        dTN = {'single','double'};                          % different data types for alll ke
        nel = {10,20,30,40};                                % problem size
    end
    
    
    methods(Test)
        
        function [elements, nodes] = TestMeshCreation(~, nel,dTE,dTN)
            [elements, nodes] = CreateMesh(nel,nel,nel,dTE,dTN,0,0);
        end
        
        function [iK, jK] = TestIndexCreation(~, elements)
            [iK, jK] = IndexScalarSymCPU(elements);  
        end
        
    end
end
