load('matlab.mat');
allSamples = vertcat(perfres.Samples);
averange = varfun(@median, allSamples, 'InputVariables', 'MeasuredTime', 'GroupingVariables', 'Name');
stdeviat = varfun(@std, allSamples, 'InputVariables', 'MeasuredTime', 'GroupingVariables', 'Name');
T = table(averange.Name,averange.GroupCount,averange.median_MeasuredTime,stdeviat.std_MeasuredTime,...
    'VariableNames',{'Function_Name','Test_number','Mean_time','Std_time'});
