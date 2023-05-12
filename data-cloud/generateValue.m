%% generate values
clear all

%% generate arrival instance
load('dataIns')
K = length(dataIns); % no. of instances
theta = 250;
M = 1; % number of values for each instance

for k = 1:K
  instanceName = ['arrival_Instance/arrIns',num2str(k)];
  load(instanceName);
  
   
   N = size(arrIns,1);
   jobvalueCell = cell(M,1);
   jobweightCell = cell(M,1);
   for m = 1:M
   %% generate value and weight for each job
    randv = unifrnd(1,theta,N,1); %uniform distributed in [1,\theta] % values
    
    weightCandidate = [0.01,0.03,0.05];
    randw = floor(unifrnd(1,4,N,1));
    jobweight = weightCandidate(randw)';
    jobValue = randv.*arrIns(:,3).*jobweight;
    
    jobvalueCell{m} = jobValue;
    jobweightCell{m} = jobweight;
   end

    
   %% save value and weight
   valueName = ['values/jobvalue',num2str(k)];
   save(valueName,'jobvalueCell','jobweightCell')
end

