function result=Feature_T(data_T)
N = length(data_T);
TF1 = max(data_T);        
TF2 = min(data_T);   
TF3 = mean(data_T);   
TF4 = var(data_T);        
TF5 = std(data_T);       
TF6 = kurtosis(data_T);      
TF7 = skewness(data_T);       
TF8 = rms(data_T);          
TF9 = TF1-TF2;                 
TF10 = mean(abs(data_T));      
TF11 = TF8/TF10;               
TF12 = max(abs(data_T))/TF8;   
TF13 = TF9/TF8;                
TF14 = TF9/TF10;                
TF15 = TF9/mean(sqrt(abs(data_T)))^2;   
TF16 = 1/N*sum((data_T).^3);    
TF17 = TF4/TF10;                
TF18 = max(abs(data_T))/TF4;    
result=[TF1 TF2 TF3 TF4 TF5 TF6 TF7 TF8 TF9 TF10 TF11 TF12 TF13 TF14 TF15 TF16 TF17 TF18];
end