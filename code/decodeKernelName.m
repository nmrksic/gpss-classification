function [kernelName] = decodeKernelName(encoderMatrix)
% this function takes an encoder matrix and returns the string with the
% name of that kernel.

 n = size(encoderMatrix, 1);
 m = size(encoderMatrix, 2);
 
 msg = [];
 
 lengths = zeros(n, 1);
   
 for i = 1:n
     
     lengths(i) = nnz(encoderMatrix(i, :));
     
     if(lengths(i)==0) % n should contain the actual number of product terms
         n = i-1;
         break;
     end
     
 end
 
 for i = 1:n
     if i > 1
         msg = [msg, ' + '];
     end
     
     for j = 1:lengths(i)
        
         if j>1
            msg = [msg, 'xSE', num2str(encoderMatrix(i, j))];
         else 
             msg = [msg, 'SE', num2str(encoderMatrix(i, j))];
         end
         
     end
 end
 
kernelName = msg;

end