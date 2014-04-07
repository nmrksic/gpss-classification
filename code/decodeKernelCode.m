function [kernelName] = decodeKernelCode(encoderMatrix, dim)
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
 
 isCovSum = 0;
 
 if ( n > 1) % we have  a covSum
    isCovSum = 1;
    msg = ['{@covSum, {'];
 end
 
 
 isCovProd = 0; % used for each term inside the following loop:
 
 for i = 1:n
   
     if lengths(i) > 1 % then need a isCovProd
         isCovProd = 1;
         msg = [msg, '{@covProd, {'];
     end
     
     for j = 1:lengths(i)
        
         % just need a covMask kernel for that dimension
         
         mask = zeros(1, dim); 
         
         mask( encoderMatrix(i, j) ) = 1;
         
         msg = [msg, ' {@covMask, {[']; 
         msg = [msg, num2str(mask)];
         msg = [msg, '], {@covSEiso}}}'];
         
         if (j < lengths(i) ) % need a comma to separate prod. kernels
             msg = [ msg, ','];
         end
         
                  
     end
     
     if ( isCovProd == 1)
         msg = [msg, ' }}'];
     end
 
     if ( i < n) % need a comma to separate summed kernels
             msg = [msg, ','];
     end
     
     isCovProd = 0; % reset for another iteration of this loop.
     
 end
 

 if (isCovSum == 1) %close the two brackets opened in initial covSum
     msg = [msg, ' }}'];
 end

 kernelName = msg; % do not insert ; at the end.
 
end