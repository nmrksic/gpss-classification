function newEncoder = sortEncoder(encoderMatrix)
% function to rearrange elements of a sum of products encoder matrix to
% print in order (i.e. SE1 + SE2 * SE3 instead of SE3 * SE2 + SE1

% First sort each row by its columns. 

 n = size(encoderMatrix, 1);
 m = size(encoderMatrix, 2);
 
 lengths = zeros(n, 1);
   
 for i = 1:n
     
     lengths(i) = nnz(encoderMatrix(i, :));
     
     if(lengths(i)==0) % n should contain the actual number of product terms
         n = i-1;
         break;
     end
     
 end
 
 newEncoder = zeros(n, max(lengths)); 
 
 for i = 1:n
     newEncoder(i, 1:lengths(i)) = sort (encoderMatrix(i, find( encoderMatrix(i, :))  ) ); 
 end
 
 newEncoder = sortrows (newEncoder);