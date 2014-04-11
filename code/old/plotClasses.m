function [] = plotClasses (X, Y)

D = size(X, 2);
if D==1
   plot(X, Y, 'r+')
end
if D==2
    plot(X(Y>0,1), X(Y>0,2), 'r+');
    hold on
    plot(X(Y<=0,1), X(Y<=0,2), 'b+');
    hold off;
end

if D==3

    hold off;
    scatter3(X(Y==1,1), X(Y==1,2), X(Y==1, 3), 'marker', 'o');
    hold on
    scatter3(X(Y==-1,1), X(Y==-1,2), X(Y==-1, 3), 'marker', 'x');
    hold off; 

end


end
