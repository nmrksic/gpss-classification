function [] = plotOverfit()

expNames = cell(4, 1);

expNames{1} = 'r_liver';
expNames{2} = 'r_breast';
expNames{3} = 'r_pima';
expNames{4} = 'r_heart';

% iterate across all metrics (BIC, BIC light, AIC):

filepaths = cell(3, 1); 

filepaths{1} = 'results/Archive/runs/BICruns/';
filepaths{2} = 'results/Archive/runs/BIClightruns/';
filepaths{3} = 'results/Archive/runs/AICruns/';

dimensions = [6, 9, 8, 13];

colours = cell(3, 1);

colours{1} = 'r.';
colours{2} = 'b.';
colours{3} = 'g.';

data = zeros(3, 20, 1000);
counts = zeros(3, 20); % counts how many elements we added to each stage/metric pair

close all; figure(1); hold on;

xlim([0, 15]);

p = cell(3, 1);

    for metric = 1:3
        
        for dataset = 1:4
            
             if ( ~ (metric == 3 && dataset == 4) ) 
            
                for fold = 1:10

                    filename = [filepaths{metric}, expNames{dataset}, '/fold', num2str(fold), '/searchStats.mat'];

                    load(filename); 

                    TrainTestDiff = trainAccuraciesList - testAccuracciesList;

                    plot( 1 +  0.25* (rand(size(dimensions(dataset), 1)) - 0.5),  TrainTestDiff ( 1:dimensions(dataset) )  , colours{metric} ); hold on;
                    
                    data(metric, 1, (counts(metric, 1) + 1) : (counts(metric, 1) + dimensions(dataset) ) ) = TrainTestDiff ( 1:dimensions(dataset) );
                    
                    counts(metric, 1) = counts(metric, 1) +  dimensions(dataset); 

                        for stage = dimensions(dataset) + 1 : numel(TrainTestDiff)

                            if (trainAccuraciesList(stage) == 0 && testAccuracciesList(stage) == 0)
                                break;
                            end

                            p{metric} = plot ( (stage - dimensions(dataset) + 1) + 0.25* (rand() - 0.5) , TrainTestDiff (stage), colours{metric} ); 
                            
                            data(metric, stage - dimensions(dataset) + 1, ( counts(metric, (stage - dimensions(dataset) + 1)) + 1 ))  = TrainTestDiff ( stage );
                            counts(metric, (stage - dimensions(dataset) + 1)) = counts(metric, (stage - dimensions(dataset) + 1)) +  1;  

                        end

                end
             end
                     
        end


    end

  % Add legend (in title to avoid setting all the legend parameters):
  t = title([' Train minus test accuracy at different search stages: ', '{\color{red}BIC,  }' , '{\color{blue}BIC light  }', 'and  ', '{\color{green}AIC}']);
  set(t, 'FontSize', 24);

  means = zeros(3, 20); 
  
  for i = 1:3
      for j = 1:9
          means(i, j) = mean( squeeze(data(i, j, 1:counts(i,j))));
      end
  end
  
 X =  nan(500, 27);
  
  for j = 1:9
      
      X( 1:counts(1,j), j*3-2) = squeeze(data(1, j, 1:counts(1,j)));
      X( 1:counts(2,j), j*3-1) = squeeze(data(2, j, 1:counts(2,j)));
      X( 1:counts(3,j), j*3)  = squeeze(data(3, j, 1:counts(3,j)));
    
      if j == 1
          positions = [0.8, 1, 1.2];
      else
          positions = [positions, j-0.2, j, j+0.2];
      end
    
  end
  
 size(X)
  
 positions 
  
 group = 1:27;
  
 boxplot(X, group, 'positions', positions);
  

  % xlabel('Search depth level');
  
  plot(1:9, means(1, 1:9), 'r', 'linewidth', 2);
  plot(1:9, means(2, 1:9), 'b');
  plot(1:9, means(3, 1:9), 'g');

  
end

