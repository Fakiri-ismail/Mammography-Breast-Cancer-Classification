%clear all;

vgg19;
convnet = vgg19;  
 
images = imageDatastore('C:\Users\DELL XPS\Desktop\Matlab_VGG\cliches', 'IncludeSubfolders', true, 'LabelSource', 'foldernames'); 
 
tbl = countEachLabel(images)
minSetCount = min(tbl{:,2})

images = splitEachLabel(images, minSetCount, 'randomize'); 
images.ReadFcn = @(filename)readAndPreprocessImage(filename); 
 
[trainingImages,validationImages, testingImages] = splitEachLabel(images,0.7, 0.1,'randomized'); 
 
layersTransfer = convnet.Layers(1:end-3); 
 
numClasses = numel(categories(trainingImages.Labels))

layers = [layersTransfer 
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)    
    softmaxLayer   
    classificationLayer];

miniBatchSize =10;
numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize); 
options = trainingOptions('sgdm',...    
    'MiniBatchSize',miniBatchSize,...    
    'MaxEpochs',10,...
    'InitialLearnRate',1e-4,...     
    'Verbose',false,...     
    'Plots','training-progress',...     
    'ValidationData',validationImages,...   
    'ValidationFrequency',numIterationsPerEpoch);

%Trainig
VGGnetTransfer = trainNetwork(trainingImages,layers,options); 

%Prediction
predictedLabels = classify(VGGnetTransfer,testingImages);

% Matrice de confusion
[C,order] = confusionmat(predictedLabels, testingImages.Labels);
gather(C);
gather(order);

%%Plot matrice de confusion
cm = confusionchart(predictedLabels,testingImages.Labels);

%%Calcul de precision, recall et f1-score
precision = diag(C)./sum(C,2);
recall = diag(C)./sum(C,1)';

%%Print Statistic
fprintf('%d\n',order(1))
fprintf('precision : %.3f.\n',precision(1));
fprintf('recall : %.3f.\n',recall(1));
fprintf('f1-score : %.3f.\n\n',2*precision(1)*recall(1)/(precision(1)+recall(1)));

fprintf('%d\n',order(2))
fprintf('precision : %.3f.\n',precision(2));
fprintf('recall : %.3f.\n',recall(2));
fprintf('f1-score : %.3f.\n\n',2*precision(2)*recall(2)/(precision(2)+recall(2)));

fprintf('Accuracy : %.2f %%.\n\n', sum(diag(C))/sum(sum(C))*100) 

% Affichage des images
numTestImages = numel(testingImages.Labels);
idx = randperm(numTestImages,4);
figure 
for i = 1:numel(idx)   
    subplot(2,2,i)    
    I = readimage(testingImages,idx(i));     
    label = predictedLabels(idx(i));     
    imshow(I)    
    title(char(label))
end 
 
function Iout = readAndPreprocessImage(filename) 
 
I = imread(filename); 
if ismatrix(I) 
    I = cat(3,I,I,I);
end 
  
Iout = imresize(I, [224 224]); 
 
end