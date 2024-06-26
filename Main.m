clc, clearvars, close all;
eficacia_1=0;
eficacia_2=0;
eficacia_3=0;
eficacia_4=0; 


% Imagenes en matriz binaria 20x20

imgx = [ "logo Bluetooth " "logo C++" "logo chatgpt" "Logo creeper" "Logo de mario" ...
         "Logo de python" "Logo de Flash" "Pokebola" "Logo de Twitter" ...
         "Logo de Facebook" "Logo de Apple" "Pez" "Logo de Youtube" ...
         "Logo de Instagram" "Logo de Windows" "Logo de Android" "Corazon" "Logo de Github" ...
         "Logo de Matlab" "Logo de Netflix" "Logo de Nike" "Dota 2" "Logo de Discord" ...
         "Steve" "Signo zodiacal del Virgo" "Signo zodiacal de Aries" "Signo zodiacal de Tauro" ...
         "Signo zodiacal de Geminis" "Signo zodiacal de Cancer" "Signo zodiacal de Leo" ...
         "Signo zodiacal de Libra" "Signo zodiacal de Escorpio" "Signo zodiacal de Sagitario" ...
         "Signo zodiacal de Capricornio" "Signo zodiacal de Acuario" "Signo zodiacal de Piscis" ...
         "logo de telegram" "logo de twitch" "logo de snapchat" "logo de batman" "rayo" "Logo pacman  " "Planta" "Luna" ...
         "Nube" "logo de UPC" "Logo de Spotify" "Logo de Xbox" "Logo de PlayStation" "Logo de Play Store" ];

load imgns.mat X

%% Imagenes con ruido 
XX = Ruido(X);
%% Imagenes con ruido
 %for i=1:cant    
     %figure();
     %imshow(XX(:,:,i));
     %impixelinfo;
 %end
%% Algoritmo  
%  Identidad 50x50
D = eye(50);


% Se trabaja con 3 capas
% Pesos para función cross entropy
W11 = 2*rand(50, 400) - 1; % 1era hidden layer 50 neuronas 
W12 = 2*rand(50, 50) - 1;  % 2da hidden layer 50 neuronas 
W13 = 2*rand(50, 50) - 1;  
%W14 = 2*rand(50, 50) - 1;

% Pesos para función coste de error medio cuadrático
W21 = W11; 
W22 = W12; 
W23 = W13; 
%W24 = W14;

for epoch = 1:40000 
    
 	[W11, W12, W13] = Entrenamiento_CE(W11, W12, W13, X, D); % Cross entropy
    [W21, W22, W23] = Entrenamiento_SE(W21, W22, W23, X, D); % error medio cuadratico
    
    
end   
    N = 50;
    for k = 1:N
        %cross entropy sin ruido
        x = reshape(X(:, :, k), 400, 1);
        
        
        v1 = W11*x;
        y1 = ReLU(v1);
        
        v2 = W12*y1;
        y2 = ReLU(v2);   
        
        v3 = W13*y2;
        y =  Softmax(v3);


        [~,idk1]=max(y);
        if idk1 == k
            eficacia_1 = eficacia_1 + 1;
        end
       

        %cross entropy con  ruido

        xx = reshape(XX(:, :, k), 400, 1);
        

        v1_1 = W11*xx;
        y1_1 = ReLU(v1_1);
        
        v2_1 = W12*y1_1;
        y2_1 = ReLU(v2_1);   
        
        v3_1 = W13*y2_1;
        y_1 = Softmax(v3_1); 

       
        [~,idk3]=max(y_1);
         if idk3 == k
             eficacia_3= eficacia_3 + 1;
         end 
        
         % error medio cuadratico sin ruido
        vv1 = W21*x;
        yy1 = ReLU(vv1);
        
        vv2 = W22*yy1;
        yy2 = ReLU(vv2);   
        
        vv3 = W23*yy2;
        yy =  Softmax(vv3);

        
        [~,idk2]=max(yy);
        if idk2 == k
            eficacia_2 = eficacia_2 + 1;
        end 

        % error medio cuadratico con ruido
        
        vv1_1 = W21*xx;
        yy1_1 = ReLU(vv1_1);
        
        vv2_1 = W22*yy1_1;
        yy2_1 = ReLU(vv2_1);   
        
        vv3_1 = W23*yy2_1;
        yy_1 = Softmax(vv3_1);
        [~,idk4]=max(yy_1);
        if idk4 == k
            eficacia_4 = eficacia_4 + 1;
        end


    end
    

eficacia_1 = eficacia_1/N;
eficacia_2 = eficacia_2/N;
eficacia_3 = eficacia_3/N;
eficacia_4 = eficacia_4/N;

fprintf("Para Cross Entropy \n");
fprintf("La eficencia sin ruido es: %f\n %",eficacia_1);
fprintf("La eficencia con ruido es: %f\n % \n",eficacia_3);

fprintf("Para Error Cuadratico medio \n");
fprintf("La eficencia sin ruido es: %f\n %",eficacia_2);
fprintf("La eficencia con ruido es: %f\n %",eficacia_4);

