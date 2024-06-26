function Im = Ruido(X)
n=3;
for i=1:50
    M=X(:, :, i);
    % subplot(1,2,1)
    % imshow(M)
    % x = 2+randperm(6,6);
    % y = 2+randperm(6,6);

    x = randperm(10,3);
    y = randperm(10,3);

    for j=1:n
        if M(x(j),y(j))==1
            M(x(j),y(j))=0;
        else
            M(x(j),y(j))=1;
        end
    end
    X(:, :, i)=M;
    % subplot(1,2,2)
    % imshow(M)
end
    
figure('units','normalized','outerposition',[0 0 1 1])

for i=1:50
    subplot(20,20,i);
    imshow(X(:,:,i));
end
Im = X;
return
