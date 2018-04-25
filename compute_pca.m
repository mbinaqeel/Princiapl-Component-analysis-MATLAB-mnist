function [evecs,evals] = compute_pca(X)
copyX=X';
x_mean = mean(copyX);
copyX = bsxfun(@minus, copyX, x_mean);
trans_copyX = copyX';
[N,D]=size(copyX);

if(N>D)
    S=(1/N)*(trans_copyX*copyX);
    [evecs,evals]=eig(S);
    [evals,I]=sort(diag(evals),'descend');
    evecs = evecs(:, I);
end

if(N<D)
    S=(1/N)*(copyX*trans_copyX);
    [evecs,evals]=eig(S);
    [evals,I]=sort(diag(evals),'descend');
    evecs = evecs(:, I);
    evecs=trans_copyX*evecs;
end

discarded_evals = ones(D,1);
for i=1:D
    for j=i:D
        discarded_evals(i) = discarded_evals(i) + evals(j,:);
    end
end


Ms = [1 10 50 250 784];
image_no = 5;
iterations = size(Ms,2);
x_Hat = zeros(iterations,784);
for ind=1:iterations
    M1 = Ms(ind);
    alpha = zeros(M1);
    for i = 1:M1
        diff1 = X(:,image_no)' - x_mean;
        alpha(i) = (diff1) * evecs(:,i) ;
    end
    temp1 = 0;
    for i = 1:M1
        temp1 = temp1 + alpha(i) * evecs(:,i);
    end

    x_Hat(ind,:) = x_mean + temp1';
end

% it plots figure 12.3
subplot(1,5,1),axis image
imagesc(reshape(x_mean,28,28)')
title('Mean');
for i = 1:4
    subplot(1,5,i+1),axis image
    imagesc(reshape(evecs(:,i),28,28)')
    title(strcat('\lambda = ',num2str(evals(i)/10000),'e+05'));
    disp(evals(i));
end
%it plots figure 12.4
% subplot(1,2,1),axis image
% plot(evals);
% subplot(1,2,2),axis image
% plot(discarded_evals);

%it plots figure 12.5
% subplot(1,iterations+1,1),axis image
% imagesc(reshape(X(:,image_no),28,28)')
% title('Original');
% for i = 1:iterations
%     subplot(1,iterations+1,i+1),axis image
%     imagesc(reshape(x_Hat(i,:)',28,28)');
%     title(strcat('M = ',num2str(Ms(i))));
% end

end