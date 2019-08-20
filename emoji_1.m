D = ['./1/'; './2/'; './3/'; './4/'; './5/'];
data = [];
for i = 1:5
    S = dir(fullfile(D(i,:),'*.jpg'));
    for k = 1:numel(S)
        F = fullfile(D(i,:),S(k).name);
        I = rgb2gray(imresize(imread(F), [100 100]));
        I(I >= 127) = 255;
        I(I < 127) = 0;
        a = reshape(I', [1 10000]);
        a = [(i -1)  a];
        data = [data; a];
        imwrite(I, strcat('./100x100_5/',S(k).name), 'jpg');
    end
end
csvwrite('100_5.csv',data);