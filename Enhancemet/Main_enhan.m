clear all;
close all;

for k=1:20
    path_Vis=strcat('C:\Users\Administrator\Documents\image fusion\tno20\ORG_VIS\','VIS',num2str(k),'.jpg');
    path_IR=strcat('C:\Users\Administrator\Documents\image fusion\tno20\ORG_IR\','IR',num2str(k),'.jpg');
    
    [img1, img2, para.name] = PickName(path_Vis, path_IR);
    paraShow1.fig = 'Visible image';
    paraShow2.fig = 'Infrared image';
%     ShowImageGrad(img2, paraShow2);
%     ShowImageGrad(img1, paraShow1);
    %% ---------- Visibility enhancement for visible image--------------
    img1E = Ehn_GF(img1);
    img1 = img1E;
    img1=img1/255;
%     figure, imshow(img1)
    
    %% ---------- Infrared image normalization--------------
    mi = min(img2(:));
    ma = max(img2(:));
    img2 = (img2-mi)/(ma-mi)*255;
    img2=img2/255;
%     figure, imshow(img2)
    
    %% ----------Save Enhanced infrared and visible image----------
    addres_of_image=strcat('C:\Users\Administrator\Desktop\ensamble-Resnet\Enhancemet\Enh_VIS\','Enh_VIS',num2str(k),'.jpg');
    imwrite(img1,addres_of_image,'jpg');
    
    addres_of_image=strcat('C:\Users\Administrator\Desktop\ensamble-Resnet\Enhancemet\Enh_IR\','Enh_IR',num2str(k),'.jpg');
    imwrite(img2,addres_of_image,'jpg');

end
