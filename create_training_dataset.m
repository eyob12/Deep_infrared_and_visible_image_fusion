% input data: take image from coco-dataset 
%Output: concatenated patch image which has focused and unfocused from original, GX and Gy 

clear
f=1;

for k=1:500
input_adress=strcat('path\','COCO_ ',num2str(k),'.jpg');
    imshow(input_adress)   
    im{k}=(imread(input_adress));
    if size(im{k},3)==3
      im{k}=rgb2gray(im{k});

    end
    im_blur1{k}=imgaussfilt(im{k},9,'FilterSize',15);
   % imshow(im_blur1{k})  
    [m,n]=size(((im{k})));
    
    x0=im{k};
    x1=im_blur1{k};
    [Gx0, Gy0] = imgradientxy(x0);
    [Gx1, Gy1] = imgradientxy(x1);

    for i = 1:floor(m/32)
    for j = 1:floor(n/32)
        f=f+1;
        
        im_Block(:,:,1) = x0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,1) = x1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block(:,:,2) = Gx0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,2) = Gx1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block (:,:,3)= Gy0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,3) = Gy1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block=uint8(im_Block);
        im_Block_Blurred=uint8(im_Block_Blurred);

        focus_unfocus=vertcat(im_Block,im_Block_Blurred);
        unfocus_focus=vertcat(im_Block_Blurred,im_Block);
  
    
    addres_of_image=strcat('F:\123\focuse_unfocuse\','sharp_blur_ver1_',num2str(f),'.jpg');
    imwrite(focus_unfocus,addres_of_image,'jpg');
    
    addres_of_image=strcat('F:\123\unfocuse_focuse\','sharp_blur_ver1_',num2str(f),'.jpg');
    imwrite(unfocus_focus,addres_of_image,'jpg');
        
    end
    end
k
end  
%%
for k=501:1000
input_adress=strcat('path\','COCO_ ',num2str(k),'.jpg');
        imshow(input_adress)   
    im{k}=(imread(input_adress));
    if size(im{k},3)==3
      im{k}=rgb2gray(im{k});
    end
    im_blur1{k}=imgaussfilt(im{k},9,'FilterSize',11);

    
    [m,n]=size(((im{k})));
    
        x0=im{k};
    x1=im_blur1{k};
    [Gx0, Gy0] = imgradientxy(x0);
    [Gx1, Gy1] = imgradientxy(x1);

    

    for i = 1:floor(m/32)
    for j = 1:floor(n/32)
        f=f+1;
        
        im_Block(:,:,1) = x0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,1) = x1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block(:,:,2) = Gx0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,2) = Gx1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block (:,:,3)= Gy0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,3) = Gy1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block=uint8(im_Block);
        im_Block_Blurred=uint8(im_Block_Blurred);

        focus_unfocus=vertcat(im_Block,im_Block_Blurred);
        unfocus_focus=vertcat(im_Block_Blurred,im_Block);
  
    
    addres_of_image=strcat('F:\123\focuse_unfocuse\','sharp_blur_ver1_',num2str(f),'.jpg');
    imwrite(focus_unfocus,addres_of_image,'jpg');
    
    addres_of_image=strcat('F:\123\unfocuse_focuse\','sharp_blur_ver1_',num2str(f),'.jpg');
    imwrite(unfocus_focus,addres_of_image,'jpg');
        
    end
    end
k
end  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for k=1001:1500
input_adress=strcat('path\','COCO_ ',num2str(k),'.jpg');
        imshow(input_adress)   
    im{k}=(imread(input_adress));
    if size(im{k},3)==3
      im{k}=rgb2gray(im{k});
    end
    im_blur1{k}=imgaussfilt(im{k},9,'FilterSize',13);

    
    [m,n]=size(((im{k})));
    
        x0=im{k};
    x1=im_blur1{k};
    [Gx0, Gy0] = imgradientxy(x0);
    [Gx1, Gy1] = imgradientxy(x1);

    

    for i = 1:floor(m/32)
    for j = 1:floor(n/32)
        f=f+1;
        
        im_Block(:,:,1) = x0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,1) = x1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block(:,:,2) = Gx0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,2) = Gx1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block (:,:,3)= Gy0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,3) = Gy1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block=uint8(im_Block);
        im_Block_Blurred=uint8(im_Block_Blurred);

        focus_unfocus=vertcat(im_Block,im_Block_Blurred);
        unfocus_focus=vertcat(im_Block_Blurred,im_Block);
  
    
    addres_of_image=strcat('F:\123\focuse_unfocuse\','sharp_blur_ver1_',num2str(f),'.jpg');
    imwrite(focus_unfocus,addres_of_image,'jpg');
    
    addres_of_image=strcat('F:\123\unfocuse_focuse\','sharp_blur_ver1_',num2str(f),'.jpg');
    imwrite(unfocus_focus,addres_of_image,'jpg');
        
    end
    end
k
end  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for k=1501:2000
input_adress=strcat('path\','COCO_ ',num2str(k),'.jpg');
       imshow(input_adress)   
    im{k}=(imread(input_adress));
    if size(im{k},3)==3
      im{k}=rgb2gray(im{k});
    end
    im_blur1{k}=imgaussfilt(im{k},9,'FilterSize',15);

    
    [m,n]=size(((im{k})));
    
        x0=im{k};
    x1=im_blur1{k};
    [Gx0, Gy0] = imgradientxy(x0);
    [Gx1, Gy1] = imgradientxy(x1);

    

    for i = 1:floor(m/32)
    for j = 1:floor(n/32)
        f=f+1;
        
        im_Block(:,:,1) = x0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,1) = x1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block(:,:,2) = Gx0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,2) = Gx1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block (:,:,3)= Gy0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,3) = Gy1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block=uint8(im_Block);
        im_Block_Blurred=uint8(im_Block_Blurred);

        focus_unfocus=vertcat(im_Block,im_Block_Blurred);
        unfocus_focus=vertcat(im_Block_Blurred,im_Block);
  
    
    addres_of_image=strcat('F:\123\focuse_unfocuse\','sharp_blur_ver1_',num2str(f),'.jpg');
    imwrite(focus_unfocus,addres_of_image,'jpg');
    
    addres_of_image=strcat('F:\123\unfocuse_focuse\','sharp_blur_ver1_',num2str(f),'.jpg');
    imwrite(unfocus_focus,addres_of_image,'jpg');
        
    end
    end

k
end  




%%



%valid

for k=2001:2300
input_adress=strcat('path\','COCO_ ',num2str(k),'.jpg');
      imshow(input_adress)   
    im{k}=(imread(input_adress));
    if size(im{k},3)==3
      im{k}=rgb2gray(im{k});
    end
    im_blur1{k}=imgaussfilt(im{k},9,'FilterSize',9);

    
    [m,n]=size(((im{k})));
    
        x0=im{k};
    x1=im_blur1{k};
    [Gx0, Gy0] = imgradientxy(x0);
    [Gx1, Gy1] = imgradientxy(x1);

    

    for i = 1:floor(m/32)
    for j = 1:floor(n/32)
        f=f+1;
        
        im_Block(:,:,1) = x0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,1) = x1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block(:,:,2) = Gx0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,2) = Gx1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block (:,:,3)= Gy0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,3) = Gy1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block=uint8(im_Block);
        im_Block_Blurred=uint8(im_Block_Blurred);

        focus_unfocus=vertcat(im_Block,im_Block_Blurred);
        unfocus_focus=vertcat(im_Block_Blurred,im_Block);
  
    
    addres_of_image=strcat('F:\123\Test_folder\focused_unfocused\','sharp_blur_ver1_',num2str(f),'.jpg');
    imwrite(focus_unfocus,addres_of_image,'jpg');
    
    addres_of_image=strcat('F:\123\Test_folder\unfocused_focused\','sharp_blur_ver1_',num2str(f),'.jpg');
    imwrite(unfocus_focus,addres_of_image,'jpg');
        
    end
    end

k
end  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for k=2301:2600
input_adress=strcat('path\','COCO_ ',num2str(k),'.jpg');
      imshow(input_adress)   
    im{k}=(imread(input_adress));
    if size(im{k},3)==3
      im{k}=rgb2gray(im{k});
    end
    im_blur1{k}=imgaussfilt(im{k},9,'FilterSize',11);

    
    [m,n]=size(((im{k})));
    
        x0=im{k};
    x1=im_blur1{k};
    [Gx0, Gy0] = imgradientxy(x0);
    [Gx1, Gy1] = imgradientxy(x1);

    

    for i = 1:floor(m/32)
    for j = 1:floor(n/32)
        f=f+1;
        
        im_Block(:,:,1) = x0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,1) = x1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block(:,:,2) = Gx0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,2) = Gx1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block (:,:,3)= Gy0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,3) = Gy1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block=uint8(im_Block);
        im_Block_Blurred=uint8(im_Block_Blurred);

        focus_unfocus=vertcat(im_Block,im_Block_Blurred);
        unfocus_focus=vertcat(im_Block_Blurred,im_Block);
  
    
    addres_of_image=strcat('F:\123\Test_folder\focused_unfocused\','sharp_blur_ver1_',num2str(f),'.jpg');
    imwrite(focus_unfocus,addres_of_image,'jpg');
    
    addres_of_image=strcat('F:\123\Test_folder\unfocused_focused\','sharp_blur_ver1_',num2str(f),'.jpg');
    imwrite(unfocus_focus,addres_of_image,'jpg');
        
    end
    end

k
end  
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for k=2601:2900
input_adress=strcat('path\','COCO_ ',num2str(k),'.jpg');
       imshow(input_adress)   
    im{k}=(imread(input_adress));
    if size(im{k},3)==3
      im{k}=rgb2gray(im{k});
    end
    im_blur1{k}=imgaussfilt(im{k},9,'FilterSize',13);

    
    [m,n]=size(((im{k})));
    
        x0=im{k};
    x1=im_blur1{k};
    [Gx0, Gy0] = imgradientxy(x0);
    [Gx1, Gy1] = imgradientxy(x1);

    

    for i = 1:floor(m/32)
    for j = 1:floor(n/32)
        f=f+1;
        
        im_Block(:,:,1) = x0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,1) = x1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block(:,:,2) = Gx0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,2) = Gx1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block (:,:,3)= Gy0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,3) = Gy1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block=uint8(im_Block);
        im_Block_Blurred=uint8(im_Block_Blurred);

        focus_unfocus=vertcat(im_Block,im_Block_Blurred);
        unfocus_focus=vertcat(im_Block_Blurred,im_Block);
  
    
    addres_of_image=strcat('F:\123\Test_folder\focused_unfocused\','sharp_blur_ver1_',num2str(f),'.jpg');
    imwrite(focus_unfocus,addres_of_image,'jpg');
    
    addres_of_image=strcat('F:\123\Test_folder\unfocused_focused\','sharp_blur_ver1_',num2str(f),'.jpg');
    imwrite(unfocus_focus,addres_of_image,'jpg');
        
    end
    end

k
end  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for k=2901:3200
input_adress=strcat('path\','COCO_ ',num2str(k),'.jpg');
      imshow(input_adress)   
    im{k}=(imread(input_adress));
    if size(im{k},3)==3
      im{k}=rgb2gray(im{k});
    end
    im_blur1{k}=imgaussfilt(im{k},9,'FilterSize',15);

    
    [m,n]=size(((im{k})));
    
        x0=im{k};
    x1=im_blur1{k};
    [Gx0, Gy0] = imgradientxy(x0);
    [Gx1, Gy1] = imgradientxy(x1);

    

    for i = 1:floor(m/32)
    for j = 1:floor(n/32)
        f=f+1;
        
        im_Block(:,:,1) = x0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,1) = x1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block(:,:,2) = Gx0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,2) = Gx1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block (:,:,3)= Gy0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,3) = Gy1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block=uint8(im_Block);
        im_Block_Blurred=uint8(im_Block_Blurred);

        focus_unfocus=vertcat(im_Block,im_Block_Blurred);
        unfocus_focus=vertcat(im_Block_Blurred,im_Block);
  
    
    addres_of_image=strcat('F:\123\Test_folder\focused_unfocused\','sharp_blur_ver1_',num2str(f),'.jpg');
    imwrite(focus_unfocus,addres_of_image,'jpg');
    
    addres_of_image=strcat('F:\123\Test_folder\unfocused_focused\','sharp_blur_ver1_',num2str(f),'.jpg');
    imwrite(unfocus_focus,addres_of_image,'jpg');
        
    end
    end

k
end  

