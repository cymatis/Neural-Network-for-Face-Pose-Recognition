import os
from PIL import Image

base_dir = os.getcwd()

re_img = Image.new('L',(32,30),'white')
img = Image.open(os.path.join(base_dir, 'weight_raw/dense0.png'))
temp_img = Image.new('L',(32,1),'white')

for m in range(0,3):
    cropped_img = img.crop((0,m,960,m+1))

    for n in range(0, 30):
        small_area = (32*n,0,32*(n+1),1)
        print(small_area)
        temp_img = cropped_img.crop(small_area)

        re_img.paste(temp_img,(0,n))

    re_img.save(os.path.join(base_dir, 'weight_v' + str(m) + '.png'))

#######################################################
'''
re_img = Image.new('L',(4,3),'white')
img = Image.open(os.path.join(base_dir, 'weight_raw/dense0.png'))
temp_img = Image.new('L',(4,1),'white')

for m in range(0,3):
    cropped_img = img.crop((0,m,960,m+1))

    for n in range(0, 32):
        small_area = (30*(n),0,30*(n+1),1)
        print(small_area)
        temp_img = cropped_img.crop(small_area)

        re_img.paste(temp_img,(0,n))

    re_img.save(os.path.join(base_dir, 'weight_v' + str(m) + '.png'))
'''