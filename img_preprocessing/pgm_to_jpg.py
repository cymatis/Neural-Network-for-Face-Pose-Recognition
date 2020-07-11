import os
from PIL import Image 

base_dir = os.getcwd()

db_dir = os.path.join(base_dir, 'Face dataset')

####################################################

# location of original face dataset (pgm)
train_dir = os.path.join(base_dir, 'Face dataset/train')
validation_dir = os.path.join(base_dir, 'Face dataset/val')
test_dir = os.path.join(base_dir, 'Face dataset/test')

####################################################

# location of converted face dataset (png 8bit grayscale) !!! jpg does not support 8bit grayscale !!!
jpg_train_dir = os.path.join(base_dir, 'Face dataset_png/train')
jpg_validation_dir = os.path.join(base_dir, 'Face dataset_png/val')
test_dir = os.path.join(base_dir, 'Face dataset_png/test')

img_folder_list = os.listdir(db_dir)

####################################################

for list in img_folder_list:
    purpose_dir = os.path.join(db_dir, list)
    direction_list = os.listdir(purpose_dir)
    
    for direction in direction_list:
        direction_dir = os.path.join(purpose_dir, direction)
        out_direction_dir = direction_dir.replace('Face dataset','Face dataset_png')

        img_list = os.listdir(direction_dir)

        for img_name in img_list:
            img_dir = os.path.join(direction_dir, img_name)
            im = Image.open(img_dir)
            im = im.convert('L') # 'L' 8bit grayscale

            name, ext = img_name.split('.')
            path_dir = os.path.join(direction_dir, name+ext)
            print(path_dir)

            if not(os.path.isdir(out_direction_dir)):
                os.makedirs(os.path.join(out_direction_dir))

            im.save(os.path.join(out_direction_dir, name + '.png')) # save pgm image to 8bit grayscale png

