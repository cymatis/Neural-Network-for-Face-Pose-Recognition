import os
import shutil

####################################################

base_dir = os.path.join(os.getcwd(),'Face dataset')
target = os.path.join(base_dir, 'all_test2.txt') # all_train.txt, all_test1.txt, all_test2.txt
output_dir = os.path.join(base_dir, 'test') # train, val, test

list = open(target, 'r') # read txt file

line = list.readline() # read first line

####################################################

while(line): # cycle until no more lines left, None = escape cycle
    line = line.replace('\n', '') # line read

    left_dir, right_dir = line.split('/') # extracting file name

    name, direction, emotion, dummy1, dummy2 = right_dir.split('_') # extracting direction

    target_dir = os.path.join(base_dir + '/faces', line) # target image dir

    destination_dir = os.path.join(output_dir, direction) # direction to file location
    

    shutil.copy(target_dir, destination_dir) # copy

    line = list.readline() # read new line for while activation
    '''
    print(line)
    print(left_dir)
    print(right_dir)
    print (direction)
    print (target_dir)
    print (destination_dir)
    '''

list.close() # file close
