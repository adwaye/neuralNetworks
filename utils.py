#taken from tensorflow tutorial
import numpy as np
import os
import tensorflow as tf

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


def update_dir(nameInit):
 output_directory = nameInit
 if not os.path.exists(output_directory):
            output_directory = os.path.join(output_directory,'1')
 else:
            highest_num = 0
            for f in os.listdir(output_directory):
                if os.path.exists(os.path.join(output_directory, f)):
                    file_name = os.path.splitext(f)[0]
                    try:
                        file_num = int(file_name)
                        if file_num > highest_num:
                            highest_num = file_num
                    except ValueError:
                        print('The file name "%s" is not an integer. Skipping' % file_name)

            output_directory = os.path.join(output_directory, str(highest_num + 1))
 os.makedirs(output_directory)
 return output_directory


def _run(tensor):
    with tf.Session() as sess:
         sess.run(tf.global_variables_initializer())
         res = sess.run(tensor)
    return res


# define a function to list tfrecord files.
def list_tfrecord_file(file_list):
    tfrecord_list = []
    for i in range(len(file_list)):
        current_file_abs_path = os.path.abspath(file_list[i])
        if current_file_abs_path.endswith(".tfrecord"):
            tfrecord_list.append(current_file_abs_path)
            print("Found %s successfully!" % file_list[i])
        else:
            pass
    return tfrecord_list

# Traverse current directory
def tfrecord_auto_traversal(dirPath='/home/amr62/Documents/TheEffingPhDHatersGonnaHate/ARxrays/patchesTFrecord/1'):
    current_folder_filename_list = os.listdir(dirPath) # Change this PATH to traverse other directories if you want.
    if current_folder_filename_list != None:
        print("%s files were found under current folder. " % len(current_folder_filename_list))
        print("Please be noted that only files end with '*.tfrecord' will be load!")
        tfrecord_list = list_tfrecord_file(current_folder_filename_list)
        if len(tfrecord_list) != 0:
            for list_index in xrange(len(tfrecord_list)):
                print(tfrecord_list[list_index])
        else:
            print("Cannot find any tfrecord files, please check the path.")
    return tfrecord_list



if __name__ == "__main__":
  print('loaded some functions')



