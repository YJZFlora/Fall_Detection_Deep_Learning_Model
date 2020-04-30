import os

# change this to the path to your directory
root_dir = '/path/to/directory'

def extract_landmark():
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            # get file name without extension
            file_name = os.path.splitext(file)[0]
            extension = os.path.splitext(file)[1].lower()

            if (extension == '.mp4' or extension == '.avi' or extension == '.mov') and (not os.path.isdir(os.path.join(subdir, file_name))):
                # create a folder with the same file name
                folder_dir = os.path.join(subdir, file_name)
                os.mkdir(folder_dir)

                # run the command to extract features
                cmd_str = './build/examples/openpose/openpose.bin --video "' + os.path.join(subdir, file) + '" --write_json "' + folder_dir + '" --display 0 --render_pose 0 --face --hand' 
                os.system(cmd_str)


if __name__ == "__main__":
    extract_landmark()


