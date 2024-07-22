import os
import shutil

def copy_directories(src_dir, dest_dir):
    # Ensure the source directory exists
    if not os.path.exists(src_dir):
        raise ValueError(f"Source directory '{src_dir}' does not exist")

    # Number of dirs in dest
    N = len(list(filter(lambda path : os.path.isdir(os.path.join(dest_dir, path)), os.listdir(dest_dir))))

    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Iterate through the items in the source directory
    for i, item in enumerate(os.listdir(src_dir)):
        renamed_item = f"env_{i + N}"
        src_path = os.path.join(src_dir, item)

        # Check if the item is a directory
        if os.path.isdir(src_path):
            # Copy the directory and its contents to the destination
            dest_path = os.path.join(dest_dir, renamed_item)
            shutil.copytree(src_path, dest_path)

if __name__ == "__main__":
    source_directory = '/home/atari_ws/data/record_/test/'
    destination_directory = '/home/atari_ws/data/record_/train/'
    
    copy_directories(source_directory, destination_directory)
    print(f"Copied all directories from {source_directory} to {destination_directory}")
