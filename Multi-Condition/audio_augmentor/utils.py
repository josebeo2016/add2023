import os

def recursive_list_files(path, file_type=["wav", "mp3", "flac"]):
    """Recursively lists all files in a directory and its subdirectories"""
    files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            real_file_type = filename.split(".")[-1]
            if (real_file_type in file_type):
                files.append(os.path.join(dirpath, filename))
    return files