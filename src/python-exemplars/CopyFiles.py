'''
Created on 8/16/2025 at 4:41 AM
By yuvaraj
Module Name: CopyFiles
'''
import shutil
import os
import sys

def count_files_in_directory(directory):
    """Count all files inside a directory (recursively)."""
    file_count = 0
    for root, _, files in os.walk(directory):
        file_count += len(files)
    return file_count

def copy_directory_with_progress(src, dest):
    """
    Copies all folders and files from src to dest with progress tracking.
    """
    if not os.path.exists(dest):
        os.makedirs(dest)

    total_files = count_files_in_directory(src)
    copied_files = 0

    for root, dirs, files in os.walk(src):
        # Create corresponding directories in destination
        relative_path = os.path.relpath(root, src)
        dest_dir = os.path.join(dest, relative_path)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        # Copy files with progress
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_dir, file)

            shutil.copy2(src_file, dest_file)
            copied_files += 1

            # Print progress
            progress = (copied_files / total_files) * 100
            sys.stdout.write(f"\rCopying... {progress:.2f}% ({copied_files}/{total_files} files)")
            sys.stdout.flush()

    print("\n✅ Copy completed!")

def main():
    # Example usage
    src_path = r"This PC\Apple iPhone\Internal Storage"
    dest_path = r"F:\iPhone13-backup\Pics"

    copy_directory_with_progress(src_path, dest_path)

if __name__ == "__main__":
    main()
