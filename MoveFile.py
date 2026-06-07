import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def moveFile(username):
    data_folder = os.path.join(BASE_DIR, "data")
    selected_data_folder = os.path.join(BASE_DIR, "selectedData")

    user_number = username[4:]
    user_data = f'S{user_number}.pkl'

    for file_name in os.listdir(data_folder):
        source_path = os.path.join(data_folder, file_name)
        if (file_name == user_data):
            shutil.move(source_path, os.path.join(selected_data_folder, file_name))

    for file_name in os.listdir(selected_data_folder):
        source_path = os.path.join(selected_data_folder, file_name)
        if(file_name != user_data):
            shutil.move(source_path, os.path.join(data_folder, file_name))