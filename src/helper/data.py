

import platform
import os
import shutil
import zipfile

def load_kaggle_dataset(author_name, dataset_name, on_google_colab, 
                        download_path, goal_path,
                        kaggle_local_path="./", kaggle_file_name="kaggle.json"):
    """
    Example:

    >>> load_kaggle_dataset(
            author_name="joosthazelzet", 
            dataset_name="lego-brick-images",
            on_google_colab=True,            
            download_path="/content/", 
            goal_path=f"/content/dataset",         
            kaggle_local_path="./", 
            kaggle_file_name="kaggle.json")
    """
    # variables
    print("Set some variables...")
    dataset_download_name = f"{author_name}/{dataset_name}"
   
    zip_file_name = f"{dataset_name}.zip"
    zip_file_download_path = os.path.join(download_path, zip_file_name)

    kaggle_file_cur_path = os.path.join(kaggle_local_path, kaggle_file_name)
    kaggle_goal_path = os.path.expanduser("~/.kaggle") if platform.system().lower() == "windows" else "/root/.config/kaggle"
    kaggle_goal_file_path = os.path.join(kaggle_goal_path, kaggle_file_name)

    # make sure that the goal path exist
    os.makedirs(goal_path, exist_ok=True)
    os.makedirs(kaggle_goal_path, exist_ok=True)

    print("Finding and placing the API file...")
    # upload in google colab
    if on_google_colab:
        kaggle_local_path = "./"
        kaggle_file_cur_path = os.path.join(kaggle_local_path, kaggle_file_name)
        if os.path.exists(kaggle_file_cur_path):
            os.remove(kaggle_file_cur_path)

        from google.colab import files
        files.upload()  # choose your local 'kaggle.json' file

    # get the kaggle API file to the right spot
    if os.path.exists(kaggle_goal_file_path):
        os.remove(kaggle_goal_file_path)
    shutil.copy2(kaggle_file_cur_path, kaggle_goal_path)
    os.chmod(kaggle_goal_file_path, 600)    # set right rights
    print(f"Cpopied to: {kaggle_goal_path}")

    # init Kaggle API
    from kaggle.api.kaggle_api_extended import KaggleApi
    print("Autheticating at Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    # make sure the file not exist already
    if os.path.exists(zip_file_download_path):
        os.remove(zip_file_download_path)

    # download kaggle dataset
    print("Downloading dataset...")
    #    -> dataset name just in the https link the last 2 items
    # !kaggle datasets download -d joosthazelzet/lego-brick-images
    api.dataset_download_files(dataset_download_name, path=download_path, unzip=False)

    # Unzip the downloaded dataset
    print("Unzipping dataset...")
    if os.path.exists(goal_path):
        shutil.rmtree(goal_path)
    # !unzip -q "lego-brick-images.zip" -d dataset
    with zipfile.ZipFile(zip_file_download_path, "r") as zip_ref:
        zip_ref.extractall(goal_path)

    # delete zip file
    os.remove(zip_file_download_path)

    print(f"Congratulations! Downloaded successfull '{dataset_name}' from '{author_name}' 🥳😎")





