import pickle
""" HELPING FUNCTION
For Saving Trained face of single user
 - label : For naming dataset file
 - trained_user : Data that needs to be saved in dat file
"""
def save_user_dataset(label, trained_user):
    dataset_file = "dataset_" + label + ".dat"
    dataset_path = "./trained_data_set/"
    with open(dataset_path + dataset_file, 'wb') as f:
        pickle.dump(trained_user, f)
    print("Trained data for :", label, " - Saved Successfully!")