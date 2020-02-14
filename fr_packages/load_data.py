import os
import pickle
""" HELPING FUNCTION
For Find best matching face from all known faces
 - dataset_face_names : Array of names of trained faces
 - dataset_face_attributes : Array of attributes of trained faces
 - unknown_face_attributes : Array of attributes of unknown faces
 - tolerance : Given tolerance
"""
def load_trained_data():
    all_face_encodings = {}
    path = "./trained_data_set/"
    trained_datasets = os.listdir(path)
    for dataset in trained_datasets:
        if dataset==".DS_Store":
            continue
        with open(path + dataset, 'rb') as f:
            attributes = pickle.load(f)
        all_face_encodings.update(attributes)
    return all_face_encodings