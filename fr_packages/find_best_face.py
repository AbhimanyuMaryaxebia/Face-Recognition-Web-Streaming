import numpy as np
import face_recognition

""" HELPING FUNCTION
For Find best matching face from all known faces
 - dataset_face_names : Array of names of trained faces
 - dataset_face_attributes : Array of attributes of trained faces
 - unknown_face_attributes : Array of attributes of unknown faces
 - tolerance : Given tolerance
"""
def find_best_face(dataset_face_names, dataset_face_attributes, unknown_face_attributes, tolerance):
    face_distances = face_recognition.face_distance(dataset_face_attributes, unknown_face_attributes)
    best_match_index = np.argmin(face_distances)
    if face_distances[best_match_index] < tolerance:
        percent = int((1-face_distances[best_match_index])*100)
        best_match_name = dataset_face_names[best_match_index]
        return best_match_name + " " + str(percent) + "%"
    else:
        return "un-identified"