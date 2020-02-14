""" MAJOR FUNCTION
For training or re-training single user face
 - user_image_1 : Latest User image with clear face
 - label : Unique Employee username/code for face labelling
 - extension : Default User Image extension is ".jpg"
"""
import face_recognition
from flask import request, jsonify

from fr_packages.save_dataset import save_user_dataset


def train_face():
    req_data = request.get_json()
    print(req_data)
    user_image = req_data['user_image']
    extension = req_data['extension']
    uid = req_data['uid']
    input_image = user_image + extension
    input_path = './training_input/'
    image_matrix = face_recognition.load_image_file(input_path + input_image)
    face_encodings = face_recognition.face_encodings(image_matrix)
    if len(face_encodings) > 0:
        face_encoding = face_encodings[0]
        trained_user = {uid: face_encoding}
        save_user_dataset(uid, trained_user)
        return jsonify({"train" : f"{uid} face attributes saved"})
    else:
        return jsonify({"train" : "Failed. Try different image"})