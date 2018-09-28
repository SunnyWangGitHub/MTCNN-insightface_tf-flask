#python3
# -*- coding: utf-8 -*-
# File  : test_flask.py
# Author: Wang Chao
# Date  : 2018/9/5
import scipy
import flask
from werkzeug.utils import secure_filename
import os
import tensorflow as tf

from tf_insightface.apps import example
from MT import test_img

app = flask.Flask("web_app")

def redirect_upload():
    # example.load_model()
    return flask.render_template(template_name_or_list='upload_image.html')
app.add_url_rule(rule='/',endpoint='homepage',view_func=redirect_upload)

def upload_image():
    global secure_file_name
    if flask.request.method == "POST":
        img_file = flask.request.files["image_file"]
        secure_file_name = secure_filename(img_file.filename)
        img_path = os.path.join(app.root_path+'/upload',secure_file_name)
        img_file.save(img_path)
        print("image uploaded successfully")
        print(img_path)

        L = test_img.detect(img_path) #调用MTCVNN
        example.face_recognition(L)   #调用出特征向量


        return flask.redirect(flask.url_for(endpoint='homepage'))
    return "image upload failed"
app.add_url_rule(rule="/upload/", endpoint="upload", view_func=upload_image, methods=["POST"])


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=7777,debug=True)
