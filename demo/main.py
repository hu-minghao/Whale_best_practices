# -*- coding:utf-8 -*-
import io
import numpy as np
from PIL import Image
from flask import Flask,render_template,request,Response,redirect,url_for
from werkzeug.utils import secure_filename
import flask
import tensorflow as tf
import os
import time
from tensorflow.python.client import device_lib

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

label_id_name_dict = \
    {
    0:"daffodil",
    1:"lilyvalley",
    2:"snowdrop"
    }

app = Flask(__name__)
model_name = r'./tmp/output_graph.pb'

def create_graph():
    with tf.gfile.FastGFile( model_name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        print('creat graph success')

create_graph()
with tf.Session() as sess:
    # 主页为上传
    @app.route('/')
    def index():
        return redirect(url_for('upload'))

    @app.route("/predict/<imgname>")
    def predict(imgname):
        data = {}
        print("Hello")
        image_path = './static/uploads/img/'+imgname
        print(image_path)
        img = Image.open(image_path)
        print('image shape', img.size)
        if img:
            print("world")
            output_buffer = io.BytesIO()
            img.save(output_buffer, format='JPEG')
            image = output_buffer.getvalue()
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            # 输入图像（jpg格式）数据，得到softmax概率值（一个shape=(1,1008)的向量）
            start_time = time.time()
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image})
            end_time = time.time()
            print('detect cost time {}'.format(end_time - start_time))
            # 将结果转为1维数据
            predictions = np.squeeze(predictions)
            print(predictions)
            index = np.argmax(predictions, 0)
            data["predictions"] = []
            res = {}
            res['tagName'] = label_id_name_dict[index]
            res['probability'] = predictions[index].item()
            data["predictions"].append(res)
            print(data)
        else:
            print('No data')
        # return the data dictionary as a JSON response
        return render_template('imgcontrast.html', imgname=imgname, doc=data)


    # 上传图片的页面
    @app.route('/upload', methods=['POST', 'GET'])
    def upload():
        if request.method == 'POST':
            f = request.files['file']
            basepath = os.path.dirname(__file__)  # 当前文件所在路径
            filetype = str(os.path.splitext(secure_filename(f.filename))[-1])
            if (filetype == ".jpg" or filetype == ".png"):
                upload_path = os.path.join(basepath, 'static/uploads/img',
                                           secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
                f.save(upload_path)
                return redirect('/predict/' + secure_filename(f.filename))
            else:
                return "上传文件格式不支持"

        return render_template('upload.html')

    # if this is the main thread of execution first load the model and
    # then start the server
    if __name__ == "__main__":
        print(("* Loading tensorflow model and Flask starting server..."
               "please wait until server has fully started"))
        app.run(host='0.0.0.0', port=5000, debug=True)
        print(device_lib.list_local_devices())
