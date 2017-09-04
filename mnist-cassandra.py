# coding=utf-8
import os
import tensorflow as tf
from flask import Flask, request,render_template,jsonify
from werkzeug import secure_filename
import time
from cassandra.cluster import Cluster
from convert_pic import *
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG','jpeg','JPEG'])
#admitted forms

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getcwd()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
#limit the upload files

#load the exist tensorflow graph
sess = tf.Session()
saver = tf.train.import_meta_graph("./checkpoint/model.ckpt.meta")
saver.restore(sess, './checkpoint/model.ckpt')
keep_prob = tf.get_default_graph().get_tensor_by_name('dropout/Placeholder:0')
x = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
y_conv = tf.get_default_graph().get_tensor_by_name('fc2/add:0')

#connect to the cassandra data base
cluster = Cluster(['0.0.0.0'])
session = cluster.connect()
session.execute("create KEYSPACE if not exists mnist_database WITH replication = {'class':'SimpleStrategy', 'replication_factor': 2};")
session.execute("use mnist_database")
session.execute("create table if not exists images(id uuid, digits int, upload_image list<float>, upload_time timestamp, primary key(id));")


#judge the files is admited
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

#the upload page
@app.route('/upload')
def upload_test():
    return render_template('upload.html')


# upload picture and return the number
@app.route('/api/upload', methods=['POST'], strict_slashes=False)
def api_upload():
    f = request.files['file']  # use name 'file' to get the file
    if f and allowed_file(f.filename):
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        array = imageprepare(filename)
        prediction = tf.argmax(y_conv, 1)
        y_pre = prediction.eval(feed_dict={x: [array], keep_prob: 1.0}, session=sess)
        uploadtime=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        session.execute("INSERT INTO images(id, digits, upload_image, upload_time) values(uuid(), %s, %s, %s)",[y_pre[0], array, uploadtime])
        return jsonify({'The digits in this image is':str(y_pre[0])})


if __name__ == '__main__':
    app.run(port=5000)