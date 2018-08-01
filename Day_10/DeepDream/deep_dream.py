import numpy as np
from functools import partial
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
import urllib.request
import os
import zipfile
import cv2

def main():
    url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
    data = './data'
    model_name = 'inception5h.zip'
    #print(model_name)
    zip_file = './data/inception5h.zip'
    #print(zip_file)
    img_noise = np.random.uniform(size=(224,224,3)) + 150.0
    model_fn = 'tensorflow_inception_graph.pb'

    #tensorflow session
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)

    with tf.gfile.FastGFile(os.path.join(data, model_fn), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    
    #input tensor
    t_input = tf.placeholder(np.float32, name = 'input')
    imagenet_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
    tf.import_graph_def(graph_def, {'input': t_preprocessed})

    #load the convolutional layers
    layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
    
    feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

    print("Number of Layers: ", len(layers))
    print("Total number of feature channels:", sum(feature_nums))

    def strip_consts(graph_def, max_const_size=32):
        """Strip large constant values from graph_def."""
        strip_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = strip_def.node.add()
            n.MergeFrom(n0)
            if n.op == 'Const':
                tensor = n.attr['value'].tensor
                size = len(tensor.tensor_content)
                if size > max_const_size:
                    tensor.tensor_content = "<stripped %d bytes>"%size
        return strip_def
      
    def rename_nodes(graph_def, rename_func):
        res_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = res_def.node.add()
            n.MergeFrom(n0)
            n.name = rename_func(n.name)
            for i, s in enumerate(n.input):
                n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
        return res_def
      
    def showarray(a):
        a = np.uint8(np.clip(a, 0, 1)*255)
        plt.imshow(a)
        plt.show()
        
    def visstd(a, s=0.1):
        '''Normalize the image range for visualization'''
        return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5
    
    def T(layer):
        '''Helper for getting layer output tensor'''
        return graph.get_tensor_by_name("import/%s:0"%layer)
    
    def render_naive(t_obj, img0=img_noise, iter_n=20, step=1.0):
        t_score = tf.reduce_mean(t_obj)
        t_grad = tf.gradients(t_score, t_input)[0]
        
        img = img0.copy()
        for _ in range(iter_n):
            g, _ = sess.run([t_grad, t_score], {t_input:img})
            g /= g.std()+1e-8
            img += g*step
        showarray(visstd(img))
        
    def tffunc(*argtypes):
        placeholders = list(map(tf.placeholder, argtypes))
        def wrap(f):
            out = f(*placeholders)
            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
            return wrapper
        return wrap
    
    def resize(img, size):
        img = tf.expand_dims(img, 0)
        return tf.image.resize_bilinear(img, size)[0,:,:,:]
    resize = tffunc(np.float32, np.int32)(resize)
    
    def calc_grad_tiled(img, t_grad, tile_size=512):
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for y in range(0, max(h-sz//2, sz),sz):
            for x in range(0, max(w-sz//2, sz),sz):
                sub = img_shift[y:y+sz,x:x+sz]
                g = sess.run(t_grad, {t_input:sub})
                grad[y:y+sz,x:x+sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)

    def render_deepdream(t_obj, img0=img_noise, iter_n = 10, step = 1.8, octave_n = 9, octave_scale = 1.1):
        t_score = tf.reduce_mean(t_obj)
        t_grad = tf.gradients(t_score, t_input)[0]
    
        img = img0
        octaves = []
        for _ in range(octave_n-1):
            hw = img.shape[:2]
            lo = resize(img, np.int32(np.float32(hw)/octave_scale))
            hi = img-resize(lo, hw)
            img = lo
            octaves.append(hi)
        
        for octave in range(octave_n):
            if octave>0:
                hi = octaves[-octave]
                img = resize(img, hi.shape[:2])+hi
            for _ in range(iter_n):
                g = calc_grad_tiled(img, t_grad)
                img += g*(step / (np.abs(g).mean()+1e-7))
            
            return img/255.0
    
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret, frame = cap.read()
        #gradient ascent
        output = cv2.resize(render_deepdream(tf.square(T('mixed4c')), frame), (720, 720))

        cv2.imshow('output', output)

        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            cap.release()
            break
if __name__ == "__main__":
    main()