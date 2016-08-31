import argparse
import sys
import numpy as np
import os
from os.path import isfile, join
import h5py
from PIL import Image
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.optimizers import SGD
import json
from scipy.spatial import distance
import operator


def process_arguments(args):
    parser = argparse.ArgumentParser(description='Calculating VGG activation vectors on a folder of images')
    parser.add_argument('--vgg_path', action='store', help='path to location of VGG-16 weights')
    parser.add_argument('--images_path', action='store', help='path to directory of images')
    parser.add_argument('--output_path', action='store', help='path to where to put output json file')
    parser.add_argument('--num_dimensions', action='store', default=2,
                        help='dimensionality of t-SNE points (default 2)')
    parser.add_argument('--perplexity', action='store', default=30, help='perplexity of t-SNE (default 30)')
    params = vars(parser.parse_args(args))
    # do check here instead
    return params


def get_image(path):
    try:
        img = Image.open(path)
    except:
        print("Error loading %s: skipping" % path)
        return None
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224), Image.ANTIALIAS)
    im = np.array(img.getdata(), np.uint8)
    im = im.reshape(img.size[1], img.size[0], 3).astype(np.float32)
    im[:, :, 0] -= 123.68  # mean-centering and transposition is probably unecessary
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 103.939
    im = im.transpose((2, 0, 1))
    im = np.expand_dims(im, axis=0)
    return im


def VGG_16(weights_path):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    print("finished loading VGGNet")
    return model


def main(model, images_path, tsne_path, tsne_dimensions, tsne_perplexity):
    # get images
    candidate_images = [f for f in os.listdir(images_path) if isfile(join(images_path, f))]
    # analyze images and grab activations
    #images = []
    with open(tsne_path, 'w') as outfile:
        for idx, image_path in enumerate(candidate_images):
            file_path = join(images_path, image_path)
            try:
                image = get_image(file_path);
                if image is not None:
                    print "getting activations for %s %d/%d" % (image_path, idx, len(candidate_images))
                    acts = model.predict(image)[0]
                    #images.append((file_path, acts))
                    entry = {'path': file_path, "vector": acts.tolist()}
                    json.dump(entry, outfile)
            except:
                print('Something happened with get_image()')

    data = []


        outfile.write('[')
        for path, vector in images:
            print('trying to jsonify vector: ' + str(type(vector.tolist())))
            entry = {'path': path, "vector": vector.tolist()}
            #data.append(entry)
            json.dump(entry, outfile)
            #json.dump(data, outfile)

        outfile.write(']')


if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])
    vgg_path = params['vgg_path']
    images_path = params['images_path']
    tsne_path = params['output_path']
    tsne_dimensions = int(params['num_dimensions'])
    tsne_perplexity = int(params['perplexity'])

    model = VGG_16(vgg_path)  # load model
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    main(model, images_path, tsne_path, tsne_dimensions, tsne_perplexity)
