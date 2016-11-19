import argparse
import sys
import util
import json
from sklearn import decomposition
from keras.optimizers import SGD


def process_arguments(args):
    parser = argparse.ArgumentParser(description='Calculating VGG activation vectors on a folder of images')
    parser.add_argument('--vgg_path', action='store', help='path to location of VGG-16 weights')
    parser.add_argument('--images_path', action='store', help='path to directory of images')
    parser.add_argument('--output_path', action='store', help='path to where to put output json file')
    params = vars(parser.parse_args(args))
    # do check here instead
    return params


def main(model, images_path, output_path):
    activations, filenames = util.calc_activations(model, images_path)
    with open(output_path + '_pca.json', 'w') as outfile:
        pca = decomposition.RandomizedPCA(n_components=300).fit(activations)
        final_acts = pca.transform(activations)
        data = []
        for i in xrange(len(final_acts)):
            entry = {'path': filenames[i], "vector": final_acts[i].tolist()}
            data.append(entry)
        json.dump(data, outfile)


if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])
    vgg_path = params['vgg_path']
    images_path = params['images_path']
    output_path = params['output_path']

    model = util.VGG_16(vgg_path)  # load model
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    main(model, images_path, output_path)
