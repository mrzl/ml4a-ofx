import argparse
import sys
from keras.optimizers import SGD
import numpy as np
import json, codecs
from scipy.spatial import distance
import operator
import os
import shutil
from sklearn import decomposition

import util


def process_arguments(args):
    parser = argparse.ArgumentParser(description='Saving similar images')
    parser.add_argument('--vgg_path', action='store', help='path to location of VGG-16 weights')
    parser.add_argument('--main_image_path', action='store', help='path to them main image')
    parser.add_argument('--training_images_path', action='store', help='path to directory of the training images. needed for pca')
    parser.add_argument('--activations_json_path', action='store', help='path to json file containing activations')
    parser.add_argument('--output_path', action='store', help='path to where to put output json file')
    parser.add_argument('--num_images', action='store', help='amount of images to copy')
    params = vars(parser.parse_args(args))
    # do check here instead
    return params


if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])
    vgg_path = params['vgg_path']
    main_image_path = params['main_image_path']
    output_path = params['output_path']
    activations_json_path = params['activations_json_path']
    training_images_path = params['training_images_path']
    num_images = int(params['num_images'])

    model = util.VGG_16(vgg_path)  # load model
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    print('done creating model')

    print('getting activations for main image')
    main_image = util.get_image(main_image_path)
    acts = model.predict(main_image)[0]
    print('calculating activations from training images in order to initialize the PCA')
    activations, filenames = util.calc_activations(model, training_images_path)

    pca = decomposition.RandomizedPCA(n_components=300).fit(activations)
    # pca.
    # pca.fit(activations_collection)
    print(acts.shape)
    final_acts = pca.transform(acts)
    print(final_acts.shape)
    #final_acts = [-2.446349714622012, 1.3818476574673058, 1.3724461917432493, -1.3425478124539094, -1.8032602344780604,
    #              -0.43487324272899247, -0.6648452534781493, -0.3817656948763483, 0.04404506073616678,
    #              0.1057471474521155, -0.03095613958095083, -0.173760581665124, 0.08924653035253771,
    #              -0.8857581607273216, 0.035040931275045324, 0.41971574667325073, -0.020907333840135826,
    #              0.06218461364092781, -0.5309471435093558, 0.0570825462912279]

    print('done getting activations for main image')

    model = ""

    print('loading json vector database')
    text = codecs.open(activations_json_path, 'r', encoding='utf-8').read()
    print('done opening file')
    bnew = json.loads(text)
    print('done jsonifying')
    images = np.array(bnew)
    print('done numpyifying')

    print('starting to calculate distances')
    distances = []
    i = 0
    for vector in images:
        d = distance.euclidean(np.asarray(vector['vector'], dtype=np.float32), final_acts)
        distances.append((vector['path'], d))
        i += 1
    print('done. sorting')
    # sorting by second parameter
    distances.sort(key=operator.itemgetter(1))

    print('done sorting. fixing the output path')
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    print('Checking ' + str(len(distances)) + ' distances and copying ' + str(num_images) + ' to ' + output_path)
    counter = 0
    for i in distances:
        #print('distance: ' + str(i[1]))
        if counter < num_images:
            similar_image = i[0]
            path, filename = os.path.split(similar_image)

            shutil.copyfile(similar_image, output_path + '/' + filename)
            print('Copying ' + similar_image + ' to ' + output_path + filename + ' because distance was ' + str(i[1]))
            counter += 1