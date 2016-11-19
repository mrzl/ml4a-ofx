import sys
import json
import argparse
import os
from scipy.spatial import distance
import shutil


def process_arguments(args):
    parser = argparse.ArgumentParser(description='tSNE on audio')
    parser.add_argument('--tsne_json_path', action='store', help='path to tsne activations json file')
    parser.add_argument('--image_path', action='store', help='path to image you want similar images of')
    parser.add_argument('--output_path', action='store', help='path to where to put output the similar images')
    parser.add_argument('--max_distance', action='store', default=2, help='maximum distance for similarity (normalized)')
    params = vars(parser.parse_args(args))
    # do check here instead
    return params

class NearestNeighbours(object):

    image_filenames = []
    image_paths = []
    points = []

    def parse(self, filename):
        self.filename, self.file_extension = os.path.splitext(filename)
        with open(filename) as file:
            data = json.load(file)
            for entry in data:
                path = entry['path']
                point = entry['point']
                self.image_filenames.append(os.path.basename(path))
                self.image_paths.append(path)
                self.points.append(point)

    def get_similar(self, file, max_distance, output_path):
        print('Checking ' + str(len(self.image_filenames)) + ' other images for similarity')
        index = self.image_filenames.index(file)
        image_vector = self.points[index]
        similar_images_paths = []
        similar_images_filenames = []
        similarity = []
        for p in self.points:
            vector_a = (image_vector[0], image_vector[1])
            vector_b = (p[0], p[1])
            dist = distance.euclidean(vector_a, vector_b)
            if dist < max_distance:
                ind = self.points.index(p)
                similar_images_paths.append(self.image_paths[ind])
                similar_images_filenames.append(self.image_filenames[ind])
                similarity.append(dist)

        if os.path.isdir(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)
        for i in similar_images_paths:
            _ind = similar_images_paths.index(i)
            shutil.copyfile(i, output_path + '/' + similar_images_filenames[_ind])
            print('Copying ' + i + ' to ' + self.filename + ' because distance was ' + str(similarity[_ind]))


if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])

    filename = params['tsne_json_path']
    sample_image = params['image_path']
    output_path = params['output_path']
    max_distance = float(params['max_distance'])

    nn = NearestNeighbours()
    nn.parse(filename)
    nn.get_similar(sample_image, max_distance, output_path)