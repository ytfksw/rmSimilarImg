import os
import shutil
from argparse import ArgumentParser
import numpy as np
from sklearn.cluster import KMeans
from keras.preprocessing import image
from extractor import XceptionExtractor, WHashExtractor


np.random.seed(0)
EXTENSTIONS = ['.png', '.jpg']
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def target_list(input_dir):
    targets = []
    for root, dirs, files in os.walk(input_dir):
        for fname in files:
            _, file_extension = os.path.splitext(fname)
            if not file_extension in EXTENSTIONS:
                continue
            targets.append(os.path.join(root, fname))
    return targets
                   

def sampling(vec, sampled_data_num):
    kmeans = KMeans(n_clusters=sampled_data_num, random_state=0).fit(vec)
    labels = kmeans.labels_
    out = []
    for ind in range(sampled_data_num):
        out.extend(np.random.choice((np.where(labels == ind)[0]), 1))
    return out


def argparser():
    parser = ArgumentParser()
    parser.add_argument("INPUT_DIR", help="input image directory")
    parser.add_argument("OUTPUT_DIR", help="output image directory")
    parser.add_argument("NUM_SAMPLE", type=int, help="number of sample")
    parser.add_argument("--extractor", default="Xception",
                        choices=["Xception", "whash"],
                        help="choise of extractor")
    args = parser.parse_args()
    return args


def main():
    args = argparser()
    targets = target_list(args.INPUT_DIR)
    images = [image.img_to_array(image.load_img(target, target_size=(299, 299)))
            for target in targets]
    images = np.array(images)

    if args.extractor == "Xception":
        extractor = XceptionExtractor()
    elif args.extractor == "whash":
        extractor = WHashExtractor()

    feature_vectors = extractor.feature_extraction(images)
    sampled_index = sampling(feature_vectors, args.NUM_SAMPLE)
    output_targets = [targets[index] for index in sampled_index]

    for output_target in output_targets:
        basename = os.path.basename(output_target)
        shutil.copy(output_target, os.path.join(args.OUTPUT_DIR, basename))
    print('done.')


if __name__ == "__main__":
    main()
