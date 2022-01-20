import numpy as np
from skimage.feature import hog


def zero_padding(image, new_height, new_width):
    height = image.shape[0]
    width = image.shape[1]

    if new_height >= height and new_width >= width:
        vertical_padding = int(np.round((new_height - height) / 2, decimals=0))
        horizontal_padding = int(np.round((new_width - width) / 2, decimals=0))

        out = np.zeros((height + 2 * vertical_padding, width + 2 * horizontal_padding))
        out[vertical_padding: height + vertical_padding, horizontal_padding: width + horizontal_padding] = image

    else:
        out = image

    return out


def get_padded_image_dimensions():
    new_height = 32
    new_width = 32
    return new_height, new_width


def extract_image_features(image, params):
    '''

    Extract features for single image

    :param image: 28x28 gray-scale image (numpy ndarray)
    :param params:
    :return:
    '''

    new_height, new_width = get_padded_image_dimensions()

    image_padded = zero_padding(image, new_height=new_height, new_width=new_width)
    hog_vector = hog(image_padded,
                     orientations=params['number_of_orientations'],
                     pixels_per_cell=params['pixels_per_cell'],
                     cells_per_block=params['cells_per_block'],
                     feature_vector=True)
    return hog_vector


def extract_features_for_image_set(images, params):
    new_height, new_width = get_padded_image_dimensions()
    number_of_images = images.shape[0]

    # (n_blocks_row * n_blocks_col * n_cells_row * n_cells_col * n_orient)
    number_of_blocks_row = new_height / (params['pixels_per_cell'][0] * params['cells_per_block'][0])
    number_of_blocks_col = new_width / (params['pixels_per_cell'][1] * params['cells_per_block'][1])
    feature_vector_length = int(number_of_blocks_row * number_of_blocks_col * params['cells_per_block'][0] * \
                                params['cells_per_block'][1] * params['number_of_orientations'])

    X = np.zeros((number_of_images, feature_vector_length))

    for i in range(number_of_images):
        image = np.reshape(images[i, :], (28, 28))
        X[i, :] = extract_image_features(image, params)

    return X


def predict(model, params, image):

    '''

    :param model:
    :param params:
    :param image: 28x28 gray-scale image (numpy ndarray)
    :return:
    '''

    features = extract_image_features(image, params)
    prediction = model.predict(X=np.reshape(features, (1,-1)))
    return prediction
