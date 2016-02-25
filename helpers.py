import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import color


def load_demo_image():
    im = color.rgb2lab(Image.open('../object_recognition/img/Patern_test.jpg')) / 100.0
    return im[..., 0]


def zca_whitening(inputs):
    sigma = np.dot(inputs, inputs.T) / inputs.shape[1]  # Correlation matrix
    U, S, V = np.linalg.svd(sigma)  # Singular Value Decomposition
    epsilon = 0.1  # Whitening constant, it prevents division by zero
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0 / np.sqrt(np.diag(S) + epsilon))), U.T)  # ZCA Whitening matrix
    return np.dot(ZCAMatrix, inputs), ZCAMatrix  # Data whitening


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(X[0].shape[:2], tile_shape, tile_spacing)
        ]
    if len(X[0].shape)>2:
        out_shape.append(X[0].shape[2])

    if True:
        # if we are dealing with only one channel
        H, W = X[0].shape[:2]
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_img = X[tile_row * tile_shape[1] + tile_col]


                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                    tile_row * (H + Hs): tile_row * (H + Hs) + H,
                    tile_col * (W + Ws): tile_col * (W + Ws) + W,:
                    ] = this_img*c
        return out_array


def show_image(img, title=None, bw=False, per_row=2):
    cmap = 'viridis' if not bw else 'gray'
    if isinstance(img, list):
        total = len(img)
        plt.title(title)
        per_row = per_row
        cols = np.ceil(total / per_row)
        for id, k in enumerate(img):
            plt.subplot(cols, per_row, 1 + id)
            plt.grid(False)
            if isinstance(k, tuple):
                plt.title(k[1])
                plt.imshow(k[0], cmap=cmap, interpolation='nearest')
            else:
                plt.imshow(k, cmap=cmap, interpolation='nearest')
    else:
        plt.title(title)
        plt.grid(False)
        plt.imshow(img, cmap=cmap, interpolation='nearest')

    plt.tight_layout()
    plt.show()

def oned_to_flat(o):
    if o.shape[-1] == 1:
        o = o.reshape((o.shape[0], o.shape[1]))
    return (o/o.max()).astype('float64')

def show_representations(model, X_test, number=5, dim=28, do_reshape=True):
    representations = model.predict(X_test[:number ** 2, ...])

    def flat_to_shaped(x):
        return x.reshape((x.shape[0], dim, dim,1)) if do_reshape else x

    _r = tile_raster_images(
            X=flat_to_shaped(representations),
            img_shape=(dim, dim), tile_shape=(number, number),
            tile_spacing=(1, 1), output_pixel_vals=False)

    _o = tile_raster_images(
            X=flat_to_shaped(X_test),
            img_shape=(dim, dim), tile_shape=(number, number),
            tile_spacing=(1, 1), output_pixel_vals=False)
    print(_r.min())
    print(_r.max())
    show_image([(oned_to_flat(_o), 'Source'), (oned_to_flat(_r), 'Representations')])


def keras2rgb(t):
    return np.swapaxes(np.swapaxes(t, 1, 2), 2, 3)


if __name__ == '__main__':
    i = load_demo_image()
    print(i.shape)
    show_image(i)
