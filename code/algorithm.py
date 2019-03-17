# CSC320 Winter 2019
# Assignment 3
# (c) Olga (Ge Ya) Xu, Kyros Kutulakos
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY PROHIBITED. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

#
# DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
#

# import basic packages
import numpy as np

# basic numpy configuration

# set random seed
np.random.seed(seed=131)
# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore')


# This function implements the basic loop of the PatchMatch
# algorithm, as explained in Section 3.2 of the paper.
# The function takes an NNF f as input, performs propagation and random search,
# and returns an updated NNF.
#
# The function takes several input arguments:
#     - source_patches:      The matrix holding the patches of the source image,
#                            as computed by the make_patch_matrix() function. For an
#                            NxM source image and patches of width P, the matrix has
#                            dimensions NxMxCx(P^2) where C is the number of color channels
#                            and P^2 is the total number of pixels in the patch. The
#                            make_patch_matrix() is defined below and is called by the
#                            initialize_algorithm() method of the PatchMatch class. For
#                            your purposes, you may assume that source_patches[i,j,c,:]
#                            gives you the list of intensities for color channel c of
#                            all pixels in the patch centered at pixel [i,j]. Note that patches
#                            that go beyond the image border will contain NaN values for
#                            all patch pixels that fall outside the source image.
#     - target_patches:      The matrix holding the patches of the target image.
#     - f:                   The current nearest-neighbour field
#     - alpha, w:            Algorithm parameters, as explained in Section 3 and Eq.(1)
#     - propagation_enabled: If true, propagation should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step
#     - random_enabled:      If true, random search should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step.
#     - odd_iteration:       True if and only if this is an odd-numbered iteration.
#                            As explained in Section 3.2 of the paper, the algorithm
#                            behaves differently in odd and even iterations and this
#                            parameter controls this behavior.
#     - best_D:              And NxM matrix whose element [i,j] is the similarity score between
#                            patch [i,j] in the source and its best-matching patch in the
#                            target. Use this matrix to check if you have found a better
#                            match to [i,j] in the current PatchMatch iteration
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            you can pass them to/from your function using this argument

# Return arguments:
#     - new_f:               The updated NNF
#     - best_D:              The updated similarity scores for the best-matching patches in the
#                            target
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            return them in this argument and they will be stored in the
#                            PatchMatch data structure


def propagation_and_random_search(source_patches, target_patches,
                                  f, alpha, w,
                                  propagation_enabled, random_enabled,
                                  odd_iteration, best_D=None,
                                  global_vars=None
                                ):
    new_f = f.copy()

    #import time

    #now = time.clock()

    if best_D is None:
        best_D = initialize_D(f, source_patches, target_patches)

    #after_D = time.clock()
    # print 'D init took: ', after_D - now, 'seconds'

    for i in range(source_patches.shape[0]):
        for j in range(source_patches.shape[1]):
            if propagation_enabled:
                #now = time.clock()

                dist, patch = propagation(odd_iteration, best_D, new_f, source_patches[i][j], (i, j), target_patches)
                if dist != 0:
                    best_D[i][j] = dist
                    new_f[i][j] = patch

                #after_D = time.clock()
                # print 'propagation took: ', after_D - now, 'seconds'

            if random_enabled:
                current = best_D[i][j]
                #now = time.clock()
                best_patch = rand_search(w, alpha, (i, j), new_f, source_patches[i][j], target_patches, current)
                new_f[i][j] = (best_patch[0] - i, best_patch[1] - j)
                #after = time.clock()
                # print 'rand search took: ', after - now, 'seconds'

    return new_f, best_D, global_vars


# This function uses a computed NNF to reconstruct the source image
# using pixels from the target image. The function takes two input
# arguments
#     - target: the target image that was used as input to PatchMatch
#     - f:      the nearest-neighbor field the algorithm computed
# and should return a reconstruction of the source image:
#     - rec_source: an openCV image that has the same shape as the source image
#
# To reconstruct the source, the function copies to pixel (x,y) of the source
# the color of pixel (x,y)+f(x,y) of the target.
#
# The goal of this routine is to demonstrate the quality of the computed NNF f.
# Specifically, if patch (x,y)+f(x,y) in the target image is indeed very similar
# to patch (x,y) in the source, then copying the color of target pixel (x,y)+f(x,y)
# to the source pixel (x,y) should not change the source image appreciably.
# If the NNF is not very high quality, however, the reconstruction of source image
# will not be very good.
#
# You should use matrix/vector operations to avoid looping over pixels,
# as this would be very inefficient

def reconstruct_source_from_target(target, f):
    im_shape = (f.shape[0], f.shape[1])

    g = make_coordinates_matrix(im_shape)
    mapping = f + g

    x = np.clip(mapping[:, :, 0], 0, target.shape[0] - 1)
    y = np.clip(mapping[:, :, 1], 0, target.shape[1] - 1)

    rec_source = target[x, y]

    return rec_source


# This function takes an NxM image with C color channels and a patch size P
# and returns a matrix of size NxMxCxP^2 that contains, for each pixel [i,j] in
# in the image, the pixels in the patch centered at [i,j].
#
# You should study this function very carefully to understand precisely
# how pixel data are organized, and how patches that extend beyond
# the image border are handled.


def make_patch_matrix(im, patch_size):
    phalf = patch_size // 2
    # create an image that is padded with patch_size/2 pixels on all sides
    # whose values are NaN outside the original image
    padded_shape = im.shape[0] + patch_size - 1, im.shape[1] + patch_size - 1, im.shape[2]
    padded_im = np.zeros(padded_shape) * np.NaN
    padded_im[phalf:(im.shape[0] + phalf), phalf:(im.shape[1] + phalf), :] = im

    # Now create the matrix that will hold the vectorized patch of each pixel. If the
    # original image had NxM pixels, this matrix will have NxMx(patch_size*patch_size)
    # pixels
    patch_matrix_shape = im.shape[0], im.shape[1], im.shape[2], patch_size ** 2
    patch_matrix = np.zeros(patch_matrix_shape) * np.NaN
    for i in range(patch_size):
        for j in range(patch_size):
            patch_matrix[:, :, :, i * patch_size + j] = padded_im[i:(i + im.shape[0]), j:(j + im.shape[1]), :]

    return patch_matrix


# Generate a matrix g of size (im_shape[0] x im_shape[1] x 2)
# such that g(x,y) = [x,y]
#
# Step is an optional argument used to create a matrix that is step times
# smaller than the full image in each dimension
#
# Pay attention to this function as it shows how to perform these types
# of operations in a vectorized manner, without resorting to loops


def make_coordinates_matrix(im_shape, step=1):
    """
    Return a matrix of size (im_shape[0] x im_shape[1] x 2) such that g(x,y)=[x,y]
    """
    range_x = np.arange(0, im_shape[1], step)
    range_y = np.arange(0, im_shape[0], step)
    axis_x = np.repeat(range_x[np.newaxis, ...], len(range_y), axis=0)
    axis_y = np.repeat(range_y[..., np.newaxis], len(range_x), axis=1)

    return np.dstack((axis_y, axis_x))


# compute D between patches
def computeD(patch1, patch2):
    #import time
    #now = time.clock()
    diff = (patch1 - patch2) ** 2
    diff[np.isnan(diff)] = 255 ** 2
    #after = time.clock()
    # print 'compute a simple D took ', after - now, ' seconds'
    return np.sum(np.sum(diff, axis=0))


# compute initial best_D
def initialize_D(f, source_patches, target_patches):
    """
    initialize D using initial f

    """
    im_shape = (source_patches.shape[0], source_patches.shape[1])

    sources = make_coordinates_matrix(im_shape)
    source_x = np.clip(sources[:, :, 0], 0, source_patches.shape[0] - 1)
    source_y = np.clip(sources[:, :, 1], 0, source_patches.shape[1] - 1)

    targets = f + sources
    x = np.clip(targets[:, :, 0], 0, target_patches.shape[0] - 1)
    y = np.clip(targets[:, :, 1], 0, target_patches.shape[1] - 1)

    square_difference = (source_patches[source_x, source_y] - target_patches[x, y]) ** 2

    square_difference[np.isnan(square_difference)] = 255 ** 2

    return np.sum(np.sum(square_difference, axis=2), axis=2)


# propagation procedure for one patch in source
def propagation(odd_iteration, best_D, f, source_patch, coordinate, target_patches):
    """
    perform propagation for one patch in source
    return the best distance score and the corresponding NNF value

    """
    candidate_offsets = []
    if odd_iteration:
        candidate_offsets.append([-1, 0])
        candidate_offsets.append([0, -1])
    else:
        candidate_offsets.append([1, 0])
        candidate_offsets.append([0, 1])

    current_best = best_D[coordinate]

    source_coords = np.array(coordinate) + np.array(candidate_offsets)
    source_x = np.clip(source_coords[:, 0], 0, best_D.shape[0] - 1)
    source_y = np.clip(source_coords[:, 1], 0, best_D.shape[1] - 1)

    offsets = f[source_x, source_y]
    target_coords = offsets + coordinate
    x = np.clip(target_coords[:, 0], 0, target_patches.shape[0] - 1)
    y = np.clip(target_coords[:, 1], 0, target_patches.shape[1] - 1)

    target1 = target_patches[x[0]][y[0]]
    target2 = target_patches[x[1]][y[1]]

    dist_1 = computeD(source_patch, target1)
    dist_2 = computeD(source_patch, target2)

    a = np.array([current_best, dist_1, dist_2])
    idx = np.argmin(a)

    if idx != 0:
        return a[idx], offsets[idx - 1]
    else:
        return 0, 0


# random search procedure for one patch in source
def rand_search(window, alpha, coord, f, source_patch, target_patches, current_best_D):
    """
    perform the random search step for patch centered at coord
    and return the coordinate of best-matching patch in target

    """
    old = f[coord[0], coord[1]]

    offsets = []
    size = window * alpha
    while size > 1:
        rand = np.random.uniform(-1, 1, (2, ))
        new = old + size * rand
        offsets.append(new)
        size *= alpha

    patches = np.array(coord) + offsets
    patches = patches.astype(int)
    x = np.clip(patches[:, 0], 0, target_patches.shape[0] - 1)
    y = np.clip(patches[:, 1], 0, target_patches.shape[1] - 1)

    diff = []
    for i in range(len(offsets)):
        diff.append(computeD(source_patch, target_patches[x[i], y[i]]))

    idx = np.argmin(diff)

    return x[idx], y[idx]
