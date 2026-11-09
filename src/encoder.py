import collections
import math
import pickle

import cv2
import newdata.datacreate
import numpy

import numpy as np

import matplotlib.pyplot as plt


def getImage(filename):
    return cv2.imread(filename, cv2.IMREAD_COLOR)


def getImageRaw(img: cv2.UMat):
    return img.get()


def pushToListDict(dictionary: dict, key, value):
    if key in dictionary:
        dictionary[key].append(value)
    else:
        dictionary[key] = [(value, 1)]


def pushToDictBGR(dictionary: dict, key, value):
    if key in dictionary:
        dictionary[key][0] += value[0]
        dictionary[key][1] += value[1]
        dictionary[key][2] += value[2]
        dictionary[key][3] += 1
    else:
        dictionary[key] = [int(value[0]), int(value[1]), int(value[2]), 1]


def incrementDict(dictionary: dict, key, value):
    if key in dictionary:
        dictionary[key] += 1
    else:
        dictionary[key] = 1


def tupleToKey(to_be_key: collections.abc.Iterable):
    key = 14881026732586785282010203713473
    for i in to_be_key:
        key = (key ^ int(i) * 35612964170653523793897223775381) % 65588536464688571772573639371471
    return key % 2 ** 32


def getColorScheme(img: cv2.Mat, COLOR_BITWIDTH=8):
    color_space_st = []
    for i in range(COLOR_BITWIDTH + 1):
        color_space_st.append({})
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(COLOR_BITWIDTH + 1):
                pushToDictBGR(color_space_st[k], tupleToKey((
                    img[i, j, 0] // (2 ** k),
                    img[i, j, 1] // (2 ** k),
                    img[i, j, 2] // (2 ** k),
                )), img[i, j])
    for k in range(COLOR_BITWIDTH + 1):
        for x in color_space_st[k]:
            color_space_st[k][x][0] /= color_space_st[k][x][3]
            color_space_st[k][x][1] /= color_space_st[k][x][3]
            color_space_st[k][x][2] /= color_space_st[k][x][3]
            color_space_st[k][x] = tuple(color_space_st[k][x])
    return color_space_st


def showColorSpace(color_space: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xs = np.zeros((color_space.shape[0], ))
    ys = np.zeros((color_space.shape[0], ))
    zs = np.zeros((color_space.shape[0], ))
    cs = np.zeros((color_space.shape[0], 3, ))
    szs = np.zeros((color_space.shape[0], ))
    for i, col in enumerate(color_space):
        xs[i] = col[0]
        ys[i] = col[1]
        zs[i] = col[2]
        szs[i] = 10 * col[3] ** 0.25
        cs[i, 0] = col[0] / 256
        cs[i, 1] = col[1] / 256
        cs[i, 2] = col[2] / 256
    ax.scatter(xs, ys, zs, c=cs, s=szs, marker='o')

    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')

    plt.show()


def saveColorSpace(color_space: list[dict], name):
    with open(f'imageColorSpaces/{name}.data', 'wb') as f:
        pickle.dump(color_space, f)


def loadColorSpace(name):
    with open(name, 'rb') as f:
        color_space = pickle.load(f)
    return color_space


def test(filename, n=0):
    img = getImage(filename)

    sct = getColorScheme(img)

    saveColorSpace(sct, filename.split('/')[-1])

    sc = sct[n]

    sc_arr = np.zeros((len(sc), 4))
    for i, key in enumerate(sc):
        rgb = np.zeros((4,))
        rgb[0] = sc[key][2]
        rgb[1] = sc[key][1]
        rgb[2] = sc[key][0]
        rgb[3] = sc[key][3]
        sc_arr[i] = rgb

    showColorSpace(sc_arr)


def testFrom(filename, n=0):
    sct = loadColorSpace(filename)

    print('Color Space Loaded!')

    sc = sct[n]

    sc_arr = np.zeros((len(sc), 4))
    for i, key in enumerate(sc):
        rgb = np.zeros((4,))
        rgb[0] = sc[key][0]
        rgb[1] = sc[key][1]
        rgb[2] = sc[key][2]
        rgb[3] = sc[key][3]
        sc_arr[i] = rgb

    print(len(sc_arr))
    showColorSpace(sc_arr)


def getImageBytes(img: cv2.Mat):
    return img.flatten()


def getSubChunks(color_space: list[dict], layer, chunk, COLOR_BITWIDTH):
    chunks = []
    if layer == 0:
        return chunks
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x = chunk[0] + 2 ** (COLOR_BITWIDTH - layer - 1) * i
                y = chunk[1] + 2 ** (COLOR_BITWIDTH - layer - 1) * j
                z = chunk[2] + 2 ** (COLOR_BITWIDTH - layer - 1) * k
                key = tupleToKey((x // 2 ** (layer - 1),
                                  y // 2 ** (layer - 1),
                                  z // 2 ** (layer - 1)))
                if key in color_space[layer - 1]:
                    chunks.append((x, y, z, color_space[layer - 1][key][3]))
    return chunks


def getChunkPoint(color_space: list[dict], layer, chunk):
    return color_space[layer][tupleToKey((chunk[0] // 2 ** layer,
                                          chunk[1] // 2 ** layer,
                                          chunk[2] // 2 ** layer))]


def getChunks(color_space: list[dict], layer, COLOR_BITWIDTH):
    chunks = []
    for k in color_space[layer]:
        point = color_space[layer][k]
        current_chunk = ((point[0] // 2 ** (COLOR_BITWIDTH - layer)) * 2 ** (COLOR_BITWIDTH - layer),
                         (point[1] // 2 ** (COLOR_BITWIDTH - layer)) * 2 ** (COLOR_BITWIDTH - layer),
                         (point[2] // 2 ** (COLOR_BITWIDTH - layer)) * 2 ** (COLOR_BITWIDTH - layer),
                         point[3])
        chunks.append(current_chunk)
    return chunks


def vectorLength(vector):
    return sum(map(lambda x: x * x, vector))


def getDeviation(chunk_a: tuple, chunk_b: tuple):
    alpha = 1 / (chunk_a[3] + chunk_b[3])
    center_dlt = ((chunk_a[0] * chunk_b[3] - chunk_b[0] * chunk_a[3]) * alpha,
                  (chunk_a[1] * chunk_b[3] - chunk_b[1] * chunk_a[3]) * alpha,
                  (chunk_a[2] * chunk_b[3] - chunk_b[2] * chunk_a[3]) * alpha)
    return vectorLength(center_dlt)


def mergeChunks(chunk_a: tuple, chunk_b: tuple):
    alpha = 1 / (chunk_a[3] + chunk_b[3])
    center_chunk = ((chunk_a[0] * chunk_a[3] + chunk_b[0] * chunk_b[3]) * alpha,
                    (chunk_a[1] * chunk_a[3] + chunk_b[1] * chunk_b[3]) * alpha,
                    (chunk_a[2] * chunk_a[3] + chunk_b[2] * chunk_b[3]) * alpha,
                    # 1 / alpha)
                    1)
    return center_chunk


def reduceColorSpaceRecursive(color_space: list[dict],
                              layer: int,
                              chunks: list,
                              centroids_number: int,
                              deviations: list,
                              COLOR_BITWIDTH=8):
    centroids = []
    if len(chunks) < centroids_number and layer != COLOR_BITWIDTH:
        print("Chunks:", len(chunks), layer)
        centroids_weights = []
        centroids_weights_sum = 0
        centroids_weights_function = lambda x: 1
        for chunk_i in chunks:
            centroids_weights.append([centroids_weights_function(chunk_i[3]), 0])
            centroids_weights_sum += centroids_weights_function(chunk_i[3])
        for i in range(len(centroids_weights)):
            centroids_weights[i][0] /= centroids_weights_sum
            centroids_weights[i][1] = i
        centroids_weights.sort(key=lambda x: x[0])
        remaining_centroids_number = centroids_number
        reserved_centroids_number = 0
        for i in range(len(centroids_weights)):
            centroids_number_new = max(1, int(centroids_weights[i][0] *
                                              remaining_centroids_number))
            remaining_centroids_number -= max(1,
                                              int(centroids_weights[i][0] *
                                                  remaining_centroids_number)
                                              ) - centroids_weights[i][0] * centroids_number
            if i == len(centroids_weights) - 1:
                centroids_number_new = centroids_number - reserved_centroids_number
            reserved_centroids_number += centroids_number_new
            centroids.extend(reduceColorSpaceRecursive(color_space,
                                                       layer - 1,
                                                       getSubChunks(color_space,
                                                                    layer,
                                                                    chunks[centroids_weights[i][1]],
                                                                    COLOR_BITWIDTH),
                                                       centroids_number_new,
                                                       deviations,
                                                       COLOR_BITWIDTH))
    else:
        rand_seed = len(chunks) * centroids_number
        for i in range(len(chunks) - centroids_number):
            rand_seed = ((i + 1) * rand_seed) & (2 ** 16 - 1)
            idx = rand_seed % len(chunks)
            min_deviation_j = (idx + 1) % len(chunks)
            min_deviation = getDeviation(chunks[idx], chunks[min_deviation_j])
            new_chunks = []
            for j in range(len(chunks)):
                deviation_j = getDeviation(chunks[j], chunks[idx])
                if deviation_j < min_deviation and j != idx:
                    min_deviation_j = j
                    min_deviation = deviation_j
            for j in range(len(chunks)):
                if j != min_deviation_j and j != idx:
                    new_chunks.append(chunks[j])
            new_chunks.append(mergeChunks(chunks[idx], chunks[min_deviation_j]))
            chunks = new_chunks
            deviations.append(min_deviation)
        for chunk_i in chunks:
            centroids.append(chunk_i)
    return centroids


def reduceColorSpace(color_space: list[dict], new_size, deviations=None, COLOR_BITWIDTH=8):
    if deviations is None:
        deviations = []
    layer = 0
    chunks = getChunks(color_space, layer, COLOR_BITWIDTH)
    while len(chunks) >= 0.8 * new_size:
        layer += 1
        if layer == COLOR_BITWIDTH:
            return reduceColorSpaceRecursive(color_space, layer - 1, chunks, new_size, deviations, COLOR_BITWIDTH)
        chunks = getChunks(color_space, layer, COLOR_BITWIDTH)
    return reduceColorSpaceRecursive(color_space, layer, chunks, new_size, deviations, COLOR_BITWIDTH)


def createRelation(color_space: dict, reduced_color_space: list[tuple[4]]):
    relation = {}
    true_deviations = []
    for key in color_space:
        min_div_i = 0
        min_div = getDeviation((color_space[key][0],
                                color_space[key][1],
                                color_space[key][2],
                                1),
                               (reduced_color_space[0][0],
                                reduced_color_space[0][1],
                                reduced_color_space[0][2],
                                1))
        for i in range(1, len(reduced_color_space)):
            div = getDeviation((color_space[key][0],
                                color_space[key][1],
                                color_space[key][2],
                                1),
                               (reduced_color_space[i][0],
                                reduced_color_space[i][1],
                                reduced_color_space[i][2],
                                1))
            if div < min_div:
                min_div_i = i
                min_div = div
        true_deviations.append(min_div)
        relation[key] = reduced_color_space[min_div_i]
    return relation


def testReduceFrom(filename, new_size, og_filename='samples/test 6.jpg'):
    sct = loadColorSpace(filename)

    print('Color Space Loaded!')

    print(len(sct[0]))

    deviations = []

    sc = reduceColorSpace(sct, new_size, deviations=deviations)

    print(len(deviations))

    deviations.sort()

    if len(deviations) > 5:
        print(deviations[0],
              deviations[len(deviations) // 4],
              deviations[len(deviations) // 2],
              deviations[2 * len(deviations) // 4],
              deviations[-1])

    n = 4

    sct_arr = np.zeros((len(sct[n]), 4))
    for i, key in enumerate(sct[n]):
        rgb = np.zeros((4,))
        rgb[0] = sct[n][key][0]
        rgb[1] = sct[n][key][1]
        rgb[2] = sct[n][key][2]
        rgb[3] = sct[n][key][3]
        sct_arr[i] = rgb

    showColorSpace(sct_arr)

    print(len(sc))

    sc_arr = np.zeros((len(sc), 4))
    for i, key in enumerate(sc):
        rgb = np.zeros((4,))
        rgb[0] = key[0]
        rgb[1] = key[1]
        rgb[2] = key[2]
        rgb[3] = key[3]
        sc_arr[i] = rgb

    print(len(sc_arr))
    showColorSpace(sc_arr)

    rel = createRelation(sct[0], sc)

    print("Relation Created!")

    image = getImage(og_filename)

    new_image: cv2.UMat = image.copy()
    # print(rel)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            px = rel[tupleToKey((image[i, j, 0],
                                 image[i, j, 1],
                                 image[i, j, 2]))]
            new_image[i, j, 0] = px[0]
            new_image[i, j, 1] = px[1]
            new_image[i, j, 2] = px[2]

    # new_image = cv2.resize(new_image, (512, 512))
    cv2.imshow("New Image", new_image)
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)


def encodeImage(filename, new_colorspace_size):
    sct = loadColorSpace(filename)

    print('Color Space Loaded!')

    print(len(sct[0]))

    deviations = []

    sc = reduceColorSpace(sct, new_colorspace_size, deviations=deviations)

    print(len(deviations))

    deviations.sort()

    if len(deviations) > 5:
        print(deviations[0],
              deviations[len(deviations) // 4],
              deviations[len(deviations) // 2],
              deviations[2 * len(deviations) // 4],
              deviations[-1])
    rel = createRelation(sct[0], sc)

    color_dict = {}

    for i, col in enumerate(sc):
        color_dict[tupleToKey((col[0], col[1], col[2]))] = i

    print("Relation Created!")

    image = getImage('samples/' + filename.split('/')[-1][:-5])

    new_image = image.copy()
    new_image_enc = np.zeros(shape=(image.shape[0], image.shape[1]))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            px = rel[tupleToKey((image[i, j, 0],
                                 image[i, j, 1],
                                 image[i, j, 2]))]
            new_image[i, j, 0] = px[0]
            new_image[i, j, 1] = px[1]
            new_image[i, j, 2] = px[2]
            new_image_enc[i, j] = color_dict[tupleToKey((px[0],
                                                         px[1],
                                                         px[2]))]
    # cv2.imshow("Reduced Image", new_image)
    # cv2.waitKey(0)

    return sc, new_image_enc, new_image


def encodeToString(filename, new_size):
    sc, new_image = encodeImage(filename, new_size)
    return {"w": new_image.shape[0],
            "h": new_image.shape[1],
            "colors": sc,
            "bytestring": ''.join(map(lambda x: chr(int(x) + 33), new_image.flatten()))}


def showEncodedImage(encoded_data: dict[str, any]):
    image = np.zeros(shape=(encoded_data["w"], encoded_data["h"], 3), dtype=np.uint8)
    colors = encoded_data["colors"]
    bytestring = encoded_data["bytestring"]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j, 0] = colors[ord(bytestring[i * image.shape[0] + j]) - 33][0]
            image[i, j, 1] = colors[ord(bytestring[i * image.shape[0] + j]) - 33][1]
            image[i, j, 2] = colors[ord(bytestring[i * image.shape[0] + j]) - 33][2]
    cv2.imshow("Encoded Image", image)
    cv2.waitKey(0)


# test('samples/test 3.jpg', 4)
# testFrom('imageColorSpaces/test 3.jpg.data', 0)

# testReduceFrom('imageColorSpaces/test 3.jpg.data', 32, 'samples/test 3.jpg')
#
# for i in range(356):
#     print(chr(i), i)

# data = encodeToString('imageColorSpaces/settings_icon.png.data', 16)
#
# showEncodedImage(data)

import math
from itertools import chain


def to_base_n(number, base, pad=8):
    """
    Converts a decimal integer to its string representation in a given base.

    Args:
        number (int): The decimal integer to convert.
        base (int): The desired base (must be between 2 and 36, inclusive).

    Returns:
        str: The string representation of the number in the specified base.
    """
    if not (2 <= base <= 36):
        raise ValueError("Base must be between 2 and 36.")
    if number == 0:
        return "0"

    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = ""
    is_negative = False

    if number < 0:
        is_negative = True
        number = abs(number)

    while number > 0:
        remainder = number % base
        result = digits[remainder] + result
        number //= base

    result = '0' * (pad - len(result)) + result

    if is_negative:
        result = "-" + result

    return result


def convertTX(ternary_value):
    hex_value = ''
    tmp_value = 32
    cnt = 0
    for i in ternary_value:
        if cnt == 4:
            hex_value += chr(tmp_value)
            tmp_value = 32
            cnt = 0
        tmp_value += int(i) * (3 ** cnt)
        cnt += 1
    return hex_value + chr(tmp_value)


# print(convertTX('10000110220011022110201001100').encode('latin-1'))


def normChar(c: str):
    return ord(c) - 32 if ord(c) < 128 else ord(c) - 64


def nextCodes(codes: int, cnt: int, enc_tree: str):
    prev_codes = codes
    codes //= 3
    if cnt % 4 == 0:
        codes = normChar(enc_tree[cnt // 4])
    return codes, cnt + 1, prev_codes


def decTree(enc_tree: str, colors: str, val_len):
    val = 1 << normChar(enc_tree[0])
    tree = {val: [1] + list(map(lambda x: normChar(x) * 256 // 192, colors[:val_len]))}
    cnt = 4
    next_codes, cnt, codes = nextCodes(0, cnt, enc_tree)
    t = 0
    for i in range(1, len(colors) // val_len - 1):
        while (next_codes % 3 == 2 and codes % 3 == 2) or t == 0:
            next_codes, cnt, codes = nextCodes(next_codes, cnt, enc_tree)
            val += 1 + (val + 1 if (codes % 3 == 1) else 0)
            val = (val >> 1) if (codes % 3 == 2) else val
            t = 1
        t = 0

        tree[val] = [i + 1] + list(map(lambda x: normChar(x) * 256 // 192, colors[i * val_len:(i + 1) * val_len]))
    return tree


ID = 0


class HENode:
    def __init__(self, value, freq, code=''):
        self.left = None
        self.right = None
        self.value = value
        self.depth = 1
        self.freq = freq
        self.code = code
        self.max_depth = -1
        global ID
        self.id = ID
        ID += 1

    def __add__(self, other):
        root = HENode(None, self.freq + other.freq)
        root.depth = max(self.depth, other.depth) + 1
        if self.depth < other.depth:
            root.left = self
            root.right = other
        elif self.depth > other.depth:
            root.left = other
            root.right = self
        else:
            if self.freq < other.freq:
                root.left = other
                root.right = self
            else:
                root.right = self
                root.left = other
        return root

    def __str__(self):
        return f'({self.freq}, {self.depth}, {self.code}, {str(self.value)}, {str(self.id)}, {str(self.max_depth)})'

    def __repr__(self):
        return f'({self.freq}, {self.depth}, {self.code}, {str(self.value)}, {str(self.id)}, {str(self.max_depth)})'


def swap(iterable: list, i: int, j: int):
    iterable[i], iterable[j] = iterable[j], iterable[i]
    return iterable


def pushToOrder(iterable: list, key: callable = lambda x: x):
    idx = len(iterable) - 1
    while idx > 0:
        if key(iterable[idx]) < key(iterable[idx - 1]):
            swap(iterable, idx, idx - 1)
        else:
            break
        idx -= 1
    return iterable


def traverseHETree(root: HENode, storage: dict[int, list[HENode]], path: str = '*'):
    root.code = path
    if root.left is not None:
        traverseHETree(root.left, storage, path + '0')
    if root.right is not None:
        traverseHETree(root.right, storage, path + '1')
    storage[root.depth].append(root)


def buildTreeFromBranchLength(branch_length: int,
                              root: HENode,
                              nodes: list[HENode]):

    if root is None or root.freq > branch_length or root.max_depth == 0:
        return False

    if root.left is None and root.right is None:
        if root.code == 'r':
            node = root
            for i in range(root.freq, branch_length):
                node.left = HENode(None, node.freq + 1, 'l')
                node.right = HENode(None, node.freq + 1, 'r')
                nodes.append(node.left)
                nodes.append(node.right)
                node = node.left
            node.max_depth = 0
            node.code = 'l'
            root.depth = branch_length - root.freq + 1
            root.code = 'R'
            return True
        return False

    if root.freq == branch_length:
        root.max_depth = 0

    if root.left.max_depth == 0 and root.right.max_depth == 0:
        root.max_depth = 0

    if buildTreeFromBranchLength(branch_length, root.left, nodes):
        root.depth = max(root.depth, root.left.depth + 1, root.right.depth + 1)
        return True
    if buildTreeFromBranchLength(branch_length, root.right, nodes):
        root.depth = max(root.depth, root.left.depth + 1, root.right.depth + 1)
        return True
    return False


def buildTree(data: list[list]):
    unit_length = len(data[0])
    frequencies = {}

    for i in data:
        if str(i) in frequencies:
            frequencies[str(i)] += 1
        else:
            frequencies[str(i)] = 1

    retrieve_rel = {str(i): i for i in data}

    keys_quantized = sorted(frequencies.keys(), key=lambda x: -frequencies[x])

    nodes = [HENode(i, frequencies[i]) for i in keys_quantized]
    tree = nodes.copy()

    while len(tree) > 1:
        min_node_0, min_node_1 = tree[-1], tree[-2]
        tree.pop()
        tree.pop()
        tree.append(min_node_0 + min_node_1)
        pushToOrder(tree, key=lambda x: -x.freq)

    root = tree[0]
    tmp_dict = {}
    for i in range(root.depth):
        tmp_dict[i + 1] = []
    traverseHETree(root, tmp_dict, '1')

    branch_lengths = list(map(lambda x: len(x.code), tmp_dict[1]))

    branch_lengths.sort()

    new_root = HENode(None, 1, code='r')

    new_root.depth = 0

    new_nodes = [new_root]

    for branch_length in branch_lengths:
        buildTreeFromBranchLength(branch_length, new_root, new_nodes)

    for node in new_nodes:
        node.depth = max(node.depth, 1)

    new_tmp_dict = {}
    for i in range(new_root.depth):
        new_tmp_dict[i + 1] = []
    traverseHETree(new_root, new_tmp_dict, '1')

    tmp_dict[1].sort(key=lambda x: len(x.code))

    for i in range(len(tmp_dict[1])):
        new_tmp_dict[1][i].value = retrieve_rel[tmp_dict[1][i].value]

    return new_root, new_tmp_dict[1], new_tmp_dict


def encodeSequence(__sequence: list[list]):
    root, nodes, tree = buildTree(__sequence)

    codes = {str(node.value): node.code for node in tree[1]}

    bit_len = sum([len(codes[str(token)][1:]) for token in __sequence])

    enc_seq = ''
    current_byte = 1
    current_pow = 1
    current_bit_length = 8 - (bit_len % 8)
    if current_bit_length == 8:
        enc_seq += chr(current_byte)
        current_byte = 0
        current_pow = 1
        current_bit_length = 0
    for token in __sequence:
        for enc_bit in codes[str(token)][1:]:
            current_byte <<= 1
            current_byte += current_pow * int(enc_bit)
            current_bit_length += 1
            if current_bit_length == 8:
                enc_seq += chr(current_byte)
                current_byte = 0
                current_pow = 1
                current_bit_length = 0
    if current_bit_length > 0:
        enc_seq += chr(current_byte << (8 - current_bit_length))

    tree_bits = '1' * (len(tree[1][0].code) - 1) + '0'
    for i in range(len(tree[1]) - 1):
        tree_bits += '1' * (len(tree[1][i + 1].code) - len(tree[1][i].code)) + '0'
    tree_code_bytes = int(tree_bits, base=2).to_bytes((len(tree_bits) + 7) // 8, 'big')
    tree_seq = tree_code_bytes.decode('latin-1')
    tree_data = [node_i.value for node_i in nodes]
    return enc_seq, tree_seq, tree_data


def decodeSequence(encoded_seq: str, encoded_tree_seq: str, tree_data: list):
    tree = {}
    ets_key = 1
    cnt = -1
    idx = 0
    tree_bitseq = ''
    for byte_i in encoded_tree_seq.encode('latin-1'):
        tree_bitseq += (bin(byte_i)[2:]).zfill(8)
    while tree_bitseq[idx] == '0':
        idx += 1
    while tree_bitseq[idx] == '1':
        idx += 1
        ets_key *= 2

    while idx < len(tree_bitseq):
        if tree_bitseq[idx] == '0':
            cnt += 1
            tree[bin(ets_key)[2:]] = tree_data[cnt]
            ets_key += 1
        else:
            ets_key *= 2
        idx += 1

    encoded_bitseq = ''

    for byte_i in encoded_seq.encode('latin-1'):
        encoded_bitseq += (bin(byte_i)[2:]).zfill(8)

    es_key = '1'
    seq = []

    def get6bits(s: str):
        bits = ''
        for i in s:
            bits += '0' * (8 - len(bin(ord(i) - 48))) + bin(ord(i) - 48)[2:]
        return bits

    # print(get6bits(convert8To6BIT(encoded_tree_seq)))
    # print(tree_bitseq)

    for bit_i in encoded_bitseq[8 - len(bin(encoded_seq.encode('latin-1')[0])[2:]):][1:]:
        es_key += bit_i
        if es_key in tree:
            seq.append(tree[es_key])
            es_key = '1'

    return seq


def convert8To6BIT(seq_8bit: str):

    seq_6bit = ''
    current_bit_length = 6 - ((len(seq_8bit) * 8 + 1) % 6)
    current_byte = 1
    current_bit_length += 1
    for token in seq_8bit:
        bin_token = bin(ord(token))[2::]
        bin_token = '0' * (8 - len(bin_token)) + bin_token
        for enc_bit in bin_token:
            if current_bit_length == 6:
                seq_6bit += chr(48 + current_byte)
                current_byte = 0
                current_bit_length = 0
            current_byte <<= 1
            current_byte += int(enc_bit)
            current_bit_length += 1
    if current_bit_length > 0:
        seq_6bit += chr(48 + current_byte << (6 - current_bit_length))
    b8_bits = ''
    for i in seq_8bit:
        b8_bits += '0' * (8 - len(bin(ord(i))[2:])) + bin(ord(i))[2:]
    # print(b8_bits)
    b6_bits = ''
    for i in seq_6bit:
        b6_bits += '0' * (6 - len(bin(ord(i) - 48)[2:])) + bin(ord(i) - 48)[2:]
    # print(b6_bits)
    return seq_6bit

#
# LuaEncodeTable = " !\"#$%&\'()*+,-./0123456789:;" \
#                  "<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ" \
#                  "[\\]^_`abcdefghijklmnopqstuvwxy" \
#                  "z{|}~\x7FАБВГДЕЖЗИЙКЛМНОПРСТУФХ" \
#                  "ЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуф" \
#                  "░▒▓│┤╡╢╖╕╣║╗╝╜╛┐└┴┬├─┼╞╟╚╔╩╦╠═╬" \
#                  "╧╨╤╥╙╘╒╓╫╪┘┌█▄▌▐▀рстуфхцчшщъыьэ" \
#                  "юяЁёЄєЇїЎў°∙·√№¤■"


def encodeColors(colors: list[list]):
    color_seq = ''
    for col in colors:
        color_seq += f'{chr(col[0])}{chr(col[1])}{chr(col[2])}'
    color_seq = convert8To6BIT(color_seq)
    return color_seq


def getColorBits(colors: list[list]):
    color_seq = ''
    for col in colors:
        color_seq += ('0' * (10 - len(bin(col[0]))) + bin(col[0])[2:]) + \
                     ('0' * (10 - len(bin(col[1]))) + bin(col[1])[2:]) + \
                     ('0' * (10 - len(bin(col[2]))) + bin(col[2])[2:])
    return color_seq

#
# for i in range(256):
#     print(i, chr(i))

# buildTree()
#
seq_to_encode = [[127, 0, 0],
                 [1, 0, 0],
                 [2, 0, 0],
                 [3, 0, 0],
                 [127, 0, 0],
                 [1, 0, 0],
                 [2, 0, 0],
                 [127, 0, 0],
                 [4, 0, 0],
                 [5, 0, 0],
                 [6, 0, 0],
                 [7, 0, 0],
                 [8, 0, 0],
                 [6, 0, 0],
                 [127, 0, 0],
                 [5, 0, 0]]

# print(len(seq_to_encode))
# x, y, z = encodeSequence(seq_to_encode)
# print(z, getColorBits(z))
# print(convert8To6BIT(x), convert8To6BIT(y), encodeColors(z), sep='\n')
# print(decodeSequence(x, y, z))


# print(int('01010101010100', base=2), (int('01010101010100', base=2).to_bytes(2, 'big')).decode('latin-1'))
# print(convert8To6BIT((int('01010101010100', base=2).to_bytes(2, 'big')).decode('latin-1')))


def colorSpaceReduceFrom(filename, new_size, og_filename='samples/settings_icon.png'):

    img = getImage(og_filename)

    sct = getColorScheme(img)

    saveColorSpace(sct, filename.split('/')[-1])

    sc = sct[0]

    sct = loadColorSpace(filename)

    print('Color Space Loaded!')

    print(len(sct[0]))

    deviations = []

    sc = reduceColorSpace(sct, new_size, deviations=deviations)

    print(len(deviations))

    deviations.sort()

    if len(deviations) > 5:
        print(deviations[0],
              deviations[len(deviations) // 4],
              deviations[len(deviations) // 2],
              deviations[2 * len(deviations) // 4],
              deviations[-1])

    n = 4

    sct_arr = np.zeros((len(sct[n]), 4))
    for i, key in enumerate(sct[n]):
        rgb = np.zeros((4,))
        rgb[0] = sct[n][key][2]
        rgb[1] = sct[n][key][1]
        rgb[2] = sct[n][key][0]
        rgb[3] = sct[n][key][3]
        sct_arr[i] = rgb

    showColorSpace(sct_arr)

    print(len(sc))

    sc_arr = np.zeros((len(sc), 4))
    for i, key in enumerate(sc):
        rgb = np.zeros((4,))
        rgb[0] = key[2]
        rgb[1] = key[1]
        rgb[2] = key[0]
        rgb[3] = key[3]
        sc_arr[i] = rgb

    print(len(sc_arr))
    showColorSpace(sc_arr)

    rel = createRelation(sct[0], sc)

    print("Relation Created!")

    image = getImage(og_filename)

    new_image: cv2.UMat = image.copy()
    # print(rel)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            px = rel[tupleToKey((image[i, j, 0],
                                 image[i, j, 1],
                                 image[i, j, 2]))]
            new_image[i, j, 0] = px[0]
            new_image[i, j, 1] = px[1]
            new_image[i, j, 2] = px[2]

    # new_image = cv2.resize(new_image, (512, 512))
    cv2.imshow("New Image", new_image)
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)

    return new_image


# cv2.imshow('', icon_image)
# cv2.waitKey(0)
# icon_image = colorSpaceReduceFrom('imageColorSpaces/settings_icon.png.data', 16, 'samples/settings_icon.png')

# seq = []
# for i in range(icon_image.shape[0]):
#     for j in range(icon_image.shape[1]):
#         seq.append(list(icon_image[i, j])[::-1])
# print(icon_image.shape)

# vseq = [
#         [1, 10, 1], [198, 127, 202],
#         [1, 10, 1], [198, 127, 202],
#         # [1, 10, 1], [198, 127, 202],
#         # [1, 10, 1], [198, 127, 202],
#         # [1, 10, 1], [198, 127, 202],
#         [155, 155, 155],
#         [1, 10, 1], [1, 1, 3],
#         [155, 155, 155],
#         [1, 10, 1], [1, 1, 1],
#         [155, 155, 155],
#         [1, 10, 1], [0, 1, 3],
#         [155, 0, 0],
#         [1, 10, 1], [5 + 16 * 5, 32, 88],
#         [89, 90, 33]
#         ]

vseq = [
    [1, 10, 1], [0x09, 32, 0],
    [1, 10, 1], [0x09, 33, 1],
    [1, 10, 1], [0x09, 34, 1],
    [1, 10, 1], [0x09, 35, 15],
    [1, 10, 1], [0x09, 36, 1],
    [1, 10, 1], [0x09, 37, 128],
    [1, 10, 1], [0x09, 38, 114],
    [1, 10, 1], [0x09, 39, 1],
#LABEL0:
    [1, 10, 1], [0xa9, 34, 33],
    [1, 10, 1], [0x19, 33, 32],
    [1, 10, 1], [0xa9, 32, 34],
    [1, 10, 1], [0x99, 35, 127],
    [1, 10, 1], [0x99, 39, 129],
    [1, 10, 1], [0xc9, 35, 1],
    [1, 10, 1], [0xe9, 37, 38],
    [1, 10, 1], [0x09, 34, 32],
    [1, 10, 1], [0x09, 35, 39],
    [1, 10, 1], [0xf9, 34, 35],
    [1, 10, 1], [0x25, 0, 33],
]


class SequenceBuilder:
    def __init__(self, w, h):
        self.__special_chr = [-1, -1, -1]
        self.__taken_chrs = {}
        self.__width = w
        self.__height = h
        self.__proportional = True
        self.__brush_width = 1
        self.__brush_height = 1
        self.__sequence = []
        self.__labels = {}
        self.__label_index_queue = []
        self.__macros = {}
        self.injectAsm('''
            mov r255 255
            mov r7 256
        ''')

    def addImage(self, offset_x, offset_y, img: [numpy.ndarray, cv2.Mat]):
        for i in range(img.shape[0]):
            self.offSet(offset_x, offset_y + i)
            for j in range(img.shape[1]):
                col = (img[i, j, 0] << 16) + (img[i, j, 1] << 8) + img[i, j, 2]
                self.__taken_chrs[col] = 1
                self.__sequence += [[img[i, j, 0], img[i, j, 1], img[i, j, 2]]]
        return self

    def offSet(self, offset_x, offset_y):
        self.__sequence += [self.__special_chr, [0xf0, offset_x // 16, offset_y // 16]]
        self.__sequence += [self.__special_chr, [0x01, 128 + (offset_x % 16), 128 + (offset_y % 16)]]
        return self

    def setBrushSize(self, width, height):
        self.__brush_width = width
        self.__brush_height = height
        self.__proportional = True
        div_gcd = 0
        for i in range(2, 17):
            if width % i == 0 and height % i == 0:
                div_gcd = i - 1

        self.__sequence += [self.__special_chr,
                            [0x02 + (div_gcd << 4),
                             width // (1 + div_gcd),
                             height // (1 + div_gcd)]]
        return self

    def offSetAbs(self, offset_x, offset_y):
        self.__sequence += [self.__special_chr, [0, 0, 0]]
        self.__sequence += [self.__special_chr, [0xf3, 128 + offset_x // 16, 128 + offset_y // 16]]
        self.__sequence += [self.__special_chr, [0x03, 128 + offset_x % 16, 128 + offset_y % 16]]
        return self

    def setBrushSizeAbs(self, width, height):
        self.__brush_width = width
        self.__brush_height = height
        self.__proportional = False
        div_gcd = 0
        for i in range(2, 17):
            if width % i == 0 and height % i == 0:
                div_gcd = i - 1

        self.__sequence += [self.__special_chr,
                            [0x04 + (div_gcd << 4),
                             width // (1 + div_gcd),
                             height // (1 + div_gcd)]]
        return self

    def setColor(self, r, g, b):
        width = self.__brush_width
        height = self.__brush_height
        if self.__proportional:
            self.setBrushSize(0, 0)
            self.__sequence += [[r, g, b]]
            self.setBrushSize(width, height)
        else:
            self.setBrushSizeAbs(0, 0)
            self.__sequence += [[r, g, b]]
            self.setBrushSizeAbs(width, height)
        return self

    def drawRect(self, width, height):
        i_width = self.__brush_width
        i_height = self.__brush_height
        if width < self.__brush_width or height < self.__brush_height:
            if self.__proportional:
                self.setBrushSize(width, height)
            else:
                self.setBrushSizeAbs(width, height)
        self.__sequence += [self.__special_chr,
                            [0x05,
                             width - self.__brush_width,
                             height - self.__brush_height]]
        if width < i_width or height < i_height:
            if self.__proportional:
                self.setBrushSize(i_width, i_height)
            else:
                self.setBrushSizeAbs(i_width, i_height)
        return self

    def drawRectF(self, width, height):
        i_width = self.__brush_width
        i_height = self.__brush_height
        if width < self.__brush_width or height < self.__brush_height:
            if self.__proportional:
                self.setBrushSize(width, height)
            else:
                self.setBrushSizeAbs(width, height)
        self.__sequence += [self.__special_chr,
                            [0x15,
                             width - self.__brush_width,
                             height - self.__brush_height]]
        if width < i_width or height < i_height:
            if self.__proportional:
                self.setBrushSize(i_width, i_height)
            else:
                self.setBrushSizeAbs(i_width, i_height)
        return self

    def drawLine(self, width, height):
        i_width = self.__brush_width
        i_height = self.__brush_height
        if width < self.__brush_width or height < self.__brush_height:
            if self.__proportional:
                self.setBrushSize(width, height)
            else:
                self.setBrushSizeAbs(width, height)
        self.__sequence += [self.__special_chr,
                            [0x25,
                             width - self.__brush_width,
                             height - self.__brush_height]]
        if width < i_width or height < i_height:
            if self.__proportional:
                self.setBrushSize(i_width, i_height)
            else:
                self.setBrushSizeAbs(i_width, i_height)
        return self

    def drawCircle(self, radius):
        i_width = self.__brush_width
        i_height = self.__brush_height
        if radius < self.__brush_width or radius < self.__brush_height:
            if self.__proportional:
                self.setBrushSize(radius, radius)
            else:
                self.setBrushSizeAbs(radius, radius)
        self.__sequence += [self.__special_chr,
                            [0x35,
                             radius - self.__brush_width,
                             radius - self.__brush_height]]
        if radius < i_width or radius < i_height:
            if self.__proportional:
                self.setBrushSize(i_width, i_height)
            else:
                self.setBrushSizeAbs(i_width, i_height)
        return self

    def drawCircleF(self, radius):
        i_width = self.__brush_width
        i_height = self.__brush_height
        if radius < self.__brush_width or radius < self.__brush_height:
            if self.__proportional:
                self.setBrushSize(radius, radius)
            else:
                self.setBrushSizeAbs(radius, radius)
        self.__sequence += [self.__special_chr,
                            [0x45,
                             radius - self.__brush_width,
                             radius - self.__brush_height]]
        if radius < i_width or radius < i_height:
            if self.__proportional:
                self.setBrushSize(i_width, i_height)
            else:
                self.setBrushSizeAbs(i_width, i_height)
        return self

    def drawTriangle(self, edge_ax, edge_ay, edge_bx, edge_by):
        i_width = self.__brush_width
        i_height = self.__brush_height
        if self.__proportional:
            self.setBrushSize(edge_ax, edge_ay)
        else:
            self.setBrushSizeAbs(edge_ax, edge_ay)
        self.__sequence += [self.__special_chr, [0x55, edge_bx, edge_by]]
        if self.__proportional:
            self.setBrushSize(i_width, i_height)
        else:
            self.setBrushSizeAbs(i_width, i_height)
        return self

    def drawTriangleF(self, edge_ax, edge_ay, edge_bx, edge_by):
        i_width = self.__brush_width
        i_height = self.__brush_height
        if self.__proportional:
            self.setBrushSize(edge_ax, edge_ay)
        else:
            self.setBrushSizeAbs(edge_ax, edge_ay)
        self.__sequence += [self.__special_chr, [0x65, edge_bx, edge_by]]
        if self.__proportional:
            self.setBrushSize(i_width, i_height)
        else:
            self.setBrushSizeAbs(i_width, i_height)
        return self

    def drawText(self, string: str):
        length = len(string)
        s_offset = 0
        padding = '\0' * 16
        while length > 0:
            piece_len = min(16, length)
            piece_str = (string + padding)[s_offset:s_offset + 16]
            cnt_reg = 0
            for c in piece_str[:piece_len]:
                if c == '\x00':
                    cnt_reg += 1
            length -= piece_len
            self.__sequence += [self.__special_chr,
                                [0x06 + ((piece_len - 1 - cnt_reg) << 4),
                                 ord(string[s_offset]),
                                 ord(string[s_offset + 1])]]
            for i in range(piece_len // 3):
                self.__sequence += [[ord(piece_str[i + 2]), ord(piece_str[i + 3]), ord(piece_str[i + 4])]]
            s_offset += piece_len
        return self

    def addShader(self, dred, dgreen, dblue, dalpha):
        delta = int((dred * dred + dgreen * dgreen + dblue * dblue) ** 0.5)
        theta = min(15, int(math.acos(dblue / delta) / math.pi * 8))
        phi = min(15, int(math.atan2(dgreen, dred) / math.pi * 8))
        rem_mod_alpha = abs(dalpha)
        while rem_mod_alpha > 0:
            adelta = max(-64, min(63, dalpha))
            dalpha -= adelta
            rem_mod_alpha -= abs(adelta)
            self.__sequence += [self.__special_chr, [0x07, 0, 0 + 64 + adelta]]
        while delta > 0:
            p_len = min(63, delta)
            delta -= p_len
            self.__sequence += [self.__special_chr,
                                [0x07 + (phi << 4),
                                 (theta << 5) + (p_len // 2) & 31,
                                 (p_len & 1) << 7]]
        return self

    def addButton(self, width, height, reg, flags_on, flags_off):
        i_width = self.__brush_width
        i_height = self.__brush_height
        if self.__proportional:
            self.setBrushSize(width, height)
        else:
            self.setBrushSizeAbs(width, height)
        self.__sequence += [self.__special_chr, [0x08 + (reg << 4), flags_on, flags_off]]
        if self.__proportional:
            self.setBrushSize(i_width, i_height)
        else:
            self.setBrushSizeAbs(i_width, i_height)
        return self

    def injectPredicate(self, reg, predicate, predicate_type, length):
        self.__sequence += [self.__special_chr, [0x09 + (reg << 4), length + (predicate_type << 6), predicate]]
        return self

    def injectAsm(self, nasm: str):
        instructions = nasm.split('\n')
        saddrl = len(self.__sequence)
        l_cnt = 0
        in_macro = False
        macro_name = ''
        macro_sequence = []
        for instruction in instructions:

            if len(instruction) == 0 or instruction[0] == ';':
                continue
            instruction = instruction.split(';')[0]
            skip_white_idx = 0
            while skip_white_idx < len(instruction) and\
                    (instruction[skip_white_idx] == ' ' or
                     instruction[skip_white_idx] == '\t'):
                skip_white_idx += 1
            instruction = instruction[skip_white_idx:]

            if len(instruction) == 0:
                continue

            skip_white_idx = len(instruction) - 1
            while skip_white_idx > 0 and\
                    (instruction[skip_white_idx] == ' ' or
                     instruction[skip_white_idx] == '\t'):
                skip_white_idx -= 1
            instruction = instruction[:skip_white_idx + 1]

            if len(instruction) == 0:
                continue

            # print(f"{0}: \"{instruction}\"")

            tokens = instruction.split() + ['0']
            op = tokens[0]
            arg1 = tokens[1]
            if op == 'label':
                if arg1[-1] == ':':
                    arg1 = arg1[:-1]
                self.__labels[arg1] = saddrl + 2 * l_cnt
                continue
            else:
                l_cnt += 1
        for instruction in instructions:
            instr_code = 0
            regA_code = 0
            regB_code = 0
            immediate = 0

            if len(instruction) == 0 or instruction[0] == ';':
                continue
            instruction = instruction.split(';')[0]
            skip_white_idx = 0
            while skip_white_idx < len(instruction) and\
                    (instruction[skip_white_idx] == ' ' or
                     instruction[skip_white_idx] == '\t'):
                skip_white_idx += 1
            instruction = instruction[skip_white_idx:]

            if len(instruction) == 0:
                continue

            skip_white_idx = len(instruction) - 1
            while skip_white_idx > 0 and\
                    (instruction[skip_white_idx] == ' ' or
                     instruction[skip_white_idx] == '\t'):
                skip_white_idx -= 1
            instruction = instruction[:skip_white_idx + 1]

            if len(instruction) == 0:
                continue

            # print(f"{0}: \"{instruction}\"")
            instruction = instruction.replace(',', ' ')
            tokens = instruction.split() + ['0', '', '']
            print(tokens)
            op = tokens[0]
            arg1 = tokens[1]
            if op == 'macro':
                if in_macro:
                    raise ValueError("Macros inside macros are not allowed!")
                in_macro = True
                macro_name = arg1
                self.__macros[arg1] = ([], [], {})
                continue
            if op == '}' and arg1 == '0':
                if not in_macro:
                    raise ValueError("End of macro expected!")
                in_macro = False
                continue
            if op == 'label':
                continue
            if len(arg1) > 1 and arg1[0] == 'r':
                regA_code = int(arg1[1:])
            arg2 = tokens[2]
            if len(arg2) > 1 and arg2[0] == 'r':
                regB_code = int(arg2[1:])
            arg3 = tokens[3]

            if in_macro:
                '''
                mov r32 242
                add r32 r56
                mov r33 0
                mul r33 r56
                add r33 0
                str r32, r33
                '''
                if op == 'label':
                    self.__macros[macro_name][2][arg1]
                if op == 'mov':
                    if arg2[0] == 'r':
                        macro_sequence += [(10, regA_code, regB_code)]
                    else:
                        macro_sequence += [(0, regA_code, int(arg2))]
                if op == 'add':
                    if arg2[0] == 'r':
                        macro_sequence += [(1, regA_code, regB_code)]
                    else:
                        macro_sequence += [(9, regA_code, int(arg2) + 128)]
                if op == 'inc':
                    macro_sequence += [(9, regA_code, 129)]
                if op == 'dec':
                    macro_sequence += [(9, regA_code, 127)]
                if op == 'sub':
                    if arg2[0] == 'r':
                        macro_sequence += [(2, regA_code, regB_code)]
                    else:
                        macro_sequence += [(9, regA_code, 128 - int(arg2))]
                if op == 'neg':
                    macro_sequence += [(8, regA_code, int(arg2) + 128)]
                if op == 'not':
                    macro_sequence += [(7, 255, regA_code)]
                if op == 'mul':
                    macro_sequence += [(3, regA_code, regB_code)]
                if op == 'div':
                    macro_sequence += [(4, regA_code, regB_code)]
                if op == 'or':
                    macro_sequence += [(5, regA_code, regB_code)]
                if op == 'and':
                    macro_sequence += [(6, regA_code, regB_code)]
                if op == 'xor':
                    macro_sequence += [(7, regA_code, regB_code)]
                if op == 'mod':
                    macro_sequence += [(13, regA_code, regB_code)]
                if op == 'lsh':
                    macro_sequence += [(11, regA_code, int(arg2) + 128)]
                if op == 'rsh':
                    macro_sequence += [(11, regA_code, -int(arg2) + 128)]
                if op == 'cmp':
                    macro_sequence += [(12, regA_code, int(arg2))]
                if op == 'jiz':
                    macro_sequence += [(14, regA_code, regB_code)]
                if op == 'str':
                    macro_sequence += [(15, regA_code, regB_code)]
                if op == 'ljl':
                    target = 0
                    if arg3 in self.__labels:
                        target = 32768 + self.__labels[arg3] - len(self.__sequence) - 6
                        self.__sequence += [self.__special_chr,
                                            [0x0a,
                                             regA_code,
                                             target // 256]]
                    else:
                        self.__label_index_queue.append((arg3, len(self.__sequence)))
                        self.__sequence += [self.__special_chr, [0x0a, regA_code, 0]]
                    regA_code = regB_code
                    regB_code = 0
                    immediate = target % 256
                    instr_code = 0
                if op == 'lad':
                    target = 32768 + len(self.__sequence) - 2
                    self.__sequence += [self.__special_chr,
                                        [0x0a,
                                         regA_code,
                                         target // 256]]
                    regA_code = regB_code
                    regB_code = 0
                    immediate = target % 256
                    instr_code = 0
                if op == 'sbrs':
                    self.__sequence += [self.__special_chr,
                                        [0x0b,
                                         regA_code,
                                         regB_code]]
                    continue
                if op == 'soffs':
                    self.__sequence += [self.__special_chr,
                                        [0x1b,
                                         regA_code,
                                         regB_code]]
                    continue
                if op == 'aoffs':
                    self.__sequence += [self.__special_chr,
                                        [0x2b,
                                         regA_code,
                                         regB_code]]
                    continue

                self.__sequence += [self.__special_chr,
                                    [0x0a + (instr_code << 4),
                                    regA_code,
                                    regB_code + immediate]]



                if op == 'mov':
                    if arg2[0] == 'r':
                        macro_sequence += [(10, regA_code, regB_code)]
                    else:
                        macro_sequence += [(10, regA_code, int(arg2))]
                continue

            if op == 'mov':
                if arg2[0] == 'r':
                    instr_code = 10
                else:
                    instr_code = 0
                    immediate = int(arg2)
            if op == 'add':
                if arg2[0] == 'r':
                    instr_code = 1
                else:
                    instr_code = 9
                    immediate = int(arg2) + 128
            if op == 'inc':
                instr_code = 9
                immediate = 129
            if op == 'dec':
                instr_code = 9
                immediate = 127
            if op == 'sub':
                if arg2[0] == 'r':
                    instr_code = 2
                else:
                    instr_code = 9
                    immediate = -int(arg2) + 128
            if op == 'neg':
                instr_code = 8
                immediate = 128
                if arg2 != '0':
                    immediate += int(arg2)
            if op == 'not':
                instr_code = 7
                regB_code = 255
            if op == 'mul':
                instr_code = 3
            if op == 'div':
                instr_code = 4
            if op == 'or':
                instr_code = 5
            if op == 'and':
                instr_code = 6
            if op == 'xor':
                instr_code = 7
            if op == 'mod':
                instr_code = 13
            if op == 'lsh':
                instr_code = 11
                immediate = int(arg2) + 128
            if op == 'rsh':
                instr_code = 11
                immediate = -int(arg2) + 128
            if op == 'cmp':
                instr_code = 12
                immediate = int(arg2)
            if op == 'jiz':
                instr_code = 14
                # print(arg1, arg2, arg3, regA_code, regB_code, immediate)
            if op == 'str':
                instr_code = 15
            if op == 'ljl':
                target = 0
                if arg3 in self.__labels:
                    target = 32768 + self.__labels[arg3] - len(self.__sequence) - 6
                    self.__sequence += [self.__special_chr,
                                        [0x0a,
                                         regA_code,
                                         target // 256]]
                else:
                    self.__label_index_queue.append((arg3, len(self.__sequence)))
                    self.__sequence += [self.__special_chr, [0x0a, regA_code, 0]]
                regA_code = regB_code
                regB_code = 0
                immediate = target % 256
                instr_code = 0
            if op == 'lad':
                target = 32768 + len(self.__sequence) - 2
                self.__sequence += [self.__special_chr,
                                    [0x0a,
                                     regA_code,
                                     target // 256]]
                regA_code = regB_code
                regB_code = 0
                immediate = target % 256
                instr_code = 0
            if op == 'sbrs':
                self.__sequence += [self.__special_chr,
                                    [0x0b,
                                     regA_code,
                                     regB_code]]
                continue
            if op == 'soffs':
                self.__sequence += [self.__special_chr,
                                    [0x1b,
                                     regA_code,
                                     regB_code]]
                continue
            if op == 'aoffs':
                self.__sequence += [self.__special_chr,
                                    [0x2b,
                                     regA_code,
                                     regB_code]]
                continue

            self.__sequence += [self.__special_chr,
                                [0x0a + (instr_code << 4),
                                regA_code,
                                regB_code + immediate]]
        return self

    def compile(self):
        r = 1
        g = 1
        b = 1
        while True:
            r = (r * 4245675789890798369) % 489675327681678097
            g = (g * 2312435698000990827) % 145679786543225031
            b = (b * 2435687798090904367) % 567789087653267893
            rn = (r // 79) % 256
            gn = (g // 61) % 256
            bn = (b // 97) % 256
            col = (rn << 16) + (gn << 8) + bn
            if not (col in self.__taken_chrs):
                self.__special_chr[0] = rn
                self.__special_chr[1] = gn
                self.__special_chr[2] = bn
                break
        for i, v in enumerate(self.__label_index_queue):
            target = 32768 + self.__labels[v[0]] - v[1] - 6
            self.__sequence[v[1] + 1][2] = target // 256
            self.__sequence[v[1] + 3][2] = target % 256
        enc_seq, tree, cols = encodeSequence(self.__sequence)
        return convert8To6BIT(enc_seq).encode(), convert8To6BIT(tree).encode(), encodeColors([self.__special_chr] + cols).encode()

    def printSeq(self, sep=' ', enum=False):
        if enum:
            print(*[f"{i}: {v}" for i, v in enumerate(self.__sequence)], sep=sep)
        else:
            print(*self.__sequence, sep=sep)


def bits(byte_list, f):
    return ''.join(list(map(lambda b: '0' * (10 - len(bin(f(b)))) + bin(f(b))[2:], byte_list)))


# # a, b, c = encodeSequence([[1, 10, 1], [198, 127, 202],
# #         [1, 10, 1], [198, 127, 202]] + seq)
# a, b, c = encodeSequence(vseq)
# # print(bits(a, ord), bits(b, ord))
# # print(c[10])
# a, b, c = convert8To6BIT(a), convert8To6BIT(b), encodeColors([[1, 10, 1]] + c)
# print(a.encode())
# print(b.encode())
# print(c.encode())


icon_image = getImage('PixelArtToIsoline/data/assets/minecraft/textures/block/copper_ore.png')
rgb_ii = cv2.cvtColor(icon_image, cv2.COLOR_BGR2RGB)
sc, enc, im = encodeImage('imageColorSpaces/settings_icon.png.data', 16)

sc_arr = np.zeros((len(sc), 4))
for i, key in enumerate(sc):
    rgb = np.zeros((4,))
    rgb[0] = sc[i][2]
    rgb[1] = sc[i][1]
    rgb[2] = sc[i][0]
    rgb[3] = sc[i][3]
    sc_arr[i] = rgb

showColorSpace(sc_arr)
cv2.imshow("im", im)
cv2.waitKey(0)




#
#
# def fib(n):
#     if n < 2:
#         return n
#     return fib(n - 1) + fib(n - 2)
#
#
# # sb = SequenceBuilder(32, 32)
# # sb.addImage(0, 0, rgb_ii)
# # sb.offSet(0, 0)
# # sb.setColor(1, 250, 1)
# # sb.injectAsm('mov r10 0\n'
# #              'mov r11 1\n'
# #              'mov r12 15\n'
# #              'label L0\n'
# #              'mov r13 r11\n'
# #              'add r11 r10\n'
# #              'mov r10 r13\n'
# #              'dec r12\n'
# #              'cmp r12 0\n'
# #              'ljl r14 r15 L0\n'
# #              'jiz r14 r15\n'
# #              'add r2 2\n'
# #              'add r3 3\n'
# #              'mov r4 64\n'
# #              'mod r2 r4\n'
# #              'mod r3 r4\n'
# #              'soffs r2 r3')
# # sb.drawText('\x00\x0b')
# # # sb.printSeq()
# # print(fib(14), fib(15), fib(16))
# # sb.setColor(250, 1, 250)
# # sb.drawTriangleF(16, 0, 8, 8)
# # sb.drawCircleF(2)
# # # sb.printSeq()
#
#
# sb = SequenceBuilder(64, 64)
#
# sb.setColor(120, 220, 120)
# # sb.printSeq()
# sb.setBrushSizeAbs(2, 2)
# sb.offSetAbs(10, 10)
# sb.drawRectF(32, 32)
# # sb.printSeq()
# sb.injectAsm('''
#     cmp r4 1
#     ljl r30, r31, animation
#     jiz r30, r31
#     mov r4 1
#     mov r41 120
#     mov r42 120
#     mov r43 120
#     macro m {
#         mov r10 0
#         mov r20 0
#     }
# ''')
# sb.setColor(255, 255, 255)
#
# # px_cols[user_section + stcol2 // 256] = {stcol2 & 255, stcol3 // 256, stcol3 & 255}
# # state[col2] = stcol2 + 256
#
# sb.injectAsm('''
#     label animation:
#
#     add r41 3
#     add r42 7
#     add r43 13
#     mod r41 r56
#     mod r42 r56
#     mod r43 r56
#
#     mov r32 r1
#     mov r33 r2
#     mul r33 r56
#     add r33 r3
#     str r32, r33
#
#     mov r32 242
#     add r32 r56
#     mov r33 0
#     mul r33 r56
#     add r33 0
#     str r32, r33
#
#     mov r32 r41
#     add r32 r56
#     add r32 r56
#     mov r33 r42
#     mul r33 r56
#     add r33 r43
#     str r32, r33
#
#     mov r32 r1
#     add r32 r56
#     add r32 r56
#     add r32 r56
#     mov r33 r2
#     mul r33 r56
#     add r33 r3
#     str r32, r33
#
#     mov r32 242
#     add r32 r56
#     add r32 r56
#     add r32 r56
#     add r32 r56
#     mov r33 0
#     mul r33 r56
#     add r33 0
#     str r32, r33
#
# ''')
#
#
# # sb.drawText('\x00\x38')
#
#
# print(*sb.compile(), sep='\n')
# sb.printSeq(sep=f'\n', enum=True)
# # print(decTree('\x23!,(Z.+, ',
# #               '\x20\x22\x24\x26\x28\x2a\x2c\x2e\x30\x32\x34\x36\x38\x3a\x3c\x3e\x40',
# #               1))



