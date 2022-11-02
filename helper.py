import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
import nmslib


def plot_radiomap(radiomap, min_data, max_data, beacon, centroid=None, error=0):
    sns.color_palette("flare", as_cmap=True)
    ax = sns.heatmap(radiomap, cmap="flare",
                    vmin=min_data, vmax=max_data,linewidth=.01)
    plt.title(f"[GPR model]{beacon}")
    plt.xlabel('x-index (1m x 1m square)', fontsize=10)
    plt.ylabel('y-index (1m x 1m square)', fontsize=10)

    # print centroids
    if centroid is not None:
        measure_centroid = get_centroid_measured(radiomap)
        plt.scatter(measure_centroid[1], measure_centroid[0])
        plt.scatter(centroid[1], centroid[0])

    plt.savefig(f'{beacon} GPR')
    plt.show()
    return plt


def get_centroid_measured(radiomap):
    centroid_measured = np.array(
        np.where(radiomap == 1)).reshape(1, -1)[0] + 0.5
    return centroid_measured


def get_fingerprints(radiomaps):
    data = []
    flatten_size = len(radiomaps[0])
    for cell in range(flatten_size):
        fingerprint = []
        for radiomap in radiomaps:
            fingerprint.append(radiomap[cell])
        # if not np.isin(OUT_OF_RANGE_RSSI, fingerprint):
        #     data.append(fingerprint)
        data.append(fingerprint)
    return data


def flatten_to_cell(n_rows, n_cols, id):
    col = id % n_cols
    row = id//n_cols
    return np.array([row, col])


def get_centroid_cell(cells, distances):
    scores = 1/distances
    total_score = scores.sum()

    centroid_cell = (cells.T*scores).sum(axis=1)/total_score
    return centroid_cell+0.5


def check_position_pred_accuracy(n_rows=8, n_cols=10, radiomaps=None, testing_data=None, bounds=None):
    data = get_fingerprints(radiomaps=radiomaps)
    # initialize a new index, using a HNSW index on Cosine Similarity
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(data)
    index.createIndex({'post': 2}, print_progress=True)

    RMSErr = 0
    for idx, rdata in testing_data.iterrows():
        # prepare for plot
        location_map = np.full((n_rows, n_cols), -1)

        # query for the nearest neighbours of the first datapoint
        ids, distances = index.knnQuery(rdata[2:].values, k=3)
        cells = list(map(lambda idx: flatten_to_cell(
            n_rows=n_rows, n_cols=n_cols, id=idx), ids))
        cells = np.array(cells)

        # fill in into radiomap
        for cell in cells:
            location_map[cell[0], cell[1]] = 0

        col_idx = rdata['col']
        row_idx = rdata['row']
        print(f'Test data ({row_idx},{col_idx}):', rdata[2:].values)
        print(f'Test coord {cell_to_coord(n_rows, n_cols, bounds, (row_idx+0.5,col_idx+0.5))}')
        # fill in into radiomap
        location_map[row_idx, col_idx] = 1
        # get centroids to estimate error
        centroid_measured = get_centroid_measured(location_map)
        centroid_pred = get_centroid_cell(cells, distances)
        # get centroid coords
        centroid_coord= cell_to_coord(n_rows, n_cols, bounds, centroid_pred)

        err = np.linalg.norm(centroid_measured - centroid_pred)
        RMSErr = RMSErr + err**2
        print('3 nearest neighbours (array idx):', ids, distances)
        print(f'3 nearest neighbours (matrix idx):\n({cells})')
        print(f'Centroid cell: ({centroid_pred})')
        print(f'Centroid coord: ({centroid_coord})')

        print('Error:', err)
        # print(f'Distances: {distances}')

        # plot location map
        plot_radiomap(location_map, min_data=-1, max_data=1,
                    beacon=idx, centroid=centroid_pred, error=err)
        print('=====================================')
    RMSErr = np.sqrt(RMSErr / len(testing_data))
    print('RMSE: ', RMSErr)


def get_ratio_pos(start_vector, end_vector, range, idx):
    pos = idx/range*end_vector + (range-idx)/range*start_vector
    return pos

def cell_to_coord(n_rows, n_cols, bounds, cell):
    row1 = get_ratio_pos(bounds[0],bounds[3],n_rows, cell[0])
    row2 = get_ratio_pos(bounds[1],bounds[2],n_rows, cell[0])
    coord = get_ratio_pos(row1,row2, n_cols, cell[1])
    return coord