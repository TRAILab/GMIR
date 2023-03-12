import numpy as np
import tensorflow as tf
import cv2
from lib.LiDAR_snow_sim.tools.visual_utils import open3d_vis_utils as V
#from tools.visual_utils import open3d_vis_utils as V
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
import open3d as o3d
from scipy import stats


#https://www.wevolver.com/specs/hdl-64e.lidar.sensor
BEAM_INCLINATION_MAX = np.radians(2.0) #np.radians(9.7188225)
BEAM_INCLINATION_MIN = np.radians(-24.9) #np.radians(-22.77476)


#https://www.manualslib.com/download/1988532/Velodyne-Hdl-64e-S3.html
horizontal_res = np.radians(0.1728) #3e-3 in snow sim 
HEIGHT = 64 # channels of lidar
WIDTH = np.ceil(np.radians(360)/horizontal_res).astype(int)

# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)



def filter_below_groundplane(pointcloud, tolerance=1, w=None, h=None):
    if w is None or h is None:
        valid_loc = (pointcloud[:, 2] < -1.4) & \
                    (pointcloud[:, 2] > -1.86) & \
                    (pointcloud[:, 0] > 0) & \
                    (pointcloud[:, 0] < 40) & \
                    (pointcloud[:, 1] > -15) & \
                    (pointcloud[:, 1] < 15)
        pc_rect = pointcloud[valid_loc]
        print(pc_rect.shape)
        if pc_rect.shape[0] <= pc_rect.shape[1]:
            w = [0, 0, 1]
            h = -1.55
        else:
            reg = RANSACRegressor().fit(pc_rect[:, [0, 1]], pc_rect[:, 2])
            w = np.zeros(3)
            w[0] = reg.estimator_.coef_[0]
            w[1] = reg.estimator_.coef_[1]
            w[2] = 1.0
            h = reg.estimator_.intercept_
            w = w / np.linalg.norm(w)

            print(reg.estimator_.coef_)
            print(reg.get_params())
            print(w, h)
    height_over_ground = np.matmul(pointcloud[:, :3], np.asarray(w))
    height_over_ground = height_over_ground.reshape((len(height_over_ground), 1))
    above_ground = np.matmul(pointcloud[:, :3], np.asarray(w)) - h > -tolerance
    print(above_ground.shape)
    #return np.hstack((pointcloud[above_ground, :], height_over_ground[above_ground]))
    return pointcloud[above_ground, :], w, h

def _combined_static_and_dynamic_shape(tensor):
    """Returns a list containing static and dynamic values for the dimensions.

    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.

    Args:
        tensor: A tensor of any type.

    Returns:
        A list of size tensor.shape.ndims containing integers or a scalar tensor.
    """
    static_tensor_shape = tensor.shape.as_list()
    dynamic_tensor_shape = tf.shape(input=tensor)
    combined_shape = []
    for index, dim in enumerate(static_tensor_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_tensor_shape[index])
    return combined_shape

def scatter_nd_with_pool(index,
                         value,
                         shape,
                         pool_method=tf.math.unsorted_segment_max):
    """Similar as tf.scatter_nd but allows custom pool method.

    tf.scatter_nd accumulates (sums) values if there are duplicate indices.

    Args:
        index: [N, 2] tensor. Inner dims are coordinates along height (row) and then
        width (col).
        value: [N, ...] tensor. Values to be scattered.
        shape: (height,width) list that specifies the shape of the output tensor.
        pool_method: pool method when there are multiple points scattered to one
        location.

    Returns:
        image: tensor of shape with value scattered. Missing pixels are set to 0.
    """
    if len(shape) != 2:
        raise ValueError('shape must be of size 2')
    height = shape[0]
    width = shape[1]
    # idx: [N]
    index_encoded, idx = tf.unique(index[:, 0] * width + index[:, 1])
    value_pooled = pool_method(value, idx, tf.size(input=index_encoded))
    index_unique = tf.stack(
        [index_encoded // width,
        tf.math.mod(index_encoded, width)], axis=-1)
    shape = [height, width]
    value_shape = _combined_static_and_dynamic_shape(value)
    if len(value_shape) > 1:
        shape = shape + value_shape[1:]

    image = tf.scatter_nd(index_unique, value_pooled, shape)
    return image

def compute_inclinations(beam_inclination_min, beam_inclination_max):
    """Computes uniform inclination range based on the given range and height i.e. number of channels.
    """
    diff = beam_inclination_max - beam_inclination_min
    #ratios = (0.5 + np.arange(0, HEIGHT)) / HEIGHT #[0.5, ..., 63.5] #waymo
    ratios = np.arange(0, HEIGHT) / (HEIGHT-1) #[0, ..., 63]
    inclination = ratios * diff + beam_inclination_min #[bottom row inclination, ..., top row]
    #reverse
    inclination = inclination[::-1]
    return inclination

def compute_range_image_polar(range_image, inclination):
    """Computes range image polar coordinates.

    Args:
        range_image: [H, W] tensor. Lidar range images.
        inclination: [H] tensor. Inclination for each row of the range image.
        0-th entry corresponds to the 0-th row of the range image.

    Returns:
        range_image_polar: [H, W, 3] polar coordinates.
    """
    # [W].
    #ratios = (np.arange(WIDTH, 0, -1) - 0.5) / WIDTH # [0.99, ..., 0.000] #waymo
    ratios = (np.arange(WIDTH, 0, -1) - 1) / (WIDTH-1) #[1, ..., 0]
    
    # [W].
    azimuth = (ratios * 2. - 1.) * np.pi # [180 deg in rad, ..., -180]

    # [H, W]
    azimuth_tile = np.tile(azimuth.reshape((1, WIDTH)), (HEIGHT, 1))
    # [H, W]
    inclination_tile = np.tile(inclination.reshape((HEIGHT, 1)), (1, WIDTH))

    #[H, W, 3]
    range_image_polar = np.stack([azimuth_tile, inclination_tile, range_image],
                                 axis=-1)
    return range_image_polar
def compute_range_image_cartesian(range_image_polar):

    """Computes range image cartesian coordinates from polar ones.

    Args:
        range_image_polar: [B, H, W, 3] float tensor. Lidar range image in polar
        coordinate in sensor frame.
        extrinsic: [B, 4, 4] float tensor. Lidar extrinsic.
        pixel_pose: [B, H, W, 4, 4] float tensor. If not None, it sets pose for each
        range image pixel.
        frame_pose: [B, 4, 4] float tensor. This must be set when pixel_pose is set.
        It decides the vehicle frame at which the cartesian points are computed.
        dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
        scope: the name scope.

    Returns:
        range_image_cartesian: [B, H, W, 3] cartesian coordinates.
    """
    azimuth, inclination, range_image_range = range_image_polar[:,:,0], range_image_polar[:,:,1], range_image_polar[:,:,2]

    cos_azimuth = np.cos(azimuth)
    sin_azimuth = np.sin(azimuth)
    cos_incl = np.cos(inclination)
    sin_incl = np.sin(inclination)

    # [H, W].
    x = cos_azimuth * cos_incl * range_image_range
    y = sin_azimuth * cos_incl * range_image_range
    z = sin_incl * range_image_range

    # [H, W, 3]
    range_image_points = np.stack([x, y, z], -1)
    
    return range_image_points

def extract_point_cloud_from_range_image(range_image, inclination):
    """Extracts point cloud from range image.

    Args:
        range_image: [H, W] tensor. Lidar range images.
        inclination: [H] tensor. Inclination for each row of the range image.
        0-th entry corresponds to the 0-th row (top row) of the range image.

    Returns:
        range_image_cartesian: [H, W, 3] with {x, y, z} as inner dims in vehicle
        frame.
    """
    range_image_polar = compute_range_image_polar(range_image,  inclination)
    range_image_cartesian = compute_range_image_cartesian(range_image_polar)
    return range_image_cartesian

def o3d_dynamic_radius_outlier_filter(pc: np.ndarray, alpha: float = 0.45, beta: float = 3.0,
                                    k_min: int = 3, sr_min: float = 0.04) -> np.ndarray:
        """
        :param pc:      pointcloud
        :param alpha:   horizontal angular resolution of the lidar
        :param beta:    multiplication factor
        :param k_min:   minimum number of neighbors
        :param sr_min:  minumum search radius

        :return:        mask [False = snow, True = no snow]
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
        num_points = len(pcd.points)

        # initialize mask with False
        mask = np.zeros(num_points, dtype=bool)

        k = k_min + 1

        kd_tree = o3d.geometry.KDTreeFlann(pcd)

        for i in range(num_points):

            x = pc[i,0]
            y = pc[i,1]

            r = np.linalg.norm([x, y], axis=0)

            sr = alpha * beta * np.pi / 180 * r

            if sr < sr_min:
                sr = sr_min

            [_, _, sqdist] = kd_tree.search_knn_vector_3d(pcd.points[i], k)

            neighbors = -1      # start at -1 since it will always be its own neighbour

            for val in sqdist:
                if np.sqrt(val) < sr:
                    neighbors += 1

            if neighbors >= k_min:
                mask[i] = True  # no snow -> keep

        return mask

def build_range_image_from_point_cloud(points, inclination, point_features=None, channel=None):
    """Build virtual range image from point cloud assuming uniform azimuth.

    Args:
    points: tf tensor with shape [B, N, 3]
    inclination: tf tensor of shape [B, H] that is the inclination angle per
        row. sorted from highest value to lowest.
    point_features: If not None, it is a tf tensor with shape [B, N, 2] that
        represents lidar 'intensity' and 'elongation'.

    Returns:
    range_images : [B, H, W, 3] or [B, H, W] tensor. Range images built from the
        given points. Data type is the same as that of points_vehicle_frame. 0.0
        is populated when a pixel is missing.
    ri_indices: tf int32 tensor [B, N, 2]. It represents the range image index
        for each point.
    ri_ranges: [B, N] tensor. It represents the distance between a point and
        sensor frame origin of each point.
    """

    # [B, N]
    xy_norm = np.linalg.norm(points[:, 0:2], axis=-1)
    # [B, N]
    point_inclination = np.arctan2(points[..., 2], xy_norm)

    # channel_wise_min_max_incl = np.zeros((64, 3))
    # for i in range(64):
    #     idx = channel == i
    #     max_inc = point_inclination[idx].max()
    #     min_inc = point_inclination[idx].min()
    #     median_inc = np.median(point_inclination[idx])
    #     channel_wise_min_max_incl[i, 0] = min_inc
    #     channel_wise_min_max_incl[i, 1] = max_inc
    #     channel_wise_min_max_incl[i, 2] = median_inc
    #     b=1

    # [B, N, H]
    point_inclination_diff = np.abs(inclination.reshape((1,-1)) - point_inclination.reshape((-1,1)))
    # [B, N]
    point_ri_row_indices = np.argmin(point_inclination_diff, axis=-1)

    # [B, N], within [-pi, pi]
    point_azimuth = np.arctan2(points[..., 1], points[..., 0])

    # point_azimuth_gt_pi_mask = point_azimuth > np.pi
    # point_azimuth_lt_minus_pi_mask = point_azimuth < -np.pi
    # point_azimuth = point_azimuth - point_azimuth_gt_pi_mask * 2 * np.pi
    # point_azimuth = point_azimuth + point_azimuth_lt_minus_pi_mask * 2 * np.pi

    # [B, N].
    point_ri_col_indices = WIDTH - 1.0 + 0.5 - (point_azimuth + np.pi) / (2.0 * np.pi) * WIDTH
    point_ri_col_indices = np.round(point_ri_col_indices).astype(int)

    # with tf.control_dependencies([
    #     tf.compat.v1.assert_non_negative(point_ri_col_indices),
    #     tf.compat.v1.assert_less(point_ri_col_indices, tf.cast(width, tf.int32))
    # ]):
    # [B, N, 2]
    ri_indices = np.stack([point_ri_row_indices, point_ri_col_indices], -1)
    # [B, N]
    ri_ranges = np.linalg.norm(points, axis=-1)

    #Convert to tensor
    ri_indices = tf.convert_to_tensor(ri_indices)
    ri_ranges = tf.convert_to_tensor(ri_ranges) 
    
    def build_range_image(args):
        """Builds a range image for each frame.

        Args:
            args: a tuple containing:
            - ri_index: [N, 2] int tensor.
            - ri_value: [N] float tensor.
            - num_point: scalar tensor
            - point_feature: [N, 2] float tensor.

        Returns:
            range_image: [H, W]
        """
        if len(args) == 3:
            ri_index, ri_value, num_point = args
        else:
            ri_index, ri_value, num_point, point_feature = args
            ri_value = tf.concat([ri_value[..., tf.newaxis], point_feature],
                                axis=-1)
            #ri_value = encode_lidar_features(ri_value)

        # pylint: disable=unbalanced-tuple-unpacking
        ri_index = ri_index[0:num_point, :]
        ri_value = ri_value[0:num_point, ...]
        range_image = scatter_nd_with_pool(ri_index, ri_value, [HEIGHT, WIDTH],
                                            tf.math.unsorted_segment_min)
        # if len(args) != 3:
        #     range_image = decode_lidar_features(range_image)
        return range_image

    num_points = ri_ranges.shape[0]
    elems = [ri_indices, ri_ranges, num_points]
    # if point_features is not None:
    #     elems.append(point_features)
    # range_images = tf.map_fn(
    #     fn, elems=elems, dtype=points_vehicle_frame_dtype, back_prop=False)
    
    range_images = build_range_image(elems)
    return range_images.numpy(), ri_indices.numpy(), ri_ranges.numpy()


def fill_max(range_image, empty_pixels, kernel=DIAMOND_KERNEL_5):
    # Fill empty pixels with highest range in the kernel
    dilated = cv2.dilate(range_image, kernel)
    range_image[empty_pixels] = dilated[empty_pixels]
    return range_image

def fill_based_on_freq(range_image, empty_pixels, mode='max', freq_mode='least', kernel=DIAMOND_KERNEL_5):

    #Fill empty pixels with max, min or median of most freq range bin in neighbours
    pad = 2
    padded_range_image = np.pad(range_image, pad, 'edge')
    new_padded_range_image = padded_range_image.copy()
    range_bins = np.arange(0,121,10)

    for row in range(pad, pad+HEIGHT):
        for col in range(pad, pad+WIDTH):
            #         # 5x5 diamond kernel
            # DIAMOND_KERNEL_5 = np.array(
            #     [
            #         [0, 0, 1, 0, 0],
            #         [0, 1, 1, 1, 0],
            #         [1, 1, 1, 1, 1],
            #         [0, 1, 1, 1, 0],
            #         [0, 0, 1, 0, 0],
            #     ], dtype=np.uint8)

            img_under_kernel = padded_range_image[row-pad:row+pad+1, col-pad:col+pad+1]
            values = img_under_kernel[kernel.astype(bool)]
            
            bin_counts, bin_edges, bin_num = stats.binned_statistic(values, values, 'count', bins = range_bins)
            bin_counts[bin_counts < 1] = np.nan

            if freq_mode == 'least':
                selected_bin_idx = np.nanargmin(bin_counts) #this returns index which starts from 0
            else:
                selected_bin_idx = np.nanargmax(bin_counts)
            selected_bin_values = values[bin_num == selected_bin_idx+1] #bin num starts from 1, 2, ...
            if mode == 'max':
                new_val = np.max(selected_bin_values) 
            elif mode == 'min':
                new_val = np.min(selected_bin_values)
            elif mode == 'mean':
                new_val = np.min(selected_bin_values)
            elif mode == 'median':
                new_val = np.median(selected_bin_values)
            #if median > 0:
            new_padded_range_image[row, col] = new_val #if median > 3 else padded_range_image[row, col]

    dilated = new_padded_range_image[pad:-pad, pad:-pad]
    range_image[empty_pixels] = dilated[empty_pixels]

    return range_image

# ip_basic
def fill_in_fast(depth_map, max_depth=100.0, custom_kernel=DIAMOND_KERNEL_5,
                 extrapolate=False, blur_type='bilateral'):
    """Fast, in-place depth completion.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        custom_kernel: kernel to apply initial dilation
        extrapolate: whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'bilateral' - preserves local structure (recommended)
            'gaussian' - provides lower RMSE

    Returns:
        depth_map: dense depth map
    """
    # vis_utils.cv2_show_image('initial', depth_map)
    # cv2.waitKey()

    # Invert
    valid_pixels = (depth_map > 0.1) #non-empty pixels
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # vis_utils.cv2_show_image('invert depth', depth_map*256)
    # cv2.waitKey()

    # Dilate
    depth_map = cv2.dilate(depth_map, custom_kernel)
    # vis_utils.cv2_show_image('dilate', depth_map)
    # cv2.waitKey()

    # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)
    # vis_utils.cv2_show_image('Hole closing', depth_map)
    # cv2.waitKey()


    # Fill empty spaces with dilated values
    empty_pixels = (depth_map < 0.1)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # vis_utils.cv2_show_image('fill empty pixels', depth_map)
    # cv2.waitKey()

    # Extend highest pixel to top of image
    if extrapolate:
        top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = \
                top_pixel_values[pixel_col_idx]

        # Large Fill
        empty_pixels = depth_map < 0.1
        dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
        depth_map[empty_pixels] = dilated[empty_pixels]

    # Median blur
    depth_map = cv2.medianBlur(depth_map, 5)

    # vis_utils.cv2_show_image('median blur', depth_map)
    # cv2.waitKey()

    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        # Bilateral blur
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
    elif blur_type == 'gaussian':
        # Gaussian blur
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]
    
    
    # vis_utils.cv2_show_image('bilateral blur', depth_map)
    # cv2.waitKey()
    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # vis_utils.cv2_show_image('final', depth_map)
    # cv2.waitKey()

    return depth_map


def get_nn_intensity(old_pc, new_points):
    n_neighbors = 2
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(old_pc[:, :3])
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    num_new_pts = new_points.shape[0]
    new_intensities = np.zeros(num_new_pts)

    for i in range(num_new_pts):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(new_points[i], n_neighbors)
        new_intensities[i] = np.mean(old_pc[idx][:,3], axis=0)

    new_pc = np.hstack((new_points, new_intensities.reshape((-1, 1))))

    return new_pc

def upsample(points, use_points_beam_inc = False):

    """
    Input: points [N, 5] xyzi, channel
    Output: aug_points [N+M, 4] xyzi
    Output: old_new_mask [N+M,]
    """

    if use_points_beam_inc:
        # [B, N]
        xy_norm = np.linalg.norm(points[:, 0:2], axis=-1)
        # [B, N]
        point_inclination = np.arctan2(points[..., 2], xy_norm)
        beam_inclination_min = point_inclination.min()
        beam_inclination_max = point_inclination.max()
        assert points[:,-1].min() == 0.0
        assert points[:,-1].max() == 63.0
    else:
        beam_inclination_min = BEAM_INCLINATION_MIN
        beam_inclination_max = BEAM_INCLINATION_MAX
    
    #compute inclinations
    inclinations = compute_inclinations(beam_inclination_min, beam_inclination_max)

    #V.draw_scenes(points[:,:3])

    #Remove points below ground plane
    #pc, w, h = filter_below_groundplane(points, tolerance=1)
    #V.draw_scenes(pc[:,:3])
    pc = points

    #Apply DROR
    keep_mask = o3d_dynamic_radius_outlier_filter(pc, alpha=0.45)
    snow_indices = (keep_mask == 0).nonzero()[0]#.astype(np.int16)
    range_snow_indices = np.linalg.norm(pc[snow_indices][:,:3], axis=1)
    snow_indices = snow_indices[range_snow_indices < 30]
    keep_indices = np.ones(len(pc), dtype=bool)
    keep_indices[snow_indices] = False
    pc = pc[keep_indices]

    # Build Range image from clean pc
    range_image, _, _ = build_range_image_from_point_cloud(pc[:,:3], inclinations, point_features=None, channel = pc[:, -1])
    # plt.hist(ri_ranges, bins = 20)
    # plt.show()

    # plt.hist((range_image[range_image > 0.1]).reshape((-1)), bins = 20)
    # plt.show()

    range_image = range_image.astype(np.float32)
    ###################### Hole Filling Start ####################################################
    empty_pixels = range_image < 0.1


    #range_image = fill_in_fast(range_image)
    #range_image = fill_based_on_freq(range_image, empty_pixels, mode='max', freq_mode='least', kernel=DIAMOND_KERNEL_5)
    range_image = fill_max(range_image, empty_pixels, kernel=DIAMOND_KERNEL_5)
    ###################### Hole Filling End ####################################################

    #Extract point cloud from range image
    range_image_cartesian = extract_point_cloud_from_range_image(range_image, inclinations)
    new_points = range_image_cartesian[empty_pixels].reshape((-1, 3))
    
    #Remove new points below previously estimated ground plane
    #new_points, _,_ = filter_below_groundplane(new_points, tolerance=0.1,  w=w, h=h)
    new_pt_ranges = np.linalg.norm(new_points, axis=-1)
    new_points = new_points[new_pt_ranges > 0.1]

    #Filter points based on probability
    keep_indices = np.random.choice(np.array([False, True], dtype=bool), size=new_points.shape[0], replace=True, p=[0.7, 0.3])
    new_points = new_points[keep_indices]
    # Get intensity
    new_points = get_nn_intensity(pc, new_points)
    final_points = np.vstack((pc[:,:4], new_points)) # pc has no points below ground plane and is DRORed
    #final_points = np.vstack((pc[:,:3], new_points))
    
    old_new_mask = np.zeros((final_points.shape[0], 1))
    old_new_mask[pc.shape[0]:, 0] = 1

    # range_image_color = cv2.applyColorMap(np.uint8(range_image / np.amax(range_image) * 255),
    #                     cv2.COLORMAP_JET)
    # cv2.namedWindow('range_img', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('range_img', range_image_color)
    # cv2.waitKey() 
    # V.draw_scenes(np.hstack((final_points, old_new_mask)), color_feature=4)
    #V.draw_scenes(np.hstack((final_points, old_new_mask)), color_feature=3)

    return final_points, old_new_mask

    
    

def main():
    lidar_path = '/home/barza/OpenPCDet/data/dense/lidar_hdl64_strongest/2018-02-04_12-13-33_00100.bin'

    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5) #xyzi, channel

    final_points, old_new_mask = upsample(points)
    final_points = np.hstack((final_points, old_new_mask))
    
    #V.draw_scenes(points[:,:3])

    V.draw_scenes(final_points, color_feature=4)
