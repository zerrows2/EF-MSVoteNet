# 第一篇论文备份

# Copyright (c) OpenMMLab. All rights reserved.
import cv2,os
import mmcv
import mmengine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


from tqdm import tqdm
from scipy import io as sio
from PIL import Image
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from scipy import stats
from concurrent import futures as futures
from os import path as osp
from scipy import stats as sp_stats  # 修改此处，使用别名

# 随机采样函数
def random_sampling(points, num_points, replace=None, return_choices=False):
    """Random sampling.

    Sampling point cloud to a certain number of points.

    Args:
        points (ndarray): Point cloud.
        num_points (int): The number of samples.
        replace (bool): Whether the sample is with or without replacement.
        return_choices (bool): Whether to return choices.

    Returns:
        points (ndarray): Point cloud after sampling.
        
    
    Args:
        points (ndarray): 输入的点云数据。
        num_points (int): 想要采样的点数。
        replace (bool): 是否允许重复采样。
        return_choices (bool): 是否返回采样的索引。
    Returns:
        ndarray: 采样后的点云数据。
    
    """

    if replace is None:
        replace = (points.shape[0] < num_points)
    choices = np.random.choice(points.shape[0], num_points, replace=replace)
    if return_choices:
        return points[choices], choices
    else:
        return points[choices]
     

# 复位坐标
def flip_axis_reset_camera(pc):
    ''' Flip X-right,-Z-up,Y-forward back to X-right,Y-up,Z-forward
        Input and output are both (N,3) arrays
    '''
    pc2 = np.copy(pc)
    pc2[:, [0, 1, 2]] = pc2[:, [0, 2, 1]]  # depth X,Y,Z = cam X,-Z,Y
    pc2[:, 2] *= -1  # Invert the Z-axis
    return pc2



# 设置视锥体，读取真实3D边框，真实情况下不可能实现
def frustum_sampling_TRUE(points, calib, info_annos, rgbimage):
    
    # print("true")

    """
    # 视锥体算法
    """
    # 将点云转换到摄像机坐标系
    # print(f"################################################### points 中原本的点云数量: {points.shape[0]}")    

    pc_upright_camera = np.zeros_like(points)
    pc_upright_camera[:, 0:3] = calib.project_upright_depth_to_upright_camera(points[:, 0:3])
    pc_upright_camera[:, 3:] = points[:, 3:]

    # 定义一个布尔数组，初始时假定所有点都不在任何3D边界框内
    points_in_boxes = np.zeros(len(points), dtype=bool)

    for ann_idx in range(info_annos['gt_num']):
        # 获取单个对象的3D边界框
        gt_box_upright_depth = info_annos['gt_boxes_upright_depth'][ann_idx]

        for i, point in enumerate(points):
            if is_point_in_box3d(point[:3], gt_box_upright_depth):
                points_in_boxes[i] = True

    # 筛选出位于3D边界框内的点云
    pc_in_boxes = pc_upright_camera[points_in_boxes, :]

    # 复位坐标   
    reset_pc_in_boxes = flip_axis_reset_camera(pc_in_boxes)
    # 在此处执行其他操作，如降采样等
    # print(f"################################################### reset_pc_in_boxes 中真实的点云数量: {reset_pc_in_boxes.shape[0]}")    

    return reset_pc_in_boxes
    # return mask_frustum_sampling(reset_pc_in_boxes, calib, info_annos, seglabel, rgbimage) 
    # return rgbPP(reset_pc_in_boxes, calib, info_annos, depthimage, rgbimage, seglabel)


# rgbPP
def rgbPP(points, calib, rgbimage, sample_idx):

    # 将深度图像转换为2维数组
    # depth_array = np.array(depthimage)
    # print(depthimage)

    # 保存为txt文件
    # np.savetxt('/root/autodl-tmp/depthimage.txt', depth_array, fmt='%f')

    print(f"################################################### rgbPP 初始的点云数量: {points.shape[0]}")    

    original_rgbimage_pc_image_coord = rgbimage.copy()

    original_rgbimage_pc_image_coord.fill(255)

    """
        视锥体算法
    """
    # 降采样
    # points = random_sampling(points, 50000)
    
    # 将点云转换到摄像机坐标系
    pc_upright_depth = points
    # 创建一个与 pc_upright_depth 形状相同但所有元素为零的数组
    pc_upright_camera = np.zeros_like(pc_upright_depth)
    # 前三列 cam X,Y,Z = depth X,-Z,Y
    pc_upright_camera[:, 0:3] = calib.project_upright_depth_to_upright_camera(pc_upright_depth[:, 0:3])
    # 后三列保持原样
    pc_upright_camera[:, 3:] = pc_upright_depth[:, 3:]
    
    # 将点云数据投影到图像坐标系
    pc_image_coord, pc_depth = calib.project_upright_depth_to_image(pc_upright_depth) 
    
    # depth mask
    pc_depth_reshaped = pc_depth.reshape(-1, 1)

    # 拼接图像坐标、深度值和原始点云的后三列
    # pc_newxyz = np.concatenate((pc_image_coord, pc_depth_reshaped), axis=1)
    pc_newxyz = np.concatenate((pc_image_coord, pc_depth_reshaped, pc_upright_camera[:, 3:]), axis=1)

    # 1. 四舍五入x和y坐标
    pc_newxyz[:, 0:2] = np.round(pc_newxyz[:, 0:2])


    # 2.1 对于具有相同整数坐标的点，计算RGB的平均值
    # 创建一个字典，以(x, y)坐标为键，以所有对应点的RGB值和数量为值
    rgb_sum_dict = {}
    count_dict = {}

    for x, y, _, r, g, b in pc_newxyz:

        r = int(round(r * 255))
        g = int(round(g * 255))
        b = int(round(b * 255))
        # depth = int(round(depth * 255))
        # gabor = int(round(gabor * 64000))


        key = (int(x), int(y))
        if key not in rgb_sum_dict:
            rgb_sum_dict[key] = [r, g, b]
            count_dict[key] = 1
        else:
            rgb_sum_dict[key][0] += r
            rgb_sum_dict[key][1] += g
            rgb_sum_dict[key][2] += b
            count_dict[key] += 1

    # 计算平均值
    avg_rgb_dict = {key: [value[0] / count_dict[key], value[1] / count_dict[key], value[2] / count_dict[key]]
                    for key, value in rgb_sum_dict.items()}


    # 2.2 对于具有相同整数坐标的点，保留距离最近的点
    # 创建一个字典，以(x, y)坐标为键，以(depth, r, g, b)为值
    unique_points = {}
    for x, y, depth, r, g, b in pc_newxyz:
        key = (int(x), int(y))
        if key not in unique_points or depth < unique_points[key][0]:
            unique_points[key] = (depth, r, g, b)

    # 3. 将RGB平均值映射到original_rgbimage_pc_image_coord中
    for (x, y), rgb in avg_rgb_dict.items():
        if 0 <= x < original_rgbimage_pc_image_coord.shape[1] and 0 <= y < original_rgbimage_pc_image_coord.shape[0]:
            original_rgbimage_pc_image_coord[y, x] = rgb


    # 生成保存的文件路径，使用sample_idx进行命名
    save_path = f'/root/autodl-tmp/{sample_idx}_pp.jpg'
    
    # 将图像保存为jpg文件
    Image.fromarray(original_rgbimage_pc_image_coord.astype(np.uint8)).save(save_path)
    print(f"映射后的图像已保存到: {save_path}")
        
        

    return points
    # return PP2(reset_pc_in_boxes, calib, rgbimage, original_rgbimage2)
    # return mask_frustum_sampling(reset_pc_in_boxes, calib, info_annos, seglabel, rgbimage)


# rgbPP_NOWC_TEST
def rgbPP_NOWC_TEST(pc_newxyz2, calib, rgbimage):




    original_rgbimage_pc_image_coord = rgbimage.copy()

    original_rgbimage_pc_image_coord.fill(255)

    # 将点云数据投影到图像坐标系
    pc_newxyz = calib.project_upright_depth_to_image_NOWC_TEST(pc_newxyz2) 

    # 1. 四舍五入x和y坐标
    pc_newxyz[:, 0:2] = np.round(pc_newxyz[:, 0:2])


    # 2.1 对于具有相同整数坐标的点，计算RGB的平均值
    # 创建一个字典，以(x, y)坐标为键，以所有对应点的RGB值和数量为值
    rgb_sum_dict = {}
    count_dict = {}

    for x, y, _, r, g, b in pc_newxyz:

        r = int(round(r * 255))
        g = int(round(g * 255))
        b = int(round(b * 255))
        # depth = int(round(depth * 255))
        # gabor = int(round(gabor * 64000))


        key = (int(x), int(y))
        if key not in rgb_sum_dict:
            rgb_sum_dict[key] = [r, g, b]
            count_dict[key] = 1
        else:
            rgb_sum_dict[key][0] += r
            rgb_sum_dict[key][1] += g
            rgb_sum_dict[key][2] += b
            count_dict[key] += 1

    # 计算平均值
    avg_rgb_dict = {key: [value[0] / count_dict[key], value[1] / count_dict[key], value[2] / count_dict[key]]
                    for key, value in rgb_sum_dict.items()}


    # 2.2 对于具有相同整数坐标的点，保留距离最近的点
    # 创建一个字典，以(x, y)坐标为键，以(depth, r, g, b)为值
    unique_points = {}
    for x, y, depth, r, g, b in pc_newxyz:
        key = (int(x), int(y))
        if key not in unique_points or depth < unique_points[key][0]:
            unique_points[key] = (depth, r, g, b)

    # 3. 将RGB平均值映射到original_rgbimage_pc_image_coord中
    for (x, y), rgb in avg_rgb_dict.items():
        if 0 <= x < original_rgbimage_pc_image_coord.shape[1] and 0 <= y < original_rgbimage_pc_image_coord.shape[0]:
            original_rgbimage_pc_image_coord[y, x] = rgb

    # 显示或保存映射后的图像
    # 使用matplotlib或OpenCV等库进行显示或保存
    Image.fromarray(original_rgbimage_pc_image_coord).save('/root/autodl-tmp/pp_NOWC_TEST.jpg')
    
        

    return pc_newxyz2
    # return PP2(reset_pc_in_boxes, calib, rgbimage, original_rgbimage2)
    # return mask_frustum_sampling(reset_pc_in_boxes, calib, info_annos, seglabel, rgbimage)


# rgbPP备份
def rgbPPB(points, calib, rgbimage):

    # 将深度图像转换为2维数组
    # depth_array = np.array(depthimage)
    # print(depthimage)

    # 保存为txt文件
    # np.savetxt('/root/autodl-tmp/depthimage.txt', depth_array, fmt='%f')

    print(f"################################################### rgbPP 初始的点云数量: {points.shape[0]}")    

    original_rgbimage_pc_image_coord = rgbimage.copy()

    original_rgbimage_pc_image_coord.fill(255)

    """
        视锥体算法
    """
    # 降采样
    # points = random_sampling(points, 50000)
    
    # 将点云转换到摄像机坐标系
    pc_upright_depth = points
    # 创建一个与 pc_upright_depth 形状相同但所有元素为零的数组
    pc_upright_camera = np.zeros_like(pc_upright_depth)
    # 前三列 cam X,Y,Z = depth X,-Z,Y
    pc_upright_camera[:, 0:3] = calib.project_upright_depth_to_upright_camera(pc_upright_depth[:, 0:3])
    # 后三列保持原样
    pc_upright_camera[:, 3:] = pc_upright_depth[:, 3:]
    
    # 将点云数据投影到图像坐标系
    pc_image_coord, pc_depth = calib.project_upright_depth_to_image(pc_upright_depth) 
    
    # depth mask
    pc_depth_reshaped = pc_depth.reshape(-1, 1)

    # 拼接图像坐标、深度值和原始点云的后三列
    # pc_newxyz = np.concatenate((pc_image_coord, pc_depth_reshaped), axis=1)
    pc_newxyz = np.concatenate((pc_image_coord, pc_depth_reshaped, pc_upright_camera[:, 3:]), axis=1)

    # 1. 四舍五入x和y坐标
    pc_newxyz[:, 0:2] = np.round(pc_newxyz[:, 0:2])


    # 2.1 对于具有相同整数坐标的点，计算RGB的平均值
    # 创建一个字典，以(x, y)坐标为键，以所有对应点的RGB值和数量为值
    rgb_sum_dict = {}
    count_dict = {}

    for x, y, _, r, g, b in pc_newxyz:

        r = int(round(r * 255))
        g = int(round(g * 255))
        b = int(round(b * 255))
        # depth = int(round(depth * 255))
        # gabor = int(round(gabor * 64000))


        key = (int(x), int(y))
        if key not in rgb_sum_dict:
            rgb_sum_dict[key] = [r, g, b]
            count_dict[key] = 1
        else:
            rgb_sum_dict[key][0] += r
            rgb_sum_dict[key][1] += g
            rgb_sum_dict[key][2] += b
            count_dict[key] += 1

    # 计算平均值
    avg_rgb_dict = {key: [value[0] / count_dict[key], value[1] / count_dict[key], value[2] / count_dict[key]]
                    for key, value in rgb_sum_dict.items()}


    # 2.2 对于具有相同整数坐标的点，保留距离最近的点
    # 创建一个字典，以(x, y)坐标为键，以(depth, r, g, b)为值
    unique_points = {}
    for x, y, depth, r, g, b in pc_newxyz:
        key = (int(x), int(y))
        if key not in unique_points or depth < unique_points[key][0]:
            unique_points[key] = (depth, r, g, b)

    # 3. 将RGB平均值映射到original_rgbimage_pc_image_coord中
    for (x, y), rgb in avg_rgb_dict.items():
        if 0 <= x < original_rgbimage_pc_image_coord.shape[1] and 0 <= y < original_rgbimage_pc_image_coord.shape[0]:
            original_rgbimage_pc_image_coord[y, x] = rgb

    # 显示或保存映射后的图像
    # 使用matplotlib或OpenCV等库进行显示或保存
    Image.fromarray(original_rgbimage_pc_image_coord).save('/root/autodl-tmp/ppB.jpg')
    
        

    return points
    # return PP2(reset_pc_in_boxes, calib, rgbimage, original_rgbimage2)
    # return mask_frustum_sampling(reset_pc_in_boxes, calib, info_annos, seglabel, rgbimage)


# rgbPP备份
def rgbPPB2(points, calib, rgbimage):

    # 将深度图像转换为2维数组
    # depth_array = np.array(depthimage)
    # print(depthimage)

    # 保存为txt文件
    # np.savetxt('/root/autodl-tmp/depthimage.txt', depth_array, fmt='%f')

    print(f"################################################### rgbPPB 初始的点云数量: {points.shape[0]}")    

    original_rgbimage_pc_image_coord = rgbimage.copy()

    original_rgbimage_pc_image_coord.fill(255)

    original_rgbimage = rgbimage.copy()

    # 初始化rgbimage为二值图像
    original_rgbimage.fill(255)

    """
        视锥体算法
    """
    # 降采样
    # points = random_sampling(points, 50000)
    
    # 视角转换测试 将点云的前三维度从 (x, y, z) 变为 (x, z, -y) 正视图
    # 失败
    # points[:, :3] = points[:, [0, 2, 1]]
    # points[:, 2] = -points[:, 2]

    # 将点云转换到摄像机坐标系
    pc_upright_depth = points
    # 创建一个与 pc_upright_depth 形状相同但所有元素为零的数组
    pc_upright_camera = np.zeros_like(pc_upright_depth)

    # 前三列 cam X,Y,Z = depth X,-Z,Y
    # pc_upright_camera[:, 0:3] = calib.project_upright_depth_to_upright_camera(pc_upright_depth[:, 0:3])

    # calib.project_upright_depth_to_upright_camera 逻辑：
    # pc_upright_camera2 = np.copy(pc_upright_camera)
    # pc_upright_camera2[:, [0, 1, 2]] = pc_upright_camera2[:, [0, 2, 1]]  # cam X,Y,Z = depth X,-Z,Y
    # pc_upright_camera2[:, 1] *= -1
    # pc_upright_camera = pc_upright_camera2

    # 后三列保持原样
    pc_upright_camera[:, 3:] = pc_upright_depth[:, 3:]
    
    # 将点云数据投影到图像坐标系
    pc_image_coord, pc_depth = calib.project_upright_depth_to_image(pc_upright_depth) 
    
    # depth mask
    pc_depth_reshaped = pc_depth.reshape(-1, 1)
    # 拼接图像坐标、深度值和原始点云的后三列
    # pc_newxyz = np.concatenate((pc_image_coord, pc_depth_reshaped), axis=1)
    pc_newxyz = np.concatenate((pc_image_coord, pc_depth_reshaped, pc_upright_camera[:, 3:]), axis=1)

    # 1. 四舍五入x和y坐标
    pc_newxyz[:, 0:2] = np.round(pc_newxyz[:, 0:2])


    # 2.1 对于具有相同整数坐标的点，计算RGB的平均值
    # 创建一个字典，以(x, y)坐标为键，以所有对应点的RGB值和数量为值
    rgb_sum_dict = {}
    count_dict = {}

    for x, y, _, r, g, b in pc_newxyz:

        r = int(round(r * 255))
        g = int(round(g * 255))
        b = int(round(b * 255))
        # depth = int(round(depth * 255))
        # gabor = int(round(gabor * 64000))


        key = (int(x), int(y))
        if key not in rgb_sum_dict:
            rgb_sum_dict[key] = [r, g, b]
            count_dict[key] = 1
        else:
            rgb_sum_dict[key][0] += r
            rgb_sum_dict[key][1] += g
            rgb_sum_dict[key][2] += b
            count_dict[key] += 1

    # 计算平均值
    avg_rgb_dict = {key: [value[0] / count_dict[key], value[1] / count_dict[key], value[2] / count_dict[key]]
                    for key, value in rgb_sum_dict.items()}


    # 2.2 对于具有相同整数坐标的点，保留距离最近的点
    # 创建一个字典，以(x, y)坐标为键，以(depth, r, g, b)为值
    unique_points = {}
    for x, y, depth, r, g, b in pc_newxyz:
        key = (int(x), int(y))
        if key not in unique_points or depth < unique_points[key][0]:
            unique_points[key] = (depth, r, g, b)

    # 3. 将RGB平均值映射到original_rgbimage_pc_image_coord中
    for (x, y), rgb in avg_rgb_dict.items():
        if 0 <= x < original_rgbimage_pc_image_coord.shape[1] and 0 <= y < original_rgbimage_pc_image_coord.shape[0]:
            original_rgbimage_pc_image_coord[y, x] = rgb

    # 显示或保存映射后的图像
    # 使用matplotlib或OpenCV等库进行显示或保存
    Image.fromarray(original_rgbimage_pc_image_coord).save('/root/autodl-tmp/ppB.jpg')
    
        

    return points
    # return PP2(reset_pc_in_boxes, calib, rgbimage, original_rgbimage2)
    # return mask_frustum_sampling(reset_pc_in_boxes, calib, info_annos, seglabel, rgbimage)


# 下================================================================================================================

# 将指定区域内的True数量增加到原来的三倍
def n_true_in_region(bool_array, indices, n):
    """将指定区域内的True数量增加到原来的n倍"""
    # 获取当前区域中为True的索引
    true_indices = indices[bool_array[indices]]
    num_true = len(true_indices)
    
    # 需要增加的True数量是当前True数量的n-1倍
    num_to_add = (n-1) * num_true
    
    # 获取当前区域中为False的索引
    false_indices = indices[~bool_array[indices]]
    
    # 如果需要增加的True数量超过了可用的False数量，则只增加到最大可能的数量
    num_to_add = min(num_to_add, len(false_indices))
    
    # 从False位置随机选取足够的位置转变为True
    if num_to_add > 0:
        add_indices = np.random.choice(false_indices, num_to_add, replace=False)
        bool_array[add_indices] = True

# 将指定区域内的True数量增加到原来的三倍
def triple_true_in_region(bool_array, indices):
    """将指定区域内的True数量增加到原来的三倍"""
    # 获取当前区域中为True的索引
    true_indices = indices[bool_array[indices]]
    num_true = len(true_indices)
    
    # 需要增加的True数量是当前True数量的两倍
    num_to_add = 2 * num_true
    
    # 获取当前区域中为False的索引
    false_indices = indices[~bool_array[indices]]
    
    # 如果需要增加的True数量超过了可用的False数量，则只增加到最大可能的数量
    num_to_add = min(num_to_add, len(false_indices))
    
    # 从False位置随机选取足够的位置转变为True
    if num_to_add > 0:
        add_indices = np.random.choice(false_indices, num_to_add, replace=False)
        bool_array[add_indices] = True

# 将指定区域内的True数量翻倍
def double_true_in_region(bool_array, indices):
    """将指定区域内的True数量翻倍"""
    true_indices = indices[bool_array[indices]]
    num_true = len(true_indices)
    if num_true > 0:
        # 需要翻倍的True数量，但不能超过索引区域的剩余空间
        num_to_add = min(num_true, len(indices) - num_true)
        add_indices = np.random.choice(indices[~bool_array[indices]], num_to_add, replace=False)
        bool_array[add_indices] = True

# 设置为2D标注框为先验条件，TRUE
def frustum_sampling_train_True(points, calib, info_annos, rgbimage):

    # 将点云转换到摄像机坐标系
    pc_upright_depth = points
    # 创建一个与 pc_upright_depth 形状相同但所有元素为零的数组
    pc_upright_camera = np.zeros_like(pc_upright_depth)
    # 前三列 cam X,Y,Z = depth X,-Z,Y
    pc_upright_camera[:, 0:3] = calib.project_upright_depth_to_upright_camera(pc_upright_depth[:, 0:3])
    # 后三列保持原样
    pc_upright_camera[:, 3:] = pc_upright_depth[:, 3:]
    
    # 将点云数据投影到图像坐标系
    pc_image_coord, pc_depth = calib.project_upright_depth_to_image(pc_upright_depth) 
    
    # depth mask
    pc_depth_reshaped = pc_depth.reshape(-1, 1)
    
    # 定义一个布尔数组，初始时假定所有点都不在任何box2d框内
    points_in_boxes = np.zeros(len(pc_image_coord), dtype=bool)

    for ann_idx in range(info_annos['gt_num']):
        
        # 获取单个对象的2D边界框
        bbox = info_annos['bbox'][ann_idx]

        # 提取2D边界框坐标
        box2d = bbox
        xmin, ymin, xmax, ymax = box2d
        
        box_fov_inds = (pc_image_coord[:, 0] < xmax) & (pc_image_coord[:, 0] >= xmin) & \
                    (pc_image_coord[:, 1] < ymax) & (pc_image_coord[:, 1] >= ymin)
        
        # 更新 points_in_boxes，如果点在当前框内，则将相应位置设为True
        points_in_boxes = points_in_boxes | box_fov_inds

    # 未统计到点云，则保留原始数据
    if not points_in_boxes.any():
        # print("furtusm_null")
        # 将所有元素设置为 True
        points_in_boxes[:] = True
        
        # 2维？
        coord_in_boxes = pc_image_coord[points_in_boxes, :]
        # 3维？
        pc_in_boxes = pc_upright_camera[points_in_boxes, :]
        
        # print("len 11111111: ",pc_in_boxes.shape[0])
        
        # 降采样
        reset_pc_in_boxes = random_sampling(pc_in_boxes, 50000)
        
        # print("未统计到点云，则保留原始数据")
        # print("1111111111_pc_upright_depth_len: ",len(pc_upright_depth))
        # print("2222222222_box_fov_inds_mask_len: ",len(reset_pc_in_boxes))

    else:    
        # 2维？
        coord_in_boxes = pc_image_coord[points_in_boxes, :]
        # 3维？
        pc_in_boxes = pc_upright_camera[points_in_boxes, :]

    # 复位坐标   
    reset_pc_in_boxes = flip_axis_reset_camera(pc_in_boxes)

    return reset_pc_in_boxes

    # 返回只保留2D检测框内的点云
   #  return rgbPP(reset_pc_in_boxes, calib, info_annos, rgbimage)


# 设置为2D标注框为先验条件，不重复
def frustum_sampling_train(points, calib, info_annos, rgbimage):
        
    # print(f"################################################### points 中的点云数量: {points.shape[0]}")    
    # print(calib)
    # print(info_annos)
    """
        视锥体算法
    """
    # 降采样
    # points = random_sampling(points, 50000)
    
    # 将点云转换到摄像机坐标系
    pc_upright_depth = points
    # 创建一个与 pc_upright_depth 形状相同但所有元素为零的数组
    pc_upright_camera = np.zeros_like(pc_upright_depth)
    # 前三列 cam X,Y,Z = depth X,-Z,Y
    pc_upright_camera[:, 0:3] = calib.project_upright_depth_to_upright_camera(pc_upright_depth[:, 0:3])
    # 后三列保持原样
    pc_upright_camera[:, 3:] = pc_upright_depth[:, 3:]
    
    # 将点云数据投影到图像坐标系
    pc_image_coord, pc_depth = calib.project_upright_depth_to_image(pc_upright_depth) 
    
    # depth mask
    pc_depth_reshaped = pc_depth.reshape(-1, 1)
    # pc_newxyz = np.concatenate((pc_image_coord, pc_depth_reshaped), axis=1)
    
    # 定义一个布尔数组，初始时假定所有点都不在任何box2d框内
    points_in_boxes = np.zeros(len(pc_image_coord), dtype=bool)

    points_in_boxes_index = np.zeros(len(pc_image_coord), dtype=bool)

    # 确保至少有50000点被选中
    true_count = np.sum(points_in_boxes)
    needed_count = 50000 - true_count
    if needed_count > 0:
        false_indices = np.where(~points_in_boxes)[0]
        selected_indices = np.random.choice(false_indices, min(needed_count, len(false_indices)), replace=False)
        points_in_boxes[selected_indices] = True
    
    for ann_idx in range(info_annos['gt_num']):
        
        # 获取单个对象的2D边界框
        bbox = info_annos['bbox'][ann_idx]

        # 获取单个对象的3D边界框
        # gt_box_upright_depth = info_annos['gt_boxes_upright_depth'][ann_idx]
        

        # 提取2D边界框坐标
        box2d = bbox
        # print("box2d name :", object_name)
        # print("box2d size :", box2d)

        xmin, ymin, xmax, ymax = box2d
        
        box_fov_inds = (pc_image_coord[:, 0] < xmax) & (pc_image_coord[:, 0] >= xmin) & \
                    (pc_image_coord[:, 1] < ymax) & (pc_image_coord[:, 1] >= ymin)
        
        # points_in_boxes = points_in_boxes | box_fov_inds
        points_in_boxes_index = points_in_boxes_index | box_fov_inds

    # 更新 points_in_boxes，如果点在当前框内，则将相应位置设为True
    # indices_in_box = np.where(box_fov_inds)[0]
    indices_in_box = np.where(points_in_boxes_index)[0]

    # 调用辅助函数以翻倍区域内True的数量
    n_true_in_region(points_in_boxes, indices_in_box, 10)

    coord_in_boxes = pc_image_coord[points_in_boxes, :]
    pc_in_boxes = pc_upright_camera[points_in_boxes, :]

    # 复位坐标   
    reset_pc_in_boxes = flip_axis_reset_camera(pc_in_boxes)
        
    # return mask_frustum_sampling(reset_pc_in_boxes, calib, info_annos, seglabel)

    # print(f"################################################### frustum_in_boxes 中的点云数量: {reset_pc_in_boxes.shape[0]}")  
    # reset_pc_in_boxes = random_sampling(reset_pc_in_boxes, 150000)

    # return reset_pc_in_boxes
    # return mask_frustum_sampling(reset_pc_in_boxes, calib, info_annos, seglabel, rgbimage)
    # return mask_frustum_sampling(reset_pc_in_boxes, calib, info_annos, depthimage, seglabel, rgbimage)
    return rgbPP(reset_pc_in_boxes, calib, info_annos, rgbimage)

# 设置正常视锥体备_val，初始5w个点，然后将2D框内的点增加多倍，重复
def frustum_sampling_valB(points, calib, bbox_list, info_annos, rgbimage, sample_idx):
    pc_upright_depth = points
    pc_upright_camera = np.zeros_like(pc_upright_depth)
    pc_upright_camera[:, 0:3] = calib.project_upright_depth_to_upright_camera(pc_upright_depth[:, 0:3])
    pc_upright_camera[:, 3:] = pc_upright_depth[:, 3:]

    pc_image_coord, pc_depth = calib.project_upright_depth_to_image(pc_upright_depth)

    points_in_boxes = np.zeros(len(pc_image_coord), dtype=bool)

    points_in_boxes_index = np.zeros(len(pc_image_coord), dtype=bool)

    # 确保至少有100000点被选中
    true_count = np.sum(points_in_boxes)
    needed_count = 100000 - true_count

    if needed_count > 0:
        false_indices = np.where(~points_in_boxes)[0]
        selected_indices = np.random.choice(false_indices, min(needed_count, len(false_indices)), replace=False)
        points_in_boxes[selected_indices] = True

    for bbox_info in bbox_list:
        bbox = bbox_info['bbox']
        confidence = float(bbox_info['confidence'])

        xmin, ymin, xmax, ymax = bbox
        box_fov_inds = (pc_image_coord[:, 0] < xmax) & (pc_image_coord[:, 0] >= xmin) & \
                       (pc_image_coord[:, 1] < ymax) & (pc_image_coord[:, 1] >= ymin)

        points_in_boxes_index = points_in_boxes_index | box_fov_inds
        # points_in_boxes = points_in_boxes | box_fov_inds

    # 如果 points_in_boxes_index 为空，则随机将其中 100 个点设置为 True
    if not np.any(points_in_boxes_index):
        random_indices = np.random.choice(len(points_in_boxes_index), 100, replace=False)
        points_in_boxes_index[random_indices] = True


    pc_in_boxes = pc_upright_camera[points_in_boxes_index, :]

    reset_pc_in_boxes = flip_axis_reset_camera(pc_in_boxes)
    
    # reset_pc_in_boxes = random_sampling(reset_pc_in_boxes, 50000)

    return reset_pc_in_boxes
    # return rgbPP(reset_pc_in_boxes, calib, rgbimage, sample_idx)  



# 设置正常视锥体备_val 备份，初始5w个点，然后将2D框内的点增加多倍，重复
def frustum_sampling_val(points, calib, bbox_list, info_annos, rgbimage, sample_idx):
    pc_upright_depth = points
    pc_upright_camera = np.zeros_like(pc_upright_depth)
    pc_upright_camera[:, 0:3] = calib.project_upright_depth_to_upright_camera(pc_upright_depth[:, 0:3])
    pc_upright_camera[:, 3:] = pc_upright_depth[:, 3:]

    pc_image_coord, pc_depth = calib.project_upright_depth_to_image(pc_upright_depth)

    points_in_boxes_index = np.zeros(len(pc_image_coord), dtype=bool)


    for bbox_info in bbox_list:
        bbox = bbox_info['bbox']
        # confidence = float(bbox_info['confidence'])

        xmin, ymin, xmax, ymax = bbox
        box_fov_inds = (pc_image_coord[:, 0] < xmax) & (pc_image_coord[:, 0] >= xmin) & \
                       (pc_image_coord[:, 1] < ymax) & (pc_image_coord[:, 1] >= ymin)

        points_in_boxes_index = points_in_boxes_index | box_fov_inds
        # points_in_boxes = points_in_boxes | box_fov_inds

    # 判断 points_in_boxes_index 是否全为假
    if not points_in_boxes_index.any():
        # 如果全为假，则随机选择 100,000 个点并设置为真
        random_indices = np.random.choice(len(points_in_boxes_index), 100000, replace=False)
        points_in_boxes_index[random_indices] = True
    else:
        # 如果不是全为假，则对假的部分随机选择 10% 设置为真
        false_indices = np.where(~points_in_boxes_index)[0]
        num_to_set_true = int(len(false_indices) * 0.7)  # 10%
        if num_to_set_true > 0:
            random_false_indices = np.random.choice(false_indices, num_to_set_true, replace=False)
            points_in_boxes_index[random_false_indices] = True

    pc_in_boxes = pc_upright_camera[points_in_boxes_index, :]

    reset_pc_in_boxes = flip_axis_reset_camera(pc_in_boxes)
    
    # reset_pc_in_boxes = random_sampling(reset_pc_in_boxes, 50000)

    return reset_pc_in_boxes
    # return rgbPP(reset_pc_in_boxes, calib, rgbimage, sample_idx)  



# 设置正常视锥体备_val 备份，初始5w个点，然后将2D框内的点增加多倍，重复
def frustum_sampling_valB(points, calib, bbox_list, info_annos, rgbimage, sample_idx):
    pc_upright_depth = points
    pc_upright_camera = np.zeros_like(pc_upright_depth)
    pc_upright_camera[:, 0:3] = calib.project_upright_depth_to_upright_camera(pc_upright_depth[:, 0:3])
    pc_upright_camera[:, 3:] = pc_upright_depth[:, 3:]

    pc_image_coord, pc_depth = calib.project_upright_depth_to_image(pc_upright_depth)

    points_in_boxes = np.zeros(len(pc_image_coord), dtype=bool)

    points_in_boxes_index = np.zeros(len(pc_image_coord), dtype=bool)

    # 确保至少有100000点被选中
    true_count = np.sum(points_in_boxes)
    needed_count = 100000 - true_count

    if needed_count > 0:
        false_indices = np.where(~points_in_boxes)[0]
        selected_indices = np.random.choice(false_indices, min(needed_count, len(false_indices)), replace=False)
        points_in_boxes[selected_indices] = True

    for bbox_info in bbox_list:
        bbox = bbox_info['bbox']
        confidence = float(bbox_info['confidence'])

        xmin, ymin, xmax, ymax = bbox
        box_fov_inds = (pc_image_coord[:, 0] < xmax) & (pc_image_coord[:, 0] >= xmin) & \
                       (pc_image_coord[:, 1] < ymax) & (pc_image_coord[:, 1] >= ymin)

        points_in_boxes_index = points_in_boxes_index | box_fov_inds
        # points_in_boxes = points_in_boxes | box_fov_inds

    indices_in_box = np.where(points_in_boxes_index)[0]

    # 调用辅助函数以翻倍区域内True的数量
    n_true_in_region(points_in_boxes, indices_in_box, 100)
    # double_true_in_region(points_in_boxes, indices_in_box)
    # triple_true_in_region(points_in_boxes, indices_in_box)
    
    coord_in_boxes = pc_image_coord[points_in_boxes, :]
    pc_in_boxes = pc_upright_camera[points_in_boxes, :]

    reset_pc_in_boxes = flip_axis_reset_camera(pc_in_boxes)
    
    # reset_pc_in_boxes = random_sampling(reset_pc_in_boxes, 50000)

    # return reset_pc_in_boxes
    return rgbPP(reset_pc_in_boxes, calib, rgbimage, sample_idx)  



# 设置正常视锥体备_val，初始5w个点，然后按照置信度保留2D框内的点
def frustum_sampling_val3(points, calib, bbox_list, info_annos, depthimage, rgbimage, seglabel):
        
    # print(f"################################################### points 中的点云数量: {points.shape[0]}")    
    # print(bbox_list)
    # print(info_annos)
    """
        视锥体算法
    """
    # 降采样
    # points = random_sampling(points, 50000)
    
    # 将点云转换到摄像机坐标系
    pc_upright_depth = points
    # 创建一个与 pc_upright_depth 形状相同但所有元素为零的数组
    pc_upright_camera = np.zeros_like(pc_upright_depth)
    # 前三列 cam X,Y,Z = depth X,-Z,Y
    pc_upright_camera[:, 0:3] = calib.project_upright_depth_to_upright_camera(pc_upright_depth[:, 0:3])
    # 后三列保持原样
    pc_upright_camera[:, 3:] = pc_upright_depth[:, 3:]
    
    # 将点云数据投影到图像坐标系
    pc_image_coord, pc_depth = calib.project_upright_depth_to_image(pc_upright_depth) 
    
    # depth mask
    pc_depth_reshaped = pc_depth.reshape(-1, 1)
    # pc_newxyz = np.concatenate((pc_image_coord, pc_depth_reshaped), axis=1)
    
    # 定义一个布尔数组，初始时假定所有点都不在任何box2d框内
    points_in_boxes = np.zeros(len(pc_image_coord), dtype=bool)

    # 确保至少有50000点被选中
    true_count = np.sum(points_in_boxes)
    needed_count = 50000 - true_count
    if needed_count > 0:
        false_indices = np.where(~points_in_boxes)[0]
        selected_indices = np.random.choice(false_indices, min(needed_count, len(false_indices)), replace=False)
        points_in_boxes[selected_indices] = True

    # bbox_lists格式'class_name': class_name, 'bbox': bbox, 'confidence': confidence
    for bbox_info in bbox_list:

        # 获取单个对象的2D边界框
        bbox = bbox_info['bbox']
        
        # 获取置信度
        confidence = float(bbox_info['confidence'])

        # print("Bounding Box:", bbox)
        # 获取单个对象的3D边界框
        # gt_box_upright_depth = info_annos['gt_boxes_upright_depth'][ann_idx]

        # 提取2D边界框坐标
        box2d = bbox
        # print("box2d name :", object_name)
        # print("box2d size :", box2d)

        xmin, ymin, xmax, ymax = box2d
        
        box_fov_inds = (pc_image_coord[:, 0] < xmax) & (pc_image_coord[:, 0] >= xmin) & \
                    (pc_image_coord[:, 1] < ymax) & (pc_image_coord[:, 1] >= ymin)
        
        # 找出所有在框内的点的索引
        indices_in_box = np.where(box_fov_inds)[0]

        # 根据置信度决定要降采样的点数
        num_points_to_discard = int(len(indices_in_box) * (1 - confidence))  # 基于confidence动态计算要丢弃的点数

        # 丢弃点不应使剩余点数少于100
        if len(indices_in_box) - num_points_to_discard < 100:
            num_points_to_discard = len(indices_in_box) - 100
            
        # 如果可删除的点数变成负数，则不删除任何点
        if num_points_to_discard < 0:
            num_points_to_discard = 0
        
        # 随机选择要丢弃的点
        discard_indices = np.random.choice(indices_in_box, num_points_to_discard, replace=False)
        
        # 设置选中的点为False
        box_fov_inds[discard_indices] = False

        # 更新 points_in_boxes，如果点在当前框内，则将相应位置设为True
        points_in_boxes = points_in_boxes | box_fov_inds

    
    # 2维？
    coord_in_boxes = pc_image_coord[points_in_boxes, :]
    # 3维？
    pc_in_boxes = pc_upright_camera[points_in_boxes, :]

    # 复位坐标   
    reset_pc_in_boxes = flip_axis_reset_camera(pc_in_boxes)
        
    # return mask_frustum_sampling(reset_pc_in_boxes, calib, info_annos, seglabel)

    # print(f"################################################### frustum_in_boxes 中的点云数量: {reset_pc_in_boxes.shape[0]}")  

    

    return reset_pc_in_boxes

    # return mask_frustum_sampling(reset_pc_in_boxes, calib, info_annos, seglabel, rgbimage)
    # return mask_frustum_sampling(reset_pc_in_boxes, calib, info_annos, depthimage, seglabel, rgbimage)
    # return rgbPP(reset_pc_in_boxes, calib, info_annos, depthimage, rgbimage, seglabel)

# 设置正常视锥体备_val
def frustum_sampling_val2(points, calib, bbox_list, info_annos, depthimage, rgbimage, seglabel):
        
    # print(f"################################################### points 中的点云数量: {points.shape[0]}")    
    # print(bbox_list)
    # print(info_annos)
    """
        视锥体算法
    """
    # 降采样
    # points = random_sampling(points, 50000)
    
    # 将点云转换到摄像机坐标系
    pc_upright_depth = points
    # 创建一个与 pc_upright_depth 形状相同但所有元素为零的数组
    pc_upright_camera = np.zeros_like(pc_upright_depth)
    # 前三列 cam X,Y,Z = depth X,-Z,Y
    pc_upright_camera[:, 0:3] = calib.project_upright_depth_to_upright_camera(pc_upright_depth[:, 0:3])
    # 后三列保持原样
    pc_upright_camera[:, 3:] = pc_upright_depth[:, 3:]
    
    # 将点云数据投影到图像坐标系
    pc_image_coord, pc_depth = calib.project_upright_depth_to_image(pc_upright_depth) 
    
    # depth mask
    pc_depth_reshaped = pc_depth.reshape(-1, 1)
    # pc_newxyz = np.concatenate((pc_image_coord, pc_depth_reshaped), axis=1)
    
    # 定义一个布尔数组，初始时假定所有点都不在任何box2d框内
    points_in_boxes = np.zeros(len(pc_image_coord), dtype=bool)
                    
        # bbox_lists格式'class_name': class_name, 'bbox': bbox, 'confidence': confidence
    for bbox_info in bbox_list:

        # 获取单个对象的2D边界框
        bbox = bbox_info['bbox']
        
        # 获取置信度
        confidence = float(bbox_info['confidence'])

        # print("Bounding Box:", bbox)
        # 获取单个对象的3D边界框
        # gt_box_upright_depth = info_annos['gt_boxes_upright_depth'][ann_idx]

        # 提取2D边界框坐标
        box2d = bbox
        # print("box2d name :", object_name)
        # print("box2d size :", box2d)

        xmin, ymin, xmax, ymax = box2d
        
        box_fov_inds = (pc_image_coord[:, 0] < xmax) & (pc_image_coord[:, 0] >= xmin) & \
                    (pc_image_coord[:, 1] < ymax) & (pc_image_coord[:, 1] >= ymin)
        
        
        # 找出所有在框内的点的索引
        indices_in_box = np.where(box_fov_inds)[0]

        # 根据置信度决定要降采样的点数
        num_points_to_discard = int(len(indices_in_box) * (1 - confidence))  # 基于confidence动态计算要丢弃的点数

        # 丢弃点不应使剩余点数少于100
        if len(indices_in_box) - num_points_to_discard < 100:
            num_points_to_discard = len(indices_in_box) - 100
            
        # 如果可删除的点数变成负数，则不删除任何点
        if num_points_to_discard < 0:
            num_points_to_discard = 0
        
        # 随机选择要丢弃的点
        discard_indices = np.random.choice(indices_in_box, num_points_to_discard, replace=False)
        
        # 设置选中的点为False
        box_fov_inds[discard_indices] = False
        
        # 更新 points_in_boxes，如果点在当前框内，则将相应位置设为True
        points_in_boxes = points_in_boxes | box_fov_inds


    # 未统计到点云，则保留原始数据，可能是2D算法错误也可能是真的没有点
    if not points_in_boxes.any():
        # print("furtusm_null")
        # 将所有元素设置为 True
        points_in_boxes[:] = True
        
        # 2维？
        coord_in_boxes = pc_image_coord[points_in_boxes, :]
        # 3维？
        pc_in_boxes = pc_upright_camera[points_in_boxes, :]
        
        # print("len 11111111: ",pc_in_boxes.shape[0])
        
        # 降采样
        reset_pc_in_boxes = random_sampling(pc_in_boxes, 50000)
        
        # print("未统计到点云，则保留原始数据")
        # print("1111111111_pc_upright_depth_len: ",len(pc_upright_depth))
        # print("2222222222_box_fov_inds_mask_len: ",len(reset_pc_in_boxes))

    else:    
        # 2维？
        coord_in_boxes = pc_image_coord[points_in_boxes, :]
        # 3维？
        pc_in_boxes = pc_upright_camera[points_in_boxes, :]
        
        # print("len 22222222: ",pc_in_boxes.shape[0])
    
        # 降采样
        # reset_pc_in_boxes = random_sampling(reset_pc_in_boxes, 100000)
        
        # print("已统计到点云")
        # print("1111111111_pc_upright_depth_len: ",len(pc_upright_depth))
        # print("2222222222_box_fov_inds_mask_len: ",len(reset_pc_in_boxes))
        

    # 复位坐标   
    reset_pc_in_boxes = flip_axis_reset_camera(pc_in_boxes)
        
    # return mask_frustum_sampling(reset_pc_in_boxes, calib, info_annos, seglabel)

    # print(f"################################################### frustum_in_boxes 中的点云数量: {reset_pc_in_boxes.shape[0]}")  


    return reset_pc_in_boxes

    # return mask_frustum_sampling(reset_pc_in_boxes, calib, info_annos, seglabel, rgbimage)
    # return mask_frustum_sampling(reset_pc_in_boxes, calib, info_annos, depthimage, seglabel, rgbimage)
    # return rgbPP(reset_pc_in_boxes, calib, info_annos, depthimage, rgbimage, seglabel)



# 上================================================================================================================






# SUN RGB-D 数据实例类
class SUNRGBDInstance(object):

    def __init__(self, line):
        # 解析文本行，将其分割成不同的数据部分
        data = line.split(' ')
        # 将字符串转换成浮点数
        data[1:] = [float(x) for x in data[1:]]
        # 对象类别
        self.classname = data[0]
        # 2D边界框的最小x坐标
        self.xmin = data[1]
        # 2D边界框的最小y坐标
        self.ymin = data[2]
        # 2D边界框的最大x坐标
        self.xmax = data[1] + data[3]
        # 2D边界框的最大y坐标
        self.ymax = data[2] + data[4]
        # 2D边界框数组
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])
        # 存储3D对象的中心点坐标和尺寸信息
        self.centroid = np.array([data[5], data[6], data[7]])
        self.width = data[8]
        self.length = data[9]
        self.height = data[10]
        # data[9] is x_size (length), data[8] is y_size (width), data[10] is
        # z_size (height) in our depth coordinate system,
        # l corresponds to the size along the x axis
        
        # 3D边界框的尺寸，注意在深度坐标系中的长度、宽度和高度的对应
        self.size = np.array([data[9], data[8], data[10]]) * 2
        
        # 对象的朝向和航向角
        self.orientation = np.zeros((3, )) # 初始化朝向向量
        self.orientation[0] = data[11] # 朝向向量的x分量
        self.orientation[1] = data[12] # 朝向向量的y分量
        self.heading_angle = np.arctan2(self.orientation[1],
                                        self.orientation[0]) # 航向角
        
        # 构造3D边界框的数组
        self.box3d = np.concatenate(
            [self.centroid, self.size, self.heading_angle[None]]) # 3D边界框的数组

# SUN RGB-D 数据类
class SUNRGBDData(object):
    """SUNRGBD data.

    Generate scannet infos for sunrgbd_converter.

    Args:
        root_path (str): Root path of the raw data.
        split (str, optional): Set split type of the data. Default: 'train'.
        use_v1 (bool, optional): Whether to use v1. Default: False.
        
    SUNRGBD 数据。
    为 sunrgbd_converter 生成扫描信息。
    参数： root_path （str）：原始数据的根路径。 
    split（str，可选）：设置数据的分割类型。默认值："train"：是否使用 v1：假。 
    """

    def __init__(self, root_path, split='train', use_v1=False):
        
        # 存储数据集的根目录
        self.root_dir = root_path
        # 存储数据集的分割类型（例如 'train', 'val', 'test'）
        self.split = split
        # 拼接得到训练/验证数据的目录路径
        self.split_dir = osp.join(root_path, 'sunrgbd_trainval')
        # 定义数据集中的类别
        self.classes = [
            'bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
            'night_stand', 'bookshelf', 'bathtub'
        ]
        # 创建从类别名称到标签的映射字典
        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        # 创建从标签到类别名称的映射字典
        self.label2cat = {
            label: self.classes[label]
            for label in range(len(self.classes))
        }
        # 确保提供的数据集分割类型有效
        assert split in ['train', 'val', 'test']
        # 读取相应分割类型的样本索引文件
        split_file = osp.join(self.split_dir, f'{split}_data_idx.txt')
        mmengine.check_file_exist(split_file)

        # 读取样本ID列表
        self.sample_id_list = map(int, mmengine.list_from_file(split_file))
        # 获取RGB图
        self.image_dir = osp.join(self.split_dir, 'image')
        # 相机参数
        self.calib_dir = osp.join(self.split_dir, 'calib')
        # 点云
        self.depth_dir = osp.join(self.split_dir, 'depth')
        # 获取gabor图
        self.gabor_dir = osp.join(self.split_dir, 'gabor')

        # 深度图 and 2D分割
        self.depthimage_dir = osp.join(self.split_dir, 'depth_image')
        self.seglabel_dir = osp.join(self.split_dir, 'seg_label')
        
        # 根据版本设置标签目录路径
        if use_v1:
            self.label_dir = osp.join(self.split_dir, 'label_v1')
        else:
            self.label_dir = osp.join(self.split_dir, 'label')


        # ============================================yolov9-c===============================================

        # confThres0.362_IouThres0.45
        self.train_bbox_list = self._preload_bbox_data('/root/autodl-tmp/sunrgbd2d/2/sunrgbd_rgb_train_yolov9-e_confThres0.362_IouThres0.45.txt')
        self.val_bbox_list = self._preload_bbox_data('/root/autodl-tmp/sunrgbd2d/2/sunrgbd_rgb_val_yolov9-e_confThres0.362_IouThres0.45.txt')

        # confThres0.1_IouThres0.45
        # self.train_bbox_list = self._preload_bbox_data('/root/autodl-tmp/sunrgbd2d/2/sunrgbd_rgb_train_yolov9-e_confThres0.1_IouThres0.45.txt')
        # self.val_bbox_list = self._preload_bbox_data('/root/autodl-tmp/sunrgbd2d/2/sunrgbd_rgb_val_yolov9-e_confThres0.1_IouThres0.45.txt')

        # confThres0.25_IouThres0.45
        # self.train_bbox_list = self._preload_bbox_data('/root/autodl-tmp/sunrgbd2d/2/sunrgbd_rgb_train_yolov9-e_confThres0.25_IouThres0.45.txt')
        # self.val_bbox_list = self._preload_bbox_data('/root/autodl-tmp/sunrgbd2d/2/sunrgbd_rgb_val_yolov9-e_confThres0.25_IouThres0.45.txt')

        # confThres0.5_IouThres0.45
        # self.train_bbox_list = self._preload_bbox_data('/root/autodl-tmp/sunrgbd2d/2/sunrgbd_rgb_train_yolov9-e_confThres0.5_IouThres0.45.txt')
        # self.val_bbox_list = self._preload_bbox_data('/root/autodl-tmp/sunrgbd2d/2/sunrgbd_rgb_val_yolov9-e_confThres0.5_IouThres0.45.txt')

        # confThres0.9_IouThres0.45
        # self.train_bbox_list = self._preload_bbox_data('/root/autodl-tmp/sunrgbd2d/2/sunrgbd_rgb_train_yolov9-e_confThres0.9_IouThres0.45.txt')
        # self.val_bbox_list = self._preload_bbox_data('/root/autodl-tmp/sunrgbd2d/2/sunrgbd_rgb_val_yolov9-e_confThres0.9_IouThres0.45.txt')


        # ============================================yolov9-c===============================================
        # train_confThres0.25_IouThres0.45_mixTrue
        # self.train_bbox_list = self._preload_bbox_data('/root/autodl-tmp/sunrgbd2d/sunrgbd_rgb_train_yolov9_confThres0.25_IouThres0.45_mixTrue.txt')
        # self.val_bbox_list = self._preload_bbox_data('/root/autodl-tmp/sunrgbd2d/sunrgbd_rgb_val_yolov9_confThres0.25_IouThres0.45_mixTrue.txt')

        # confThres0.25_IouThres0.45
        # self.train_bbox_list = self._preload_bbox_data('/root/autodl-tmp/sunrgbd2d/sunrgbd_rgb_train_yolov9_confThres0.25_IouThres0.45.txt')
        # self.val_bbox_list = self._preload_bbox_data('/root/autodl-tmp/sunrgbd2d/sunrgbd_rgb_val_yolov9_confThres0.25_IouThres0.45.txt')

        # confThres0.01_IouThres0.45
        # self.train_bbox_list = self._preload_bbox_data('/root/autodl-tmp/sunrgbd2d/sunrgbd_rgb_train_yolov9_confThres0.01_IouThres0.45.txt')
        # self.val_bbox_list = self._preload_bbox_data('/root/autodl-tmp/sunrgbd2d/sunrgbd_rgb_val_yolov9_confThres0.01_IouThres0.45.txt')

        # mix0.25and0.1
        # self.train_bbox_list = self._preload_bbox_data('/root/autodl-tmp/sunrgbd2d/sunrgbd_rgb_train_yolov9_confThres_mix0.25and0.1_IouThres0.45.txt')
        # self.val_bbox_list = self._preload_bbox_data('/root/autodl-tmp/sunrgbd2d/sunrgbd_rgb_val_yolov9_confThres_mix0.25and0.1_IouThres0.45.txt')


    def _preload_bbox_data(self, data_file):
        BBox_lists = {}
        with open(data_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                sample_idx = parts[0]
                class_name = parts[1]
                bbox = list(map(float, parts[2:6]))
                confidence = parts[6]
                if sample_idx not in BBox_lists:
                    BBox_lists[sample_idx] = []
                BBox_lists[sample_idx].append({'class_name': class_name, 'bbox': bbox, 'confidence': confidence})
        return BBox_lists 

    # 返回数据集中样本的数量
    def __len__(self):
        return len(self.sample_id_list)

    # 根据索引获取对应的图像文件名，并从图像目录中读取图像
    def get_image(self, idx):
        img_filename = osp.join(self.image_dir, f'{idx:06d}.jpg')
        return mmcv.imread(img_filename)
        
    # 获取图像的尺寸（高度和宽度）
    def get_image_shape(self, idx):
        image = self.get_image(idx)
        return np.array(image.shape[:2], dtype=np.int32)
    
    # 获取gabor纹理结果
    def get_gaborimage(self, idx):
        gabor_filename = osp.join(self.gabor_dir, f'{idx:06d}.jpg')
        return cv2.imread(gabor_filename, cv2.IMREAD_UNCHANGED)
    

    # 根据索引获取对应的点云文件名，并从深度目录中读取深度信息
    def get_depth(self, idx):
        depth_filename = osp.join(self.depth_dir, f'{idx:06d}.mat')
        # sio读取.mat文件
        depth = sio.loadmat(depth_filename)['instance']
        return depth

    # Frustum-calib
    def get_Frustum_Calibration(self, idx):
        calib_filepaths = osp.join(self.calib_dir, f'{idx:06d}.txt')
        Ks, Rts = self.get_calibration(idx)
        return SUNRGBD_Calibration(calib_filepath=calib_filepaths,Rtilt=Rts,K=Ks)
        
    # 返回分割掩码矩阵
    def get_seglabel(self, idx):
        seglabel_filepaths = osp.join(self.seglabel_dir, f'{idx:06d}.txt')
        """
        从给定文件中读取掩码数据，构建二维数组。

        参数:
        mask_filename: 包含掩码数据的文本文件路径。

        返回:
        # 二维数组。
        """
        segmask = []
        with open(seglabel_filepaths, 'r') as file:
            for line in file:
                # 将每行的数字分隔开，并转换为整数
                row = [int(num) for num in line.split()]
                segmask.append(row)

        # 翻转行的顺序，使左下角的点成为数组的起始点
        # segmask.reverse()
        
        return segmask   

    # 返回深度图矩阵
    def get_depthimage(self, idx):
        depthimage_filename = osp.join(self.depthimage_dir, f'{idx:06d}.png')
        
        return cv2.imread(depthimage_filename, cv2.IMREAD_UNCHANGED)
        
        
    # 根据索引获取校准文件，解析其中的相机内参（K）和旋转矩阵（Rt）
    def get_calibration(self, idx):
        calib_filepath = osp.join(self.calib_dir, f'{idx:06d}.txt')
        lines = [line.rstrip() for line in open(calib_filepath)]
        Rt = np.array([float(x) for x in lines[0].split(' ')])
        Rt = np.reshape(Rt, (3, 3), order='F').astype(np.float32)
        K = np.array([float(x) for x in lines[1].split(' ')])
        K = np.reshape(K, (3, 3), order='F').astype(np.float32)
        return K, Rt


    # 根据索引获取对应的标签文件，解析其中的对象信息
    def get_label_objects(self, idx):
        label_filename = osp.join(self.label_dir, f'{idx:06d}.txt')
        lines = [line.rstrip() for line in open(label_filename)]
        objects = [SUNRGBDInstance(line) for line in lines]
        return objects
    
    
    # 获取2Dlist
    def get_bbox_by_sample_idx(self, sample_idx, strs):
        if strs == 'val':
            return self.val_bbox_list.get(str(sample_idx), [])
        else:
            return self.train_bbox_list.get(str(sample_idx), [])


    # 从原始数据中获取信息
    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        
        """Get data infos.
            get_infos
        This method gets information from the raw data.

        Args:
            num_workers (int, optional): Number of threads to be used.
                Default: 4.
            has_label (bool, optional): Whether the data has label.
                Default: True.
            sample_id_list (list[int], optional): Index list of the sample.
                Default: None.

        Returns:
            infos (list[dict]): Information of the raw data.
        """
        
        """
        从原始数据中获取信息。
        
        Args:
            num_workers (int, optional): 使用的线程数，默认为4。
            has_label (bool, optional): 数据是否包含标签，默认为True。
            sample_id_list (list[int], optional): 样本的索引列表，默认为None。

        Returns:
            infos (list[dict]): 原始数据的信息列表。
        """
        # 判断是否是分割信息健全场景
        # with open('/root/autodl-fs/right_data_idx.txt', 'r') as f:
            # right_data_idxs = set(int(line.strip()) for line in f)





        # 处理单个场景的函数
        def process_single_scene(sample_idx):
            
            print(f'{self.split} sample_idx: {sample_idx}')

            # TODO: Check whether can move the point
            #  sampling process during training.

            
            pc_upright_depth = self.get_depth(sample_idx)


            # 初始化场景信息字典
            info = dict()
            
            # 添加点云信息
            pc_info = {'num_features': 6, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            
            # 获取图像信息
            img_path = osp.join('image', f'{sample_idx:06d}.jpg')
            
            # 获取图像信息
            image_info = {
                'image_idx': sample_idx,
                'image_shape': self.get_image_shape(sample_idx),
                'image_path': img_path
            }
            info['image'] = image_info

            # 获取校准信息
            K, Rt = self.get_calibration(sample_idx)
            calib_info = {'K': K, 'Rt': Rt}
            info['calib'] = calib_info
            
            # 获取RGB图
            rgbimage = self.get_image(sample_idx)

            # 获取深度图
            # depthimage = self.get_depthimage(sample_idx)
            
            # 获取gabor图
            # gaborimage = self.get_gaborimage(sample_idx)

            # 获取分割掩码
            # seglabel = self.get_seglabel(sample_idx)
            

            # 如果数据集包含标签，获取对象标注信息
            if has_label:
                # 获取当前样本的标签对象列表
                obj_list = self.get_label_objects(sample_idx)
                # 初始化注释字典
                annotations = {}
                # 计算标签对象列表中有效类别的数量
                annotations['gt_num'] = len([
                    obj.classname for obj in obj_list
                    if obj.classname in self.cat2label.keys()
                ])
                # 如果存在有效的标签对象
                if annotations['gt_num'] != 0:
                    
                    # 构建标签对象的类别名称数组
                    annotations['name'] = np.array([
                        obj.classname for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ])
                    
                    # 构建标签对象的2D边界框数组
                    annotations['bbox'] = np.concatenate([
                        obj.box2d.reshape(1, 4) for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ],axis=0)
                    
                    # 构建标签对象的中心点坐标数组
                    annotations['location'] = np.concatenate([
                        obj.centroid.reshape(1, 3) for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ],axis=0)
                    
                    # 构建标签对象的尺寸数组（长、宽、高）
                    annotations['dimensions'] = 2 * np.array([
                        [obj.length, obj.width, obj.height] for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ])  # lwh (depth) format
                    
                    # 构建标签对象的航向角数组
                    annotations['rotation_y'] = np.array([
                        obj.heading_angle for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ])
                    
                    # 创建索引数组，用于标识每个标签对象
                    annotations['index'] = np.arange(
                        len(obj_list), dtype=np.int32)
                    
                    # 创建标签对象的类别标签数组
                    annotations['class'] = np.array([
                        self.cat2label[obj.classname] for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ])
                    # 构建标签对象的3D边界框数组
                    annotations['gt_boxes_upright_depth'] = np.stack(
                        [
                            obj.box3d for obj in obj_list
                            if obj.classname in self.cat2label.keys()
                        ],
                        axis=0)  # (K,8)
                # 将注释信息添加到信息字典中
                info['annos'] = annotations
            
            # 确保points点云目录存在并保存点云数据
            mmengine.mkdir_or_exist(osp.join(self.root_dir, 'points'))
            
            # 创建SUNRGBD_Calibration对象
            calib = self.get_Frustum_Calibration(sample_idx)
        
            # 接入视锥体算法
            # print(f"正在处理样本：{sample_idx}, 函数调用栈：")
                # 检查seg_mask是否为空
            # PP(pc_upright_depth_subsampled, calib, rgbimage) 
            # 视锥体

            # 降采样【开启】
            pc_upright_depth_subsampled = pc_upright_depth


            """
            if sample_idx in right_data_idxs:
                # 正确的分割
                segbool = True
                pc_upright_frustum = frustum_sampling_k_means_0(pc_upright_depth_subsampled, calib, info['annos'], depthimage, rgbimage, seglabel, gaborimage, segbool, sample_idx)
            else:
                # 错误的分割
                segbool = False
                pc_upright_frustum = frustum_sampling_k_means_0(pc_upright_depth_subsampled, calib, info['annos'], depthimage, rgbimage, seglabel, gaborimage, segbool, sample_idx)
            

            # 读取二维检测算法结果
            """
            ####################视锥体部分######################
            # train采用2D检测框作为先验条件
            if self.split == 'train':
                sample_idx_str = str(sample_idx)
                bbox_list = self.get_bbox_by_sample_idx(sample_idx_str, 'train')

                # pc_upright_frustum = pc_upright_depth_subsampled
                
                pc_upright_frustum = frustum_sampling_val(pc_upright_depth_subsampled, calib, bbox_list, info['annos'], rgbimage, sample_idx)
                # pc_upright_frustum = frustum_sampling_train_True(pc_upright_depth_subsampled, calib, info['annos'], rgbimage)
                # pc_upright_frustum = frustum_sampling_TRUE(pc_upright_depth_subsampled, calib, info['annos'], rgbimage)
            # val使用2D检测算法得到的检测框
            elif self.split == 'val':
                sample_idx_str = str(sample_idx)
                bbox_list = self.get_bbox_by_sample_idx(sample_idx_str, 'val')
                
                # pc_upright_frustum = pc_upright_depth_subsampled
                
                pc_upright_frustum = frustum_sampling_val(pc_upright_depth_subsampled, calib, bbox_list, info['annos'], rgbimage, sample_idx)
                # pc_upright_frustum = frustum_sampling_train_True(pc_upright_depth_subsampled, calib, info['annos'], rgbimage)
                # pc_upright_frustum = frustum_sampling_TRUE(pc_upright_depth_subsampled, calib, info['annos'], rgbimage)
            else:
                print("Are you kidding me?")
            
            # pc_upright_frustum = frustum_sampling_train(pc_upright_depth_subsampled, calib, info['annos'], depthimage, rgbimage, seglabel)

            # SAMPLE_NUM = 50000
            # pc_upright_frustum = random_sampling(pc_upright_frustum, SAMPLE_NUM)
            # print(f"################################################### lost 中的点云数量: {pc_upright_frustum.shape[0]}")  

            # 使用视锥体算法将深度和纹理融合到点云后两位
            # pc_upright_frustum = frustum_sampling_AddFusionFeatures(pc_upright_depth_subsampled, calib, info['annos'], depthimage, gaborimage, rgbimage, seglabel)
            
            # 跳过视锥体直接生成点云
            # pc_upright_frustum = pc_upright_depth_subsampled


            """ 自动生成深度图
            save_dir = "/root/autodl-tmp/sunrgbd/sunrgbd_trainval/fix_depth/"
            depth_file = os.path.join(save_dir, f'{sample_idx:06d}.png')

            # 如果深度图文件存在，则直接读取
            if os.path.exists(depth_file):
                print(f"文件 {depth_file} 已存在，直接读取深度图...")

                # 读取归一化后的深度图
                normalized_depth_map = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
                
                # 从depth_max_min.txt读取该sample_idx对应的最大值和最小值
                max_val, min_val = read_max_min_from_file(sample_idx, save_dir)
                
                # 还原深度图
                frustum_depth_map = restore_depth_map_from_normalized(normalized_depth_map, max_val, min_val)
                print(f"深度图 {sample_idx:06d} 已还原。")
            else:
                print(f"文件 {depth_file} 不存在，生成并保存归一化深度图...")
                # 生成深度图
                frustum_depth_map = frustum_point_cloud_completion(pc_upright_frustum, calib, rgbimage)
                # 归一化深度图并保存，同时保存该场景下的最大值和最小值
                save_depth_map_and_minmax(frustum_depth_map, sample_idx, save_dir)
            """

            # 统计缺失索引，找到frustum_depth_map中缺失索引位置的深度数值
            # missing_pixel_positions = frustum_point_cloud_missing_pixel_positions(pc_upright_frustum, calib, rgbimage)


            # 根据深度数值和相机内外参数反向还原点云，并将点云补充进原点云中
            # add_points2d:x,y,depth,r,g,b
            # add_points2d = frustum_add_points2d(frustum_depth_map, missing_pixel_positions, rgbimage)

            # 将add_points2d还原并拼接在pc_upright_frustum之后
            # pc_upright_frustum = frustum_add_points3d(pc_upright_frustum, add_points2d, calib)

            

            # 测试是否转换成功
            # pc_newxyz2 = calib.project_upright_image_to_depth2(frustum_depth_map, rgbimage)
            # rgbPP_NOWC_TEST(pc_newxyz2, calib, rgbimage)


            # rgbPP(pc_upright_frustum, calib, rgbimage, sample_idx)

            # rgbPPB(pc_upright_frustum, calib, rgbimage)  
            # rgbPPTOP(pc_upright_frustum, calib, rgbimage)  

            # 验证
            # pc_upright_frustum[:, 6:7] = 0.5

            # additional_dimension = np.full((pc_upright_frustum.shape[0], 1), 0.5)
            # last_three_dimensions = pc_upright_frustum[:, -3:]  # 选择最后三列

            # 将这个新的列添加到原始数据中
            # pc_upright_frustum = np.concatenate((pc_upright_frustum, last_three_dimensions), axis=1)

            # aaa = np.array_equal(pc_upright_frustum, pc_upright_frustum2)

            # if aaa:
                # print("########################array_equal#########################")

            # pc_upright_frustum = pc_upright_depth_subsampled
            # 如果是分割信息健全场景，则用分割结果进行二次分割
            # if sample_idx in right_data_idxs:
                # print(f'sample_idx {sample_idx} seg is complete')
              #   pc_upright_frustum = frustum_sampling_SEG(pc_upright_frustum, calib, info['annos'], depthimage, rgbimage, seglabel)

            # pc_upright_frustum = frustum_sampling(pc_upright_depth_subsampled, calib, info['annos'], depthimage, rgbimage, seglabel) # 视锥体裁剪

            """
            if pc_upright_frustum.shape[1] >= 8:
                # 获取第七和第八维度数据的数量
                num_points = pc_upright_frustum.shape[0]
                
                # 生成一个随机排列的索引数组
                indices = np.random.permutation(num_points)
                
                # 选取一半的数据索引
                nine_tenths_indices = indices[:num_points * 9 // 10]
                
                # 将选中的一半数据的第七和第八维度设置为0.5
                pc_upright_frustum[nine_tenths_indices, 6] = 0.5  # 第七维度
                pc_upright_frustum[nine_tenths_indices, 7] = 0.5  # 第八维度
            """

            # 绘图
            # PP(pc_upright_frustum, calib, rgbimage) 

            # joint_entropy = long_tail_distribution_data_balancing(info['annos'], rgbimage, depthimage, seglabel)
            
            
            pc_upright_frustum.tofile(
                osp.join(self.root_dir, 'points', f'{sample_idx:06d}.bin'))
            info['pts_path'] = osp.join('points', f'{sample_idx:06d}.bin')
            
            return info
        
        # 使用指定的样本索引列表或默认的样本列表
        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        
        # 使用线程池来并行处理所有样本
        with futures.ThreadPoolExecutor(num_workers) as executor:
            
            infos = executor.map(process_single_scene, sample_id_list)


        return list(infos)



# 处理SUN RGBD数据集中的坐标系转换和投影操作
class SUNRGBD_Calibration(object):

    ''' 
        参数：
        calib_filepath: 包含相机内参和旋转矩阵的文件路径。
        Rtilt: 旋转矩阵，用于将坐标系从竖直方向调整到地面的方向。
        K: 相机内参矩阵
        Calibration matrices and utils
        We define five coordinate system in SUN RGBD dataset

        camera coodinate:
            Z is forward, Y is downward, X is rightward

        depth coordinate:
            Just change axis order and flip up-down axis from camera coord

        # X is right, Y is forward, Z is up not tilted


        upright depth coordinate: tilted depth coordinate by Rtilt such that Z is gravity direction,
            Z is up-axis, Y is forward, X is right-ward

        upright camera coordinate:
            Just change axis order and flip up-down axis from upright depth coordinate

        image coordinate:
            ----> x-axis (u)
           |
           v
            y-axis (v)

        depth points are stored in upright depth coordinate.
        labels for 3d box (basis, centroid, size) are in upright depth coordinate.
        2d boxes are in image coordinate

        We generate frustum point cloud and 3d box in upright camera coordinate
    '''
    def __init__(self, calib_filepath=None, Rtilt=None, K=None):
        if calib_filepath is not None:
            lines = [line.rstrip() for line in open(calib_filepath)]
            Rtilt = np.array([float(x) for x in lines[0].split(' ')])
            self.Rtilt = np.reshape(Rtilt, (3, 3), order='F')
            K = np.array([float(x) for x in lines[1].split(' ')])
            # 存储外参
            self.K = np.reshape(K, (3, 3), order='F')
        else:
            assert Rtilt is not None and K is not None
            self.Rtilt = Rtilt
            # 存储外参
            self.K = K

        self.f_u = self.K[0, 0]
        self.f_v = self.K[1, 1]
        self.c_u = self.K[0, 2]
        self.c_v = self.K[1, 2]


    def flip_axis_to_camera(self, pc):
        ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
            Input and output are both (N,3) array
        '''

        pc2 = np.copy(pc)
        pc2[:, [0, 1, 2]] = pc2[:, [0, 2, 1]]  # cam X,Y,Z = depth X,-Z,Y
        pc2[:, 1] *= -1
        # @ np.array([[1, 0, 0], [0, 0, 1],[0, -1, 0]])

        return pc2

    def flip_axis_to_depth(self, pc):
        pc2 = np.copy(pc)
        pc2[:, [0, 1, 2]] = pc2[:, [0, 2, 1]]  # depth X,Y,Z = cam X,Z,-Y
        pc2[:, 2] *= -1
        # @ np.array([[1, 0, 0], [0, 0, -1],[0, 1, 0]])
        return pc2


    def project_upright_depth_to_camera(self, pc):
        ''' project point cloud from depth coord to camera coordinate
            Input: (N,3) Output: (N,3)
        '''
        
        # Project upright depth to depth coordinate 使用外参数旋转点云
        pc2 = np.dot(np.transpose(self.Rtilt), np.transpose(pc[:, 0:3]))  # (3,n)

        pc2 = self.flip_axis_to_camera(np.transpose(pc2))

        return pc2

    # 将正立深度坐标系中的点云投影到图像坐标系
    def project_upright_depth_to_image(self, pc):
        ''' Input: (N,3) Output: (N,2) UV and (N,) depth '''
        """
        将正立深度坐标系中的点云投影到图像坐标系。
        
        参数:
            pc (numpy.ndarray): 点云数据，形状为 (N,3)，其中N为点的数量。
            
        返回:
            tuple: 包含两个元素的元组。
                第一个元素是形状为 (N,2) 的numpy数组，表示UV坐标。
                第二个元素是形状为 (N,) 的numpy数组，表示深度值。
        """

        # 将输入的点云 pc 从正立深度坐标系转换到摄像头坐标系。直接穿过点云场景的轴变为z轴
        # 外参矩阵R
        pc2 = self.project_upright_depth_to_camera(pc)


        """
        # print("将正立深度坐标系中的点云投影到图像坐标系")
        # 俯视图测试1
        # 计算点云中心
        center = np.mean(pc2[:, :3], axis=0)
        print(f"点云中心点: {center}")
        # 将点云中心移动到原点
        pc2[:, :3] -= center

        # 在 Z-Y 平面旋转90度，绕X轴旋转
        rotation_matrix90 = np.array([[1, 0, 0],
                                    [0, 0, -1],
                                    [0, 1, 0]])  # 绕X轴顺时针旋转90度的矩阵

        rotation_matrix60 = np.array([[1, 0, 0],
                                    [0, 0.5, -0.866],
                                    [0, 0.866, 0.5]])  # 绕X轴顺时针旋转90度的矩阵
        
        rotation_matrix45 = np.array([[1, 0, 0],
                                    [0, 0.7071, -0.7071],
                                    [0, 0.7071, 0.7071]])  # 绕X轴顺时针旋转90度的矩阵
        
        rotation_matrix30 = np.array([[1, 0, 0],
                                    [0, 0.866, -0.5],
                                    [0, 0.5, 0.866]])  # 绕X轴顺时针旋转90度的矩阵        


        rotation_matrix15 = np.array([[1, 0, 0],
                                    [0, 0.9659, -0.2588],
                                    [0, 0.2588, 0.9659]])  # 绕X轴顺时针旋转90度的矩阵

        rotation_matrix10 = np.array([[1, 0, 0],
                                    [0, 0.9848, -0.1736],
                                    [0, 0.1736, 0.9848]])  # 绕X轴顺时针旋转90度的矩阵

        rotation_matrix05 = np.array([[1, 0, 0],
                                    [0, 0.9962, -0.0872],
                                    [0, 0.0872, 0.9962]])  # 绕X轴顺时针旋转90度的矩阵

        
        pc2[:, :3] = np.dot(pc2[:, :3], rotation_matrix05.T)
        # 将点云移回原中心位置
        pc2[:, :3] += center
        """


        # 使用摄像头的内参矩阵 self.K 对点云进行投影，得到每个点在图像坐标系中的齐次坐标。内参矩阵包含了摄像头的焦距和光学中心等信息
        uv = np.dot(pc2, np.transpose(self.K))  # (n,3) # (N,3)齐次坐标
        # 将齐次坐标转换为非齐次坐标，即将 (u, v, w) 转换为 (u/w, v/w)，这样就得到了每个点在图像平面上的 UV 坐标
        uv[:, 0] /= uv[:, 2] # u坐标除以w坐标
        uv[:, 1] /= uv[:, 2] # v坐标除以w坐标

        # 函数返回两个数组：第一个是点云的 UV 坐标数组，第二个是点云的深度值数组。UV 坐标用于确定点在图像上的位置，而深度值表示点相对于摄像头的距离
        return uv[:, 0:2], pc2[:, 2]
    

    def project_upright_depth_to_image_NOWC_TEST(self, pc):

        # 将输入的点云 pc 从正立深度坐标系转换到摄像头坐标系。直接穿过点云场景的轴变为z轴
        # NO外参矩阵R
        pc2 = pc[:, 0:3]

        # 使用摄像头的内参矩阵 self.K 对点云进行投影，得到每个点在图像坐标系中的齐次坐标。内参矩阵包含了摄像头的焦距和光学中心等信息
        uv = np.dot(pc2, np.transpose(self.K))  # (n,3) # (N,3)齐次坐标
        # 将齐次坐标转换为非齐次坐标，即将 (u, v, w) 转换为 (u/w, v/w)，这样就得到了每个点在图像平面上的 UV 坐标
        uv[:, 0] /= uv[:, 2] # u坐标除以w坐标
        uv[:, 1] /= uv[:, 2] # v坐标除以w坐标

        # 
        # return uv[:, 0:2], pc2[:, 2], pc[:, 3:]
        return np.hstack((uv[:, 0:2], pc2[:, 2][:, np.newaxis], pc[:, 3:]))
    

    # 将2D坐标投影回深度坐标系
    def project_upright_image_to_depth(self, add_points2d):
        """
        将2D图像上的 (x, y, depth) 反向投影回三维点云 (X, Y, Z)
        
        参数：
        - add_points2d (np.ndarray): 包含 (x, y, depth, R, G, B) 的数组，形状为 (N, 6)。
        
        返回：
        - add_points3d (np.ndarray): 三维点云 (X, Y, Z, R, G, B)，形状为 (N, 6)。
        """

        # 提取 x, y 和 depth
        x = add_points2d[:, 0]
        y = add_points2d[:, 1]
        depth = add_points2d[:, 2]
        
        # 使用内参矩阵 self.K 的逆，计算 (X_cam, Y_cam, Z_cam)
        K_inv = np.linalg.inv(self.K)

        # 齐次坐标 (u, v, 1) 乘以深度
        uv1 = np.stack((x, y, np.ones_like(x)), axis=-1)  # 形状为 (N, 3)
        points_camera = np.dot(uv1, K_inv.T) * depth[:, np.newaxis]  # 形状为 (N, 3)
        
        # 恢复坐标轴：将 (X, -Z, Y) 变回 (X, Y, Z)
        points_camera[:, [1, 2]] = points_camera[:, [2, 1]]  # 交换 Y 和 Z
        points_camera[:, 1] *= -1  # 恢复 Z 的符号
        
        # 使用外参矩阵 self.Rtilt 的逆，计算深度坐标系下的 (X_depth, Y_depth, Z_depth)
        Rtilt_inv = np.linalg.inv(self.Rtilt)
        points_depth = np.dot(points_camera, Rtilt_inv.T)
        
        # 提取 RGB 信息
        rgb = add_points2d[:, 3:6]
        
        # 拼接点云和 RGB，形成 add_points3d
        add_points3d = np.hstack((points_depth, rgb))
        
        return add_points3d


    # 将2D坐标投影回深度坐标系
    def project_upright_image_to_depth2B(self, depth_map, rgbimage):
        """
        使用相机内参和外参矩阵，将深度图投影回三维点云。
        该点云的颜色为深度图对应的 rgbimage 中的颜色 (RGB)。
        
        参数：
        - depth_map (np.ndarray): 已还原的深度图 (height, width， depth)。
        - rgbimage (np.ndarray): 对应的 RGB 图像 (height, width, rgb)。
        
        返回：
        - points_3d (np.ndarray): 三维点云 (X, Y, Z, R, G, B)，形状为 (N, 6)。
        """

        # 获取内外参数
        K = self.K

        # 获取内参矩阵的参数
        f_x = K[0, 0]
        f_y = K[1, 1]
        c_x = K[0, 2]
        c_y = K[1, 2]

        # 获取深度图的尺寸
        height, width = depth_map.shape

        # 生成像素坐标 (u, v)
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # 展平所有像素坐标和深度值
        u = u.flatten()
        v = v.flatten()
        Z = depth_map.flatten()

        # 计算三维点的 X, Y 坐标
        X = (u - c_x) * Z / f_x
        Y = (v - c_y) * Z / f_y

        # 将 X, Y, Z 组合成三维点
        # points_3d = np.stack((X, Y, Z), axis=-1)

        # 提取 RGB 值，并展平
        rgb_flat = rgbimage.reshape(-1, 3)  # 形状为 (N, 3)
        
        # 将 X, Y, Z 和 RGB 组合成一个包含 (X, Y, Z, R, G, B) 的点云
        points_3d_rgb = np.hstack((np.stack((X, Y, Z), axis=-1), rgb_flat))



        return points_3d_rgb

    # 将2D坐标投影回深度坐标系
    def project_upright_image_to_depth2(self, depth_map, rgbimage):
        """
        使用相机内参和外参矩阵，将深度图投影回三维点云。
        该点云的颜色为深度图对应的 rgbimage 中的颜色 (RGB)。
        
        参数：
        - depth_map (np.ndarray): 已还原的深度图 (height, width)。
        - rgbimage (np.ndarray): 对应的 RGB 图像 (height, width, 3)。
        
        返回：
        - new_pointnets (np.ndarray): 带有 (X, Y, Z, R, G, B) 信息的三维点云，形状为 (H*W, 6)。
        """
        
        # 获取内参矩阵的参数
        K = self.K
        f_x = K[0, 0]
        f_y = K[1, 1]
        c_x = K[0, 2]
        c_y = K[1, 2]

        # 获取深度图的尺寸
        height, width = depth_map.shape

        # 创建new_pointnets，形状为 (height, width, 6)
        # 前3个维度是深度点云的 (X, Y, Z)，后3个维度是 RGB 值
        new_pointnets = np.zeros((height, width, 6))

        # 填充深度 (Z) 和 RGB
        new_pointnets[:, :, 2] = depth_map  # 第三维填入 Z（深度值）
        new_pointnets[:, :, 3:] = rgbimage  # 后三维填入 RGB 值

        # 生成像素坐标 (u, v)
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # 计算三维点的 X, Y 坐标
        X = (u - c_x) * depth_map / f_x
        Y = (v - c_y) * depth_map / f_y

        # 填充 X 和 Y 坐标到 new_pointnets 中
        new_pointnets[:, :, 0] = X  # 第一维填入 X
        new_pointnets[:, :, 1] = Y  # 第二维填入 Y

        # 展平 new_pointnets 为 (H*W, 6) 的形状
        new_pointnets_flat = new_pointnets.reshape(-1, 6)

        # 保存为 .bin 格式文件
        save_path = "/root/autodl-tmp/new_pointnets.bin"
        new_pointnets_flat.astype('float32').tofile(save_path)
        print(f"点云数据已保存至 {save_path}")

        return new_pointnets_flat   

    def project_upright_depth_to_upright_camera(self, pc):
        return self.flip_axis_to_camera(pc)

    def project_upright_camera_to_upright_depth(self, pc):
        return self.flip_axis_to_depth(pc)

    def project_image_to_camera(self, uv_depth):
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v
        pts_3d_camera = np.zeros((n, 3))
        pts_3d_camera[:, 0] = x
        pts_3d_camera[:, 1] = y
        pts_3d_camera[:, 2] = uv_depth[:, 2]
        return pts_3d_camera

    def project_image_to_upright_camera(self, uv_depth):
        # 从图像坐标系转换到摄像头坐标系
        pts_3d_camera = self.project_image_to_camera(uv_depth)
        # 将摄像头坐标系的点翻转到深度坐标系
        pts_3d_depth = self.flip_axis_to_depth(pts_3d_camera)
        # 应用旋转以对齐到正立深度坐标系
        pts_3d_upright_depth = np.transpose(np.dot(self.Rtilt, np.transpose(pts_3d_depth)))
        # 翻转点云
        return self.project_upright_depth_to_upright_camera(pts_3d_upright_depth)

