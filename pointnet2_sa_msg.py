# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from mmcv.cnn import ConvModule
from torch import Tensor, nn

from mmdet3d.models.layers.pointnet_modules import build_sa_module
from mmdet3d.models.layers import PointFPModule, build_sa_module
from mmdet3d.registry import MODELS
from mmdet3d.utils import OptConfigType
from .base_pointnet import BasePointNet

ThreeTupleIntType = Tuple[Tuple[Tuple[int, int, int]]]
TwoTupleIntType = Tuple[Tuple[int, int, int]]
TwoTupleStrType = Tuple[Tuple[str]]

# 多尺度pointnet
@MODELS.register_module()
class PointNet2SAMSG(BasePointNet):
    """PointNet2 with Multi-scale grouping.

    Args:
        in_channels (int): Input channels of point cloud.
        num_points (tuple[int]): The number of points which each SA
            module samples.
        radii (tuple[float]): Sampling radii of each SA module.
        num_samples (tuple[int]): The number of samples for ball
            query in each SA module.
        sa_channels (tuple[tuple[int]]): Out channels of each mlp in SA module.
        aggregation_channels (tuple[int]): Out channels of aggregation
            multi-scale grouping features.
        fps_mods Sequence[Tuple[str]]: Mod of FPS for each SA module.
        fps_sample_range_lists (tuple[tuple[int]]): The number of sampling
            points which each SA module samples.
        dilated_group (tuple[bool]): Whether to use dilated ball query for
        out_indices (Sequence[int]): Output from which stages.
        norm_cfg (dict): Config of normalization layer.
        sa_cfg (dict): Config of set abstraction module, which may contain
            the following keys and values:

            - pool_mod (str): Pool method ('max' or 'avg') for SA modules.
            - use_xyz (bool): Whether to use xyz as a part of features.
            - normalize_xyz (bool): Whether to normalize xyz with radii in
              each SA module.
    """
    """使用多尺度分组的 PointNet2.

    参数:
        in_channels (int): 点云的输入通道数。
        num_points (tuple[int]): 每个SA模块采样的点数。
        radii (tuple[float]): 每个SA模块的采样半径。
        num_samples (tuple[int]): 每个SA模块进行球查询的样本数。
        sa_channels (tuple[tuple[int]]): 每个SA模块中mlp的输出通道数。
        aggregation_channels (tuple[int]): 聚合多尺度分组特征的输出通道数。
        fps_mods (Sequence[Tuple[str]]): 每个SA模块的FPS模式。
        fps_sample_range_lists (tuple[tuple[int]]): 每个SA模块采样点的数目。
        dilated_group (tuple[bool]): 是否在每个SA模块中使用扩展球查询。
        out_indices (Sequence[int]): 输出的阶段。
        norm_cfg (dict): 归一化层的配置。
        sa_cfg (dict): 设置抽象模块的配置，可能包含以下键和值:
            - pool_mod (str): SA模块的池化方法（'max' 或 'avg'）。
            - use_xyz (bool): 是否将xyz作为特征的一部分。
            - normalize_xyz (bool): 是否在每个SA模块中以半径归一化xyz。
    """
    def __init__(self,
                 in_channels: int, # 点云的输入通道数
                 num_points: Tuple[int] = (2048, 1024, 512, 256), # 每个SA模块采样点数
                 radii: Tuple[Tuple[float, float, float]] = ( # 每个SA模块的采样半径 # SSG(0.2, 0.4, 0.8, 1.2)
                     (0.1, 0.2, 0.3),
                     (0.3, 0.4, 0.5),
                     (0.6, 0.8, 1.0),
                     (1.0, 1.2, 1.4)
                 ),
                 num_samples: TwoTupleIntType = ( # 每个SA模块进行球查询的样本数 # SSG(64, 32, 16, 16)
                                                 (64, 64, 64), 
                                                 (32, 32, 32),
                                                 (16, 16, 16),
                                                 (16, 16, 16)
                                                ),
                 sa_channels: ThreeTupleIntType = (             # 每个SA模块中mlp的输出通道数
                                                   (
                                                         (64, 64, 128),
                                                         (64, 64, 128),
                                                         (64, 64, 128)
                                                    ),
                                                   (
                                                         (128, 128, 256),
                                                         (128, 128, 256),
                                                         (128, 128, 256)
                                                    ),
                                                    (
                                                         (128, 128, 256),
                                                         (128, 128, 256),
                                                         (128, 128, 256)
                                                    ),
                                                   (
                                                         (128, 128, 256),
                                                         (128, 128, 256),
                                                         (128, 128, 256)
                                                    )
                                                    ),
                 fp_channels: Tuple[Tuple[int]] = ((256, 256), (256,256)), # 每个FP模块中每个多层感知机的输出通道数
                 aggregation_channels: Tuple[int] = (128, 256, 256, 256), # 聚合多尺度分组特征的输出通道数
                 fps_mods: TwoTupleStrType = (('D-FPS'), ('D-FPS'), ('D-FPS'), ('D-FPS')), # 每个SA模块的FPS模式
                 fps_sample_range_lists: TwoTupleIntType = ((-1), (-1), (-1), (-1)), # 每个SA模块采样点的数目
                 dilated_group: Tuple[bool] = (True, True, True, True), # 是否在每个SA模块中使用扩展球查询
                 out_indices: Tuple[int] = (2, ), # 输出的阶段
                 norm_cfg: dict = dict(type='BN2d'), # 归一化层的配置
                 sa_cfg: dict = dict( # 集合抽象模块的配置
                     type='PointSAModuleMSG',
                     pool_mod='max',
                     use_xyz=True,
                     normalize_xyz=True),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg=init_cfg) # 调用基类构造函数初始化模型配置

        self.num_sa = len(sa_channels) # 设置SA模块的数量
        self.num_fp = len(fp_channels) # FP数量

        self.out_indices = out_indices # 设置输出的索引
        
        assert max(out_indices) < self.num_sa # 确保输出索引有效
        assert len(num_points) == len(radii) == len(num_samples) == len(sa_channels) # 确保输入参数长度一致
        if aggregation_channels is not None:
            assert len(sa_channels) == len(aggregation_channels) # 确保聚合通道与SA模块数量匹配
        else:
            aggregation_channels = [None] * len(sa_channels) # 如果未提供聚合通道，设置为None

        self.SA_modules = nn.ModuleList() # 创建SA模块列表
        self.FP_modules = nn.ModuleList() # 初始化FP模块的列表
        self.aggregation_mlps = nn.ModuleList() # 创建聚合MLP列表

        sa_in_channel = in_channels - 3   # 计算不包含xyz的输入通道数
        skip_channel_list = [sa_in_channel] # 跳过通道列表

        # 将元组转换为列表
        radii = list(radii)
        num_samples = list(num_samples)


        # 遍历每个SA模块，配置和添加到模块列表
        for sa_index in range(self.num_sa):
            cur_sa_mlps = list(sa_channels[sa_index]) # 当前SA模块的MLP通道配置
            sa_out_channel = 0 # 初始化当前SA模块的输出通道数

            for radius_index in range(len(radii[sa_index])): # 遍历当前SA模块的每个半径配置
                cur_sa_mlps[radius_index] = [sa_in_channel] + list(cur_sa_mlps[radius_index]) # 更新MLP通道配置
                sa_out_channel += cur_sa_mlps[radius_index][-1] # 累加输出通道数

            if isinstance(fps_mods[sa_index], tuple): # 检查fps_mods是否为元组
                cur_fps_mod = list(fps_mods[sa_index]) # 转换为列表
            else:
                cur_fps_mod = list([fps_mods[sa_index]]) # 转换为列表

            if isinstance(fps_sample_range_lists[sa_index], tuple): # 检查fps_sample_range_lists是否为元组
                cur_fps_sample_range_list = list(
                    fps_sample_range_lists[sa_index] # 转换为列表
                )
            else:
                cur_fps_sample_range_list = list(
                    [fps_sample_range_lists[sa_index]] # 转换为列表
                )

            # 创建并添加SA模块到SA_modules列表
            self.SA_modules.append(
                build_sa_module(
                    num_point=num_points[sa_index], # SA模块的采样点数
                    radii=radii[sa_index], # SA模块的采样半径
                    sample_nums=num_samples[sa_index], # SA模块的采样数
                    mlp_channels=cur_sa_mlps, # SA模块的mlp通道
                    norm_cfg=norm_cfg, # 归一化配置
                    cfg=sa_cfg, # SA模块的额外配置
                    fps_mod=cur_fps_mod, ### 每个SA模块的FPS模式
                    fps_sample_range_list=cur_fps_sample_range_list, ### 每个SA模块采样点的数目 这里实际上不降采样
                    dilated_group=dilated_group[sa_index], ### 是否在每个SA模块中使用扩展球查询
                    bias=True ### 在每个卷积层中添加偏置项的设置
                )
            )

            skip_channel_list.append(sa_out_channel) # 添加当前SA模块的输出通道数到跳过列表

            cur_aggregation_channel = aggregation_channels[sa_index] # 获取当前聚合通道配置

            if cur_aggregation_channel is None: # 如果当前聚合通道为None
                self.aggregation_mlps.append(None) # 添加None到聚合MLP列表
                sa_in_channel = sa_out_channel # 更新SA输入通道为当前SA输出通道
            else:
                # 创建并添加聚合MLP到聚合MLP列表
                self.aggregation_mlps.append(
                    ConvModule(
                        sa_out_channel,
                        cur_aggregation_channel,
                        conv_cfg=dict(type='Conv1d'),
                        norm_cfg=dict(type='BN1d'),
                        kernel_size=1,
                        bias=True)
                    )
                sa_in_channel = cur_aggregation_channel # 更新SA输入通道为当前聚合通道

        # print("==================================", skip_channel_list)

        # skip_channel_list(128, 256, 256, 256)
        fp_channel_1 = [512, 256, 256]
        fp_channel_2 = [512, 256, 256]

        self.FP_modules.append(
            PointFPModule(mlp_channels=fp_channel_1)
        ) # 构建FP模块并添加到FP模块列表中

        self.FP_modules.append(
            PointFPModule(mlp_channels=fp_channel_2)
        ) # 构建FP模块并添加到FP模块列表中

    def forward(self, points: Tensor):
        """Forward pass.

        Args:
            points (torch.Tensor): point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            dict[str, torch.Tensor]: Outputs of the last SA module.

                - sa_xyz (torch.Tensor): The coordinates of sa features.
                - sa_features (torch.Tensor): The features from the
                    last Set Aggregation Layers.
                - sa_indices (torch.Tensor): Indices of the
                    input points.
        """
        """
        前向传播过程。

        参数:
            points (torch.Tensor): 点云数据，包含坐标和特征，形状为 (B, N, 3 + input_feature_dim)。

        返回:
            dict[str, torch.Tensor]: 最后一个SA模块的输出，包括:
                - sa_xyz (torch.Tensor): SA特征的坐标。
                - sa_features (torch.Tensor): 最后一层集合聚合层的特征。
                - sa_indices (torch.Tensor): 输入点的索引。
        """

        # 分离点坐标和特征
        xyz, features = self._split_point_feats(points)

        # 获取批次大小和点数
        batch, num_points = xyz.shape[:2]
        # 生成点索引
        indices = xyz.new_tensor(range(num_points)).unsqueeze(0).repeat(batch, 1).long()

        sa_xyz = [xyz] # 初始化SA坐标列表
        sa_features = [features] # 初始化SA特征列表
        sa_indices = [indices] # 初始化SA索引列表

        out_sa_xyz = [xyz]  # 初始化输出SA坐标列表
        out_sa_features = [features]  # 初始化输出SA特征列表
        out_sa_indices = [indices]  # 初始化输出SA索引列表

        # 遍历每个SA模块
        for i in range(self.num_sa):
            # SA_modules提取特征
            cur_xyz, cur_features, cur_indices = self.SA_modules[i](sa_xyz[i], sa_features[i])

            # 如果当前SA模块有聚合MLP，则通过聚合MLP处理特征
            if self.aggregation_mlps[i] is not None:
                cur_features = self.aggregation_mlps[i](cur_features)

            # 更新SA坐标、特征、索引列表
            sa_xyz.append(cur_xyz)
            sa_features.append(cur_features)
            sa_indices.append(torch.gather(sa_indices[-1], 1, cur_indices.long()))

            # 如果当前模块索引在输出索引列表中，则更新输出列表
            if i in self.out_indices:
                out_sa_xyz.append(sa_xyz[-1])
                out_sa_features.append(sa_features[-1])
                out_sa_indices.append(sa_indices[-1])


        """
            # 打印当前SA模块处理后的结果
            print(f"SA Module {i}:")
            print(f"  sa_xyz[{i+1}].shape: {sa_xyz[-1].shape}")
            print(f"  sa_features[{i+1}].shape: {sa_features[-1].shape}")
            print(f"  sa_indices[{i+1}].shape: {sa_indices[-1].shape}")

        # 最终输出前的汇总信息
        print("\nFinal output sizes:")
        print(f"  Total SA modules: {len(sa_xyz)}")
        for j in range(len(sa_xyz)):
            print(f"  sa_xyz[{j}].shape: {sa_xyz[j].shape}")
            print(f"  sa_features[{j}].shape: {sa_features[j].shape}")
            print(f"  sa_indices[{j}].shape: {sa_indices[j].shape}")

        aaa=[1,2,3]
        bbb=aaa[10]
        """
        fp_xyz = [sa_xyz[-1]] # 初始化FP层的点坐标列表，开始时只包含最后一个SA层的输出
        fp_features = [sa_features[-1]] # 初始化FP层的特征列表，开始时只包含最后一个SA层的输出
        fp_indices = [sa_indices[-1]] # 初始化FP层的索引列表，开始时只包含最后一个SA层的输出

        # 遍历所有FP模块，进行特征传播操作
        for i in range(self.num_fp):

            fp_features.append(
                self.FP_modules[i](
                sa_xyz[self.num_sa - i - 1], 
                sa_xyz[self.num_sa - i],
                sa_features[self.num_sa - i - 1], 
                fp_features[-1]
                )
            )

            fp_xyz.append(sa_xyz[self.num_sa - i - 1])
            fp_indices.append(sa_indices[self.num_sa - i - 1])
        # 返回最后一个SA模块的输出

        ret = dict( # 实际上fp_xyz等有三个第一个为最后一个SA的，sa_xyz等有五个，第一个为原始的
            fp_xyz=fp_xyz, # 特征传播后的点坐标 
            fp_features=fp_features, # 特征传播后的特征
            fp_indices=fp_indices, # 特征传播后的索引
            sa_xyz=sa_xyz, # 点集抽象后的点坐标
            sa_features=sa_features, # 点集抽象后的特征
            sa_indices=sa_indices # 点集抽象后的索引
        )

        return ret # 返回结果