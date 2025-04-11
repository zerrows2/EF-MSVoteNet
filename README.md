我们的论文“Enhanced frustum multi-scale VoteNet for 3D object detection in cluttered indoor scene” 已被 Applied Intelligence (SCI, JCR Q2) 接受。DOI：10.1007/s10489-025-06492-4 🎉

Our paper "Enhanced frustum multi-scale VoteNet for 3D object detection in cluttered indoor scene" has been accepted by Applied Intelligence (SCI, JCR Q2). DOI: 10.1007/s10489-025-06492-4 🎉

如果你使用我们的代码，请考虑应用我们的论文
If you use our code, please consider applying our paper

@article{Zhang2025EFMSVoteNet,
  author    = {Xiaoyu Zhang and Yu He and Chao Song and others},
  title     = {Enhanced frustum multi-scale VoteNet for 3D object detection in cluttered indoor scene},
  journal   = {Applied Intelligence},
  volume    = {55},
  pages     = {588},
  year      = {2025},
  doi       = {10.1007/s10489-025-06492-4}
}

APA : Zhang, X., He, Y., Song, C., & et al. (2025). Enhanced frustum multi-scale VoteNet for 3D object detection in cluttered indoor scene. Applied Intelligence, 55, 588. https://doi.org/10.1007/s10489-025-06492-4

IEEE : X. Zhang, Y. He, C. Song et al., "Enhanced frustum multi-scale VoteNet for 3D object detection in cluttered indoor scene," Applied Intelligence, vol. 55, p. 588, Mar. 2025. doi: 10.1007/s10489-025-06492-4


165350.log 为训练日志，34轮时达到最佳。
ef-votenet_sunrgbd-3d-msg.py 为配置文件，可以看到我们的详细配置情况。
×3~×6为EF强化倍数不同时，EF-MSVoteNet算法的变化情况。
2D_ 为使用YOLOv9-E 以SUN RGB-D 2D train为训练集 val为验证集得到的数据。
我们的实验基于MMdetection3D框架开发，容易复现和使用。

165350.log is the training log, and it reached the best performance at the 34th epoch.
ef-votenet_sunrgbd-3d-msg.py is the configuration file, where you can see our detailed setup.
The variations of the EF-MSVoteNet algorithm with different EF enhancement factors ranging from ×3 to ×6. 
The 2D_ data is obtained using YOLOv9-E, trained on the SUN RGB-D 2D train dataset, and validated on the val dataset.
Our experiments are developed based on the MMDetection3D framework, making them easy to reproduce and use. 

我们采用的实验数据为SUN RGB-D数据，数据集以及数据集工具包的下载请参照MMdetection3d:https://github.com/open-mmlab/mmdetection3d/blob/main/data/sunrgbd/README.md
The experimental data we use is from the SUN RGB-D dataset. For downloading the dataset and toolkit, please refer to MMDetection3D: https://github.com/open-mmlab/mmdetection3d/blob/main/data/sunrgbd/README.md.

关于部署我们的实验，有两个方法：
（1）使用autodl服务器，我们将直接分享镜像给您，这样就可以直接复现基于MM3D开发的EF-MSVoteNet关于SUN RGB-D的检测部分。如有需要，请email联系论文作者。——我们推荐这个方法，可以快速帮你部署模型。
（2）使用Github上提供的组件，直接替换MM3D中对应的部分，这样也可以实现模型的部署。——这个方法有一定难度，但是也不是很困难。

由于我不会在Github上创建文件夹，我只能将重要的两个散件EF以及MSVoteNet中的改进部分和配置文件直接上传，如果要使用（2）进行部署，则需要找MM3D中对应的文件并实现替换。具体文件有：
（1）将2D检测文件sunrgbd_rgb_train_yolov9-e_confThres0.362_IouThres0.45和sunrgbd_rgb_val_yolov9-e_confThres0.362_IouThres0.45放到正确位置。所谓正确位置，就是sunrgbd_data_utils.py中，self.train_bbox_list = 以及self.val_bbox_list = 后的位置，你也可以修改这个位置。
（2）使用sunrgbd_data_utils.py替换原来的/root/mmdetection3d/tools/dataset_converters/sunrgbd_data_utils.py，装载EF模块
（3）将votenet_8xb16_sunrgbd-3d-msg.py放到/root/mmdetection3d/configs/votenet/中，将votenet_msg.py放到/root/mmdetection3d/configs/_base_/models/中。
（4）将pointnet2_sa_msg.py放到/root/mmdetection3d/mmdet3d/models/backbones/中。装载MS模块。注意我们直接将FP×2固定在了骨干中，而未设定新的颈部网络，如果需要请自行修改。


Regarding the deployment of our experiment, there are two methods:
(1) Using the AutoDL server
We will directly share the image with you, which allows you to immediately reproduce the SUN RGB-D detection results of EF-MSVoteNet based on MMDetection3D.
If needed, please contact the authors via email.
— We recommend this method as it enables quick deployment of the model.


(2) Using the components provided on GitHub
You can directly replace the corresponding parts in MMDetection3D with our components to deploy the model.
— This method is more challenging, but still manageable.

Since I'm not able to create folders on GitHub, I have uploaded the two key modules — EF and the improvements to MSVoteNet — along with the necessary configuration files. If you choose to deploy using method (2), you will need to find and replace the corresponding files in MMDetection3D. The specific steps are:

Place the 2D detection files sunrgbd_rgb_train_yolov9-e_confThres0.362_IouThres0.45 and sunrgbd_rgb_val_yolov9-e_confThres0.362_IouThres0.45 in the correct location.
The correct location refers to the paths assigned in sunrgbd_data_utils.py, under self.train_bbox_list = and self.val_bbox_list =. You can also modify these paths manually if needed.

Replace the original /root/mmdetection3d/tools/dataset_converters/sunrgbd_data_utils.py with the provided sunrgbd_data_utils.py to enable the EF module.

Place votenet_8xb16_sunrgbd-3d-msg.py into /root/mmdetection3d/configs/votenet/, and
place votenet_msg.py into /root/mmdetection3d/configs/_base_/models/.

Place pointnet2_sa_msg.py into /root/mmdetection3d/mmdet3d/models/backbones/ to enable the MS module.
Note that we have fixed the FP×2 structure in the backbone and have not configured a new neck network. If needed, please modify it accordingly.

PS : 注意要先安装MMdetection3D再使用我们的代码，同时需要使用显存≥32GB的显卡，因为多尺度结构导致需要更高的显存。

P.S.: Please make sure to install MMDetection3D before using our code. Additionally, a GPU with at least 32GB of memory is required, as the multi-scale structure demands higher memory capacity.
