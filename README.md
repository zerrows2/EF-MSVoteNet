165350.log 为训练日志，34轮时达到最佳。
ef-votenet_sunrgbd-3d-msg.py 为配置文件，可以看到我们的详细配置情况。
×3~×6为EF强化倍数不同时，EF-MSVoteNet算法的变化情况。
2D_ 为使用YOLOv9-E 以SUN RGB-D 2D train为训练集 val为验证集得到的数据。
我们的实验基于MMdetection3D框架开发，容易复现和使用。我们将在论文发表后公开包括Enhanced Frustum(EF)在内的全部代码组件。

165350.log is the training log, and it reached the best performance at the 34th epoch.
ef-votenet_sunrgbd-3d-msg.py is the configuration file, where you can see our detailed setup.
The variations of the EF-MSVoteNet algorithm with different EF enhancement factors ranging from ×3 to ×6. 
The 2D_ data is obtained using YOLOv9-E, trained on the SUN RGB-D 2D train dataset, and validated on the val dataset.
Our experiments are developed based on the MMDetection3D framework, making them easy to reproduce and use. We will release all code components, including the Enhanced Frustum (EF), after the paper is published.
