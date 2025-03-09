# Robust In-Hand Reorientation with Hierarchical RL-Based Motion Primitives and Model-Based Regrasping

<p style="text-align: center;"> 
<!-- <a href="https://arxiv.org/abs/2502.07472" style="color: #0ABAB5; text-decoration: underline;">arXiv</a> | -->
<a href="https://github.com/RGMC-XL-team/inhand_reorientation" style="color: #0ABAB5; text-decoration: underline;">Code</a> |
<a href="RAP_Appendix.pdf" style="color: #0ABAB5; text-decoration: underline;">Appendix</a> |
<a href="https://drive.google.com/drive/folders/1GN-KxQTjhRVIoPXXx9aKyV5DzTSBRsf9?usp=sharing" style="color: #0ABAB5; text-decoration: underline;">Dataset</a> |
<a href="https://drive.google.com/drive/folders/1wRGaV5bB4EGYVtHWL2-WbT_hgUvj4KIQ?usp=sharing" style="color: #0ABAB5; text-decoration: underline;">CAD Files</a>
</p>

The proposed approach won the **championship** of the in-hand manipulation track of the [9th Robotic Grasping and Manipulation Competition (RGMC)](https://cse.usf.edu/~yusun/rgmc/2024.html) held at ICRA 2024. Additionally, it was awarded the **Most Elegant Solution** among all tracks of the RGMC.

The paper has been submitted to IEEE RA-P.

## Video

<!-- <video controls style="width: 100%; height: auto;">
    <source src="https://youtu.be/bOGBPTh_lsU?si=lXdxABpvdsCi3GbI" type="video/mp4">
</video> -->


<div style="text-align: center;">
<iframe width="720" height="405" src="https://www.youtube.com/embed/okt8-gXMCkc?si=6Ss2AWQZc84V-NXU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

## Abstract

In-hand manipulation has become increasingly popular in recent robotics research, probably due to the growing trend towards humanoids and Artificial General Intelligence (AGI). Although existing works show promising results, they are typically limited to laboratory conditions. The requirements of expensive dexterous hands, depth cameras and tactile sensors are challenging for algorithm reproduction and large-scale applications. To address this issue, this paper proposes a practical solution to the classic in-hand reorientation task. The proposed method is characterized by a hierarchical structure. Specifically, several in-hand motion primitives, such as object rotation and flipping, are trained using Reinforcement Learning (RL). A high-level decision module switches between these motion primitives to achieve continuous in-hand reorientation. The proposed method runs on a low-cost LEAP Hand and requires only a single RGB camera. The proposed method is validated on a cube reorientation task benchmarked at the 9th Robotic Grasping and Manipulation Competition (RGMC) at ICRA 2024. Implementation details and evaluation results are discussed in this paper. We also open-source hardware designs, code, and videos to encourage further development in this area.

## Contact

If you have any question, feel free to contact the authors: Yongpeng Jiang, [jiangyp19@gmail.com](mailto:jiangyp19@gmail.com) .

Yongpeng Jiang's Homepage is at [https://director-of-g.github.io/](https://director-of-g.github.io/).
