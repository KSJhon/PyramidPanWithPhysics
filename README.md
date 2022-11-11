# PyramidPanWithPhysics

This repository contains a copy of the code used in the paper "P3Net: Pansharpening via Pyramidal Detail Injection with Deep Physical Constraints".

# Usage

# Credit
Jon, Kyongson and Liu, Jun and Deng, Liang-Jian and Zhu, Wensheng

# Abstract
Pansharpening is an image fusion process aiming to generate high-resolution multispectral (HRMS) images from a pair of low-resolution multispectral (LRMS) images and a high-resolution PAN image. It is a fundamental and significant task for the widespread use of remote sensing images. This paper proposes a new residual learning-based multispectral pansharpening network constrained by two deep physical models, collectively termed as P3Net. It mainly consists of the mainstream PDFNet and the other two auxiliary physical models, M2PNet and H2LNet. Unlike the existing methods of processing only one image scale, the proposed PDFNet fully extracts the spatial details from the multi-level image pyramid with decreasing spatial scales. Then, the spatial information is injected into the upsampled LRMS image. Since the pan-sharpened result should be consistent with the observed inputs under the physics models, we learn deep pansharpening physics models to reflect the inverse relationships. In detail, we propose the lightweight M2PNet and H2LNet to represent the latent non-linear mappings from the HRMS image to the panchromatic (PAN) image and the LRMS image, respectively. The two pre-trained physics models are frozen and guide the training of the PDFNet, so as to drive clear physical interpretability and further suppress the spectral and spatial distortions. The comparative experiments with the existing state-of-the-art pansharpening methods on QuickBird, GaoFen, and WorldView test sets demonstrate the superiority of the proposed method in terms of both quantitative metrics and subjective visual effects. 

# Citations
If you used this code, please kindly consider citing the following paper:

@article{jon2022P3Net, author={Jon, Kyongson and Liu, Jun and Deng, Liang-Jian and Zhu, Wensheng},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={P3Net: Pansharpening via Pyramidal Detail Injection With Deep Physical Constraints}, 
  year={2022},
  volume={60},
  number={},
  pages={1-18},
  doi={10.1109/TGRS.2022.3214209}
  }
