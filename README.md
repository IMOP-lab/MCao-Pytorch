# MCao: Multi-Branch Coronary Artery Occlusion Localization Using Real-Imaginary Enhancement Fourier Wavelet-KAN

<div align=left>
  <img src="https://github.com/IMOP-lab/MCao-Pytorch/blob/main/figures/MCao-Net.png"width=80% height=80%>
</div>
<p align=left>
  Figure 1: Detailed network structure of our proposed MCao.
</p>

<div align=left>
  <img src="https://github.com/IMOP-lab/MCao-Pytorch/blob/main/figures/FrameWork.png"width=80% height=80%>
</div>
<p align=left>
  Figure 2: The system structure of our proposed CAD localization architecture.
</p>

The proposed MCao-Net is an advanced ECG-based diagnostic model tailored for localizing coronary artery lesions. This network combines multi-branch feature extraction with the Real-Imaginary Enhancement Fourier Neural Operator (RieFNO) and the wavelet-KAN attention module (wKAN) to address the complexities of ECG signals. It outperforms 12 other recent methods on the CardioLead-CAD and PTB datasets, establishing state-of-the-art accuracy in identifying key coronary artery occlusions.

We will first introduce our method and underlying principles, explaining how MCao-Net uses specialized branches and attention mechanisms to improve feature extraction from ECG signals. Next, we provide details on the experimental setup, performance metrics, and GitHub links to previous methods used for comparison. Finally, we present the experimental results, showing how MCao-Net achieves high performance across multiple datasets.

## Installation
We run SASAN and previous methods on a system running Ubuntu 22.04, with Python 3.9, PyTorch 2.0.0, and CUDA 11.8.

## Experiment

### Compare with others on the CardioLead-CAD and PTB dataset

<div align=left>
  <img src="https://github.com/IMOP-lab/MCao-Pytorch/blob/main/tables/CardioLead-CAD.png">
</div>
<p align=left>
  Figure 3: Comparison of ECG detection performance between MCao-Net and other methods on the CardioLead-CAD dataset.
</p>

<div align=left>
  <img src="https://github.com/IMOP-lab/MCao-Pytorch/blob/main/tables/PTB.png">
</div>
<p align=left>
  Figure 4: Comparison of ECG detection performance between MCao-Net and other methods on the PTB dataset.
</p>

Our method demonstrates the best performance in accuracy and sensitivity for coronary artery lesion localization, surpassing previous models on both the CardioLead-CAD and PTB datasets. The integration of the Real-Imaginary Enhancement Fourier Neural Operator (RieFNO) and wavelet-KAN attention module (wKAN) significantly enhances MCao-Net’s capability to detect rare and complex lesion patterns, thereby improving its precision in ECG-based coronary artery disease diagnosis.

### Ablation study

#### Branch Ablation Study

<div align=left>
  <img src="https://github.com/IMOP-lab/MCao-Pytorch/blob/main/tables/Ablation study of branch-specific.png">
</div>
<p align=left>
  Figure 5: Ablation study assessing the impact of individual branch-specific enhancements on network performance in the CardioLead-CAD dataset.
</p>

#### Wavelet-Based wKAN Module Ablation Study
<div align=left>
  <img src="https://github.com/IMOP-lab/MCao-Pytorch/blob/main/tables/mother wavelet.png">
</div>
<p align=left>
  Figure 6: Ablation analysis of the effectiveness of different mother wavelets in the wKAN module using the CardioLead-CAD dataset.
</p>

#### RieFNO Ablation Study Across Branches

<div align=left>
  <img src="https://github.com/IMOP-lab/MCao-Pytorch/blob/main/tables/Ablation study of RieFNO.png">
</div>
<p align=left>
  Figure 7: Ablation study of the RieFNO module across different feature extraction branches in the MCao-Net architecture.
</p>

#### wKAN Ablation Study Across Branches

<div align=left>
  <img src="https://github.com/IMOP-lab/MCao-Pytorch/blob/main/tables/Ablation study of wKAN.png">
</div>
<p align=left>
  Figure 8: Ablation study on the performance impact of removing the wKAN module from specific feature extraction branches in MCao-Net.
</p>

#### Loss Function Ablation Study

<div align=left>
  <img src="https://github.com/IMOP-lab/MCao-Pytorch/blob/main/tables/Ablation study of Loss.png">
</div>
<p align=left>
  Figure 9: Effect of different loss function variants on the performance of the MCao-Net model in ECG signal classification.
</p>
