# Assignments and Projects from the Master 2 Mathématiques, Vision, Apprentissage (MVA) at ENS Paris-Saclay

Some courses are not yet here.

### First Semester 

**Advanced learning for text and graph data (ALTEGRAD)** by M. Vazirgiannis:
* Lab 1: Hierarchical Attention Network
* Lab 2: Transfert Learning
* Lab 3: LLM finetuning with DPO
* Lab 4: Machine Learning for graph clustering/classification
* Lab 5: Deep Walk & Graph Neural Network (GNN) for node classification
* Lab 6: Graph Attention Network (GAT) & GNN for graph and node classification
* Lab 7: DeepSets & Graph generation with Variational Graph Autoencoders
* **Project**: Neural Graph Generation with conditioning ([kaggle challenge](https://www.kaggle.com/competitions/generating-graphs-with-specified-properties/overview))

Librairies used: PyTorch, transformers, networkx, torch_geometric, grakel, scipy, scikit-learn.\
Use of Huggingface.

**Geometry processing and geometric deep learning (Geometry)** by E. Corman, J. Digne and M. Ovsjanikov:\
[Course page](https://jdigne.github.io/mva_geom/)
* Lab 2: Surface parametrisation and alignement 
* Lab 4: Implementation of PointNet for 3D shape classification and segmentation
* Lab 6: Shape reconstruction and interpolation
* **Project**: Study of the ["Single Mesh Diffusion Models with Field Latents for Texture Generation"](https://single-mesh-diffusion.github.io/) article.

Librairies used: PyTorch, meshplot, scipy.

**Introduction to Probabilistic Graphical Models and Deep Generative Models (PGM)** by P. Latouche and P-A. Mattei:\
[Course page](https://lmbp.uca.fr/~latouche/mva/IntroductiontoProbabilisticGraphicalModelsMVA.html)

* **Project**: Denoising score matching for diffusion models. Study of the ["Denoising Diffusion Probabilistic Models
"](https://arxiv.org/pdf/2006.11239) paper.

Librairies used: PyTorch.\
Use of Huggingface.

**Reinforcement Learning (RL)** by E. Rachelson and C. Vernade:\
[Course page](https://erachelson.github.io/RLclass_MVA/)

* **Project**: The goal is to goal is to design a control strategy which keeps the patient healthy while avoiding prescribing drugs at every time step. See this [repository](https://github.com/RL-MVA-2024-25/mva-rl-assignment-martinjolif) for more details.

Librairies used: gymnasium, scikit-learn, PyTorch, Numpy.

**Robotics** by S. Caron, J. Carpentier, S. Bonnabel, P.B. Wieber:\
[Course page](https://scaron.info/robotics-mva/)\
[Github for assignements](https://github.com/AjSat/2024_mva_robotics_exercises)
* Lab 1: Introduction to pinocchio
* Lab 2: Forward and inverse geometry
* Lab 3: Inverse kinematics
* Lab 4: Robot control systems
* Lab 5: Reinforcement learning
* Lab 6: Contact simulation
* **Project**: Study of the ["Towards Generalizable Vision-Language Robotic Manipulation: A Benchmark and LLM-guided 3D Policy"](https://www.di.ens.fr/willow/research/gembench/) article. See both forked repository ([robot-3dlotus](https://github.com/martinjolif/robot-3dlotus/tree/main) / [RLBench](https://github.com/martinjolif/RLBench)) or this [tutorial](https://github.com/martinjolif/robot-3dlotus/blob/main/create_evaluate_new_task.md) to understand more about it and the work done.

Librairies used: pinocchio, gymnasium.

**Machine learning for Time Series** by L. Oudre: \
[Course page](https://www.laurentoudre.fr/ast.html)
* Lab 1: Convolutional Dictionary Learning (CDL), Spectral feature and Dynamic Time Warping (DTW)
* Lab 2: ARIMA Process, Sparse Coding
* Lab 3: Change-Point Detection, Wavelet Transform for Graph Signals
* **Project**: Unsupervised feature learning for time-series modeling.

Librairies used: scikit-learn, scipy, ruptures, pandas.

### Second Semester 

**Generative Modeling** by B. Galerne and A. Leclaire:\
[Course page](https://generativemodelingmva.github.io/)
* Assignement: GAN, WGAN and VAE for image generation.
* **Project**: Solving imaging inverse problems like deblurring, inpainting and super-resolution from [PnP-SGS](https://arxiv.org/pdf/2304.11134) algorithm using [DDPM](https://arxiv.org/pdf/2006.11239).

Librairies used: PyTorch, scikit-image, Pillow.

**Graphs in Machine Learning** by D. Calandriello:
* Lab 1: Spectral Clustering
* Lab 2: Semi Supervised Learning
* Lab 3: Graph Neural Networks (GNNs)

Librairies used: scikit-learn, networkx, jax, scipy.

**3D Point Cloud and Modeling (NPM3D)** by F. Goulette, J-E. Deschaud, T. Boubekeur:\
[Course page](https://www.caor.minesparis.psl.eu/presentation/cours-npm3d/)
* Lab 1: Basic operations and structures on point clouds
* Lab 2: Iterative Closest Point (ICP)
* Lab 3: Neighborhood descriptors
* Lab 4: Surface reconstruction
* Lab 5: Modeling (RANSAC algorithm)
* Lab 6: Deep learning (PointNet)
* **Project**: 3D point cloud semantic segmentation with [SPT](https://arxiv.org/pdf/2306.08045) model.

Librairies used: Numpy, scikit-learn, PyTorch. \
Use of CloudCompare.

**Representation Learning for Computer Vision** by P. Gori and L. Le Folgoc:\
[Course page](https://perso.telecom-paristech.fr/pgori/teaching/RepLearnMVA.html)

* Lab 1: Adversarial examples - blind spot in representation spaces
* Lab 2: Domain adaptation
* Lab 3: Self-supervised learning
* Lab 4: Contrastive-learning pre-training
* Lab 5: Vision Transformers (ViT)
* Lab 6: Masked-Auto-Encoder - Self-Supervised Training of ViT
* Lab 7: Masked Generative Image Transformer (MaskGIT) autoencoder
* Lab 8: Interpretability and Explainability on MedMNIST

Librairies used: PyTorch, scikit-image, scikit-learn, einops, Pillow.