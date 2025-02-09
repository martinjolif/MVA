# Project for the Time-Series course

Project in pairs for the Machine Learning for Time series course.

Study of the following ["A review of Unsupervised Feature Learning and Deep
Learning for Time-Series Modeling"](https://www.researchgate.net/publication/260086856_A_Review_of_Unsupervised_Feature_Learning_and_Deep_Learning_for_Time-Series_Modeling) paper.
The authors cite this paper ["Learning features from music audio with Deep Belief Networks"](https://ismir2010.ismir.net/proceedings/ismir2010-58.pdf) which caught our attention, they present a Deep Belief Network (DBN) that learn features
from music signal. Then these features can be used to solve music genre classification.

We implemented three methods to learn features from music signal, and use them to do music genre classification on the GTZAN dataset:

- A Deep Belief Network (DBN)
- A Convolutional Restricted Boltzmann Machine (ConvRBM)
- A Variational autoencoder with convolutional layers (ConvVAE)

