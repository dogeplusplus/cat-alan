# cat-alan

Classification of audio files of domestic cats to 1 of 10 intents. (Warning, Angry, Defence, Fighting, Happy, Hunting, Mating, Mother Call, Paining, Resting).
Model trains on the raw waveforms using the M5 architecture written in PyTorch. Applied time stretching, pitch shifting and Gaussian noise as augmentation.

Credit to the origin of the dataset and augmentation techniques is given to:
*Domestic Cat Sound Classification Using Transfer Learning*
*Yagya Raj Pandeya, Dongwhoon Kim and Joonwhoan Lee*
*https://doi.org/10.3390/app8101949*
