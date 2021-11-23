from Dataset import MultiClassDataset

dataset_train = MultiClassDataset('mnist_train.csv', dict([(str(i), i) for i in range(10)]), 0.5)
dataset_train.normalize_data(lambda x: x/255)
dataset_train.training_to_csv('mnist_train_half.csv')