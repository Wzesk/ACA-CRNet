folder for mounting training data
gcsfuse --implicit-dirs littoral-cloud-training/rgb_train coastal-datasets
fusermount -u coastal-datasets