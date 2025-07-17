class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self):
        super(BatchCollator, self).__init__()

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = transposed_batch[0]
        masks = transposed_batch[1]
        captions = transposed_batch[2]

        return images, masks, captions
