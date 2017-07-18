class DatasetContainer(object):

    train_dataset = None
    validation_dataset = None
    test_dataset = None

    license_agreement = ''

    def __init__(self):
        return

    def set_train_dataset(self, train):
        self.train = train

    def get_train_dataset(self):
        return self.train_dataset

    def set_validation_dataset(self, validation):
        self.validation = validation

    def get_validation_dataset(self):
        return self.validation_dataset

    def set_test_dataset(self, test):
        self.test = test

    def get_test_dataset(self):
        return self.test_dataset

    def set_license(self, license):
        self.license_agreement = license

    def get_license(self):
        return self.license_agreement