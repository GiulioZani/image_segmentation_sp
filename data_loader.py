from new_data_loader import DataLoader


def get_generators():
    return DataLoader("train", squeeze=True).get_loaders()
