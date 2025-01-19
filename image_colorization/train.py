# from torchvision.extension import path_arr
# import config as c
# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from image_colorization.models import MainModel
# from image_colorization.dataset import make_dataloaders
#
# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # load train data from c.train_anaglyph_paths_file
# train_anaglyphs_paths = open(c.TRAIN_ANAGLYPH_FILE, 'r').read().splitlines()
# train_reversed_paths = open(c.TRAIN_REVERSED_FILE, 'r').read().splitlines()
#
# # load val data from c.val_anaglyph_paths_file
# val_anaglyphs_paths = open(c.VALIDATION_ANAGLYPH_FILE, 'r').read().splitlines()
# val_reversed_paths = open(c.VALIDATION_REVERSED_FILE, 'r').read().splitlines()
#
# # load test data from c.test_anaglyph_paths_file
# test_anaglyphs_paths = open(c.TEST_ANAGLYPH_FILE, 'r').read().splitlines()
# test_reversed_paths = open(c.TEST_REVERSED_FILE, 'r').read().splitlines()
#
# # create dataloaders
# train_dl = make_dataloaders(path_anaglyph=train_anaglyphs_paths, path_reversed=train_reversed_paths, split='train')
# val_dl = make_dataloaders(path_anaglyph=val_anaglyphs_paths, path_reversed=val_reversed_paths, split='val')
#
#
#
#
# # train model
