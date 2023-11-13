# This file is used to configure the training or testing parameters for each task
class Config_BCIHM:
    # This dataset is for intracranial hemorrhage segmentation
    data_path = "./data/to/BCIHM/"
    save_path = "./checkpoints/to/BCIHM/"
    tensorboard_path = "./tensorboard/BCIHM/"
    load_path = ''
    save_path_code = "_"

    workers = 2                         # data loading workers (default: 8)
    epochs = 200                        # total training epochs (default: 400)
    batch_size = 2                      # batch size (default: 4)
    learning_rate = 1e-4                # initial learning rate (default: 0.001)
    momentum = 0.9                      # momentum
    classes = 2                         # the number of classes (background + foreground)
    img_size = 512                      # the input size of model
    train_split = "train"               # the file name of training set
    val_split = "val"                   # the file name of testing set
    test_split = "test"                 # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"            # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "test"
    visual = False
    modelname = "SAMIHS"

class Config_Intance:
    # This dataset is for intracranial hemorrhage segmentation
    data_path = "./data/to/Instance/"
    save_path = "./checkpoints/to/Instance/"
    tensorboard_path = "./tensorboard/Instance/"
    load_path = ''
    save_path_code = "_"

    workers = 2                         # data loading workers (default: 8)
    epochs = 200                        # total epochs to run (default: 400)
    batch_size = 2                      # batch size (default: 4)
    learning_rate = 1e-4                # initial learning rate (default: 0.001)
    momentum = 0.9                      # momentum
    classes = 2                         # the number of classes (background + foreground)
    img_size = 512                      # the input size of model
    train_split = "train"               # the file name of training set
    val_split = "val"                   # the file name of testing set
    test_split = "test"                 # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"            # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "test"
    visual = False
    modelname = "SAMIHS"


# ==================================================================================================
def get_config(task="BCIHM"):
    if task == "BCIHM":
        return Config_BCIHM()
    elif task == "Instance":
        return Config_Intance()
    else:
        assert("We do not have the related dataset, please choose another task.")