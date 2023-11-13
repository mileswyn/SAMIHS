# This file is used to configure the training or testing parameters for each task
class Config_BCIHM:
    # This dataset is for breast cancer segmentation
    data_path = "/data/openData_Med/BCIHM/"
    save_path = "./checkpoints/BCIHM_SAMICH_v7_fold_0/"
    tensorboard_path = "./tensorboard/BCIHM/"
    load_path = '/data/wyn/SAMUS/checkpoints/SAMICH_fold1/SAMICH_10191710_30_0.6611.pth'
    save_path_code = "_"

    workers = 2                         # number of data loading workers (default: 8)
    epochs = 200                        # number of total epochs to run (default: 400)
    batch_size = 2                      # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                        # thenumber of classes (background + foreground)
    img_size = 512                      # theinput size of model
    train_split = "train"   # the file name of training set
    val_split = "val"       # the file name of testing set
    test_split = "test"     # the file name of testing set # HMCQU
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "test"
    visual = False
    modelname = "SAMIHS"

class Config_Intance:
    # This dataset is for breast cancer segmentation
    data_path = "/data/openData_Med/Instance/"
    save_path = "./checkpoints/Instance_SAMICH_v7_fold_4/"
    tensorboard_path = "./tensorboard/Instance/"
    load_path = '/data/wyn/SAMUS/checkpoints/Instance_SAMICH_fold1/SAMICH_10221922_49_0.7619.pth'
    save_path_code = "_"

    workers = 2                         # number of data loading workers (default: 8)
    epochs = 200                        # number of total epochs to run (default: 400)
    batch_size = 2                      # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                        # thenumber of classes (background + foreground)
    img_size = 512                      # theinput size of model
    train_split = "train"   # the file name of training set
    val_split = "val"       # the file name of testing set
    test_split = "test"     # the file name of testing set # HMCQU
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
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