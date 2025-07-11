import time


class Config:
    labels = ["pre1935", "1935-1955", "1956-1971", "1972-1990", "post1990"]

    num_classes = len(labels)

    image_size = 224

    input_shape = [image_size, image_size, 3]

    lr_max = 5e-5  # maximum learning rate

    lr_min = 1e-6  # minimum learning rate, note you can set lr_min = lr_max to use constant learning rate

    optimizer = "adam"  # 'adam' or 'sgd'

    early_stop_patience = 10

    transfer_learning = 1  # 1 if transfer learning is used, 0 otherwise

    batch_size = 64  # higher batch size is faster but requires more memory

    num_epochs = 100  # max number of epochs

    loss = "crossentropy"

    val = 1  # 1 if validation input is available, 0 otherwise

    saved_model_dir = "weights/run3/"  # leave empty to defaul model

    in_dir = "input/run3/"

    timestr = str(time.strftime("%Y%m%d-%H%M%S"))

    out_dir = "output/" + str(timestr) + "/"
