
import argparse
import re
import os, glob, datetime
import numpy as np
from keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract
from keras.models import Model, load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, Callback
from keras.optimizers import Adam
import data_generator as dg
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

history_file = os.path.join(os.getcwd(), 'output', 'training_history.txt')  # Define your history file path

class TrainingHistoryCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            with open(history_file, 'a') as f:
                f.write(f"Epoch {epoch + 1}\n")
                for key, value in logs.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")  # Newline between epochs for readability

## Params
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--train_data', default='data/Train400', type=str, help='path of train data')
parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epoch', default=40, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=1, type=int, help='save model at every x epoches')
args = parser.parse_args()


save_dir = os.path.join('models',args.model+'_'+'sigma'+str(args.sigma)) 

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def DnCNN(depth,filters=64,image_channels=1, use_bnorm=True):
    layer_count = 0
    inpt = Input(shape=(None,None,image_channels),name = 'input'+str(layer_count))
    # 1st layer, Conv+relu
    layer_count += 1
    x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),kernel_initializer='Orthogonal', padding='same',name = 'conv'+str(layer_count))(inpt)
    layer_count += 1
    x = Activation('relu',name = 'relu'+str(layer_count))(x)
    # depth-2 layers, Conv+BN+relu
    for i in range(depth-2):
        layer_count += 1
        x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),kernel_initializer='Orthogonal', padding='same',use_bias = False,name = 'conv'+str(layer_count))(x)
        if use_bnorm:
            layer_count += 1
            #x = BatchNormalization(axis=3, momentum=0.1,epsilon=0.0001, name = 'bn'+str(layer_count))(x) 
        x = BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'bn'+str(layer_count))(x)
        layer_count += 1
        x = Activation('relu',name = 'relu'+str(layer_count))(x)  
    # last layer, Conv
    layer_count += 1
    x = Conv2D(filters=image_channels, kernel_size=(3,3), strides=(1,1), kernel_initializer='Orthogonal',padding='same',use_bias = False,name = 'conv'+str(layer_count))(x)
    layer_count += 1
    x = Subtract(name = 'subtract' + str(layer_count))([inpt, x])   # input - noise
    model = Model(inputs=inpt, outputs=x)
    
    return model


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir,'model_*.hdf5'))  # get name list of all .hdf5 files
    #file_list = os.listdir(save_dir)
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).hdf5.*",file_)
            #print(result[0])
            epochs_exist.append(int(result[0]))
        initial_epoch=max(epochs_exist)   
    else:
        initial_epoch = 0
    return initial_epoch

def log(*args,**kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"),*args,**kwargs)

def lr_schedule(epoch):
    initial_lr = args.lr
    if epoch<=30:
        lr = initial_lr
    elif epoch<=60:
        lr = initial_lr/10
    elif epoch<=80:
        lr = initial_lr/20 
    else:
        lr = initial_lr/20 
    log('current learning rate is %2.8f' %lr)
    return lr

def train_datagen(epoch_num=5,batch_size=256,data_dir=args.train_data):
    while(True):
        n_count = 0
        if n_count == 0:
            #print(n_count)
            xs = dg.datagenerator(data_dir)
            assert len(xs)%args.batch_size ==0, \
            log('make sure the last iteration has a full batchsize, this is important if you use batch normalization!')
            xs = xs.astype('float32')/255.0
            indices = list(range(xs.shape[0]))
            n_count = 1
        for _ in range(epoch_num):
            np.random.shuffle(indices)    # shuffle
            for i in range(0, len(indices), batch_size):
                batch_x = xs[indices[i:i+batch_size]]
                noise =  np.random.normal(0, args.sigma/255.0, batch_x.shape)
                batch_y = batch_x + noise 
                yield batch_y, batch_x

def sum_squared_error(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true))/2

def findLastCheckpoint(save_dir):
    files = [f for f in os.listdir(save_dir) if f.startswith('model_') and f.endswith('.keras')]
    epochs = [int(f.split('_')[1].split('.')[0]) for f in files]
    return max(epochs) if epochs else 0

class ClearSessionCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        K.clear_session()
    
if __name__ == '__main__':
    # model selection
    save_dir = 'output'  # Define your model save directory

    # Model selection
    model = DnCNN(depth=17, filters=64, image_channels=1, use_bnorm=True)
    model.summary()

    # Load the last model in matconvnet style
    initial_epoch = findLastCheckpoint(save_dir=save_dir)
    if initial_epoch > 0:
        print('Resuming by loading epoch %03d' % initial_epoch)
        model = load_model(os.path.join(os.getcwd(), save_dir, 'model_%03d.keras' % initial_epoch), compile=False)

    # Compile the model
    model.compile(optimizer=Adam(0.001), loss=sum_squared_error)

    # Use callback functions
    checkpointer = ModelCheckpoint(
        os.path.join(save_dir, 'model_{epoch:03d}.keras'), 
        verbose=1, save_weights_only=False,save_format='h5'
    )
    csv_logger = CSVLogger(os.path.join(os.getcwd(), save_dir, 'log.csv'), append=True, separator=',')
    lr_scheduler = LearningRateScheduler(lr_schedule)
    clear_session_callback = ClearSessionCallback()
    history_callback = TrainingHistoryCallback()

    steps_per_epoch = len(dg.datagenerator()) // args.batch_size
    history = model.fit(
        train_datagen(batch_size=args.batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=args.epoch,
        verbose=1,
        initial_epoch=initial_epoch,
        callbacks=[checkpointer, csv_logger, lr_scheduler,clear_session_callback, history_callback]
    )

    # Save loss and accuracy graphs
    print(history.__dir__())
    if 'loss' in history.history:
        # Plot training loss
        plt.figure()
        plt.plot(history.history['loss'], label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.savefig(os.path.join(os.getcwd(), save_dir, 'training_loss.png'))
        plt.close()

    history_file = os.path.join(os.getcwd(), save_dir, 'training_history_complete.txt')
    with open(history_file, 'w') as f:
        for key, values in history.history.items():
            f.write(f'{key}: {values}\n')
