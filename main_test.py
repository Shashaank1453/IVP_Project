import argparse
import os, time, datetime
#import PIL.Image as Image
import numpy as np
from keras.models import load_model, model_from_json
# from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import peak_signal_noise_ratio , structural_similarity
from skimage.io import imread, imsave


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='data/Test', type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['Set12'], type=list, help='name of test dataset')
    parser.add_argument('--sigma', default=25, type=int, help='noise level')
    parser.add_argument('--model_dir', default=os.path.join('models','DnCNN_sigma25'), type=str, help='directory of the model')
    parser.add_argument('--model_name', default='model.h5', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of results')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    parser.add_argument('-testing_dir', default=os.path.join("data","Train400"), type=str, help='Will use it like testing set')
    return parser.parse_args()
    
def to_tensor(img):
    if img.ndim == 2:
        return img[np.newaxis,...,np.newaxis]
    elif img.ndim == 3:
        return np.moveaxis(img,2,0)[...,np.newaxis]

def from_tensor(img):
    return np.squeeze(np.moveaxis(img[...,0],0,-1))

def log(*args,**kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"),*args,**kwargs)

def save_result(result,path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt','.dlm'):
        np.savetxt(path,result,fmt='%2.4f')
    else:
        image_uint8 = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        imsave(path,image_uint8)


def show(x,title=None,cbar=False,figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x,interpolation='nearest',cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


if __name__ == '__main__':    
    
    args = parse_args()
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    
    psnr_avg_list = []
    ssim_avg_list = []

    for mod in os.listdir(args.model_dir):

        if (mod).endswith(".keras"):
            mod = mod.split(".")[0]
            if not os.path.exists(os.path.join(args.result_dir,mod)):
                os.mkdir(os.path.join(args.result_dir,mod))
            if not os.path.exists(os.path.join(args.result_dir,mod,"images")):
                os.mkdir(os.path.join(args.result_dir,mod,"images"))

            if not os.path.exists(os.path.join(args.model_dir, mod+".keras")):
                print(os.path.join(args.model_dir, mod))
                json_file = open(os.path.join(args.model_dir,'model.json'), 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                model = model_from_json(loaded_model_json)
                # load weights into new model
                model.load_weights(os.path.join(args.model_dir,'model.h5'))
                log('load trained model on Train400 dataset by kai')
            else:
                model = load_model(os.path.join(args.model_dir, mod+".keras"),compile=False)
                log('load trained model')
                
            psnrs = []
            ssims = [] 
            
            for im in os.listdir(args.testing_dir)[len(os.listdir(args.testing_dir))//2 : -1]: 
                if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                    x = np.array(imread(os.path.join(args.testing_dir,im)), dtype=np.float32) / 255.0
                    np.random.seed(seed=0) # for reproducibility
                    y = x + np.random.normal(0, args.sigma/255.0, x.shape) # Add Gaussian noise without clipping
                    y = y.astype(np.float32)
                    y_  = to_tensor(y)
                    start_time = time.time()
                    x_ = model.predict(y_) # inference
                    elapsed_time = time.time() - start_time
                    print('%10s : %10s : %2.4f second'%(mod,im,elapsed_time))
                    x_=from_tensor(x_)
                    psnr_x_ = peak_signal_noise_ratio(x, x_)
                    ssim_x_ = structural_similarity(x, x_,data_range=1)
                    if args.save_result:
                        name, ext = os.path.splitext(im)
                        # show(np.hstack((y,x_))) # show the image
                        save_result(np.hstack((y,x_)),path=os.path.join(args.result_dir,mod,"images",name+'_dncnn'+ext)) # save the denoised image
                    psnrs.append(psnr_x_)
                    ssims.append(ssim_x_)
            
                psnr_avg = np.mean(psnrs)
                ssim_avg = np.mean(ssims)
                psnrs.append(psnr_avg)
                ssims.append(ssim_avg)
                psnr_avg_list.append(psnr_avg)
                ssim_avg_list.append(ssim_avg)
                
                if args.save_result:
                    save_result(np.hstack((psnrs,ssims)),path=os.path.join(args.result_dir,f'{mod}_results.txt'))
                    
                log('Model Name: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(mod, psnr_avg, ssim_avg))
            if args.save_result:
                save_result(np.hstack((psnr_avg_list,ssim_avg_list)),path=os.path.join(args.result_dir,'results.txt'))
                
        


