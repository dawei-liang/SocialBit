

"""
Adopted from https://github.com/anuragkr90/weak_feature_extractor, by Kumar et al., ICASSP 18

Load segmented wav, compute segment spectrogram, extract embedding features, and save as csv
"""


import librosa as lib
import numpy as np
import network_architectures as netark
import torch.nn.functional as Fx
import torch
from torch.autograd import Variable
import sys, os, csv
from collections import OrderedDict
import extractor as exm


usegpu = False


n_fft = 1024
hop_length = 512
n_mels = 128
srate=16000
trainType = 'weak_mxh64_1024'
pre_model_path = 'mx-h64-1024_0d3-1.17.pkl'
featType = 'layer16' # or layer 19 -  layer19 might not work well
globalpoolfn = Fx.avg_pool2d # can use max also
netwrkgpl = Fx.avg_pool2d # keep it fixed



def load_model(netx,modpath):
    #load through cpu -- safest
    state_dict = torch.load(modpath,map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    netx.load_state_dict(new_state_dict)

def getFeat(extractor,inpt):
    # return pytorch tensor 
    extractor.eval()
    indata = Variable(torch.Tensor(inpt),volatile=True)
    if usegpu:
        indata = indata.cuda()

    pred = extractor(indata)
    print (pred.size())
    if len(pred.size()) > 2:
        gpred = globalpoolfn(pred,kernel_size=pred.size()[2:])
        gpred = gpred.view(gpred.size(0),-1)

    return gpred


def main(y):
#    try:
#        y, sr = lib.load(filename,sr=None)
#    except:
#        raise IOError('Give me an audio  file which I can read!!')
#    
#    if len(y.shape) > 1:
#        print ('Mono Conversion') 
#        y = lib.to_mono(y)
#
#    if sr != srate:
#        print ('Resampling to {}'.format(srate))
#        y = lib.resample(y,sr,srate)

        
    mel_feat = lib.feature.melspectrogram(y=y,sr=srate,n_fft=n_fft,hop_length=hop_length,n_mels=n_mels)
    inpt = lib.power_to_db(mel_feat).T
    print('in shape', inpt.shape)
    
    #quick hack for now
    if inpt.shape[0] < 128:
        inpt = np.concatenate((inpt,np.zeros((128-inpt.shape[0],n_mels))),axis=0)
    

    # input needs to be 4D, batch_size X 1 X inpt_size[0] X inpt_size[1]
    inpt = np.reshape(inpt,(1,1,inpt.shape[0],inpt.shape[1]))
    print (inpt.shape)

    netType = getattr(netark,trainType)
    netx = netType(527,netwrkgpl)
    load_model(netx,pre_model_path)

    
    if usegpu:
        netx.cuda()
    
    feat_extractor = exm.featExtractor(netx,featType)
    
    pred = getFeat(feat_extractor,inpt)

    #numpy arrays
    feature = pred.data.cpu().numpy()
    print (feature.shape)

    # prediction for each segment in each column
    return feature


#if __name__ == '__main__':
#    # Load wave files
#    audio_root = './segmented_data/filtered/background/'
#    label = 2
#    audio_file_list = [x for x in os.listdir(audio_root) if x.endswith('.wav')]
#    save_feature_dir = './features/background'
#   
#    training_feat, test_feat = np.empty((0, 1025)), np.empty((0, 1025))   # 512 features + a label
#    for file in audio_file_list[:15]:
#        print('processing training file: %s' % file)
#        dir_audio = os.path.join(audio_root, file)    
#        feature = main(dir_audio)
#        feature = np.append(feature, label) 
#        training_feat = np.vstack((training_feat, feature))
#    for file in audio_file_list[15:]:
#        print('processing test file: %s' % file)
#        dir_audio = os.path.join(audio_root, file)    
#        feature = main(dir_audio)
#        feature = np.append(feature, label) 
#        test_feat = np.vstack((test_feat, feature))
#
#            
#    print('saving tl features and labels for training:')            
#    with open(save_feature_dir + '/training.csv', 'w') as csvfile:
#        spamwriter = csv.writer(csvfile, delimiter=',')
#        spamwriter.writerows(training_feat)
#        csvfile.close()
#        print('training tl saved.')
#    # save mfcc test set
#    print('saving tl features and labels for test:')
#    with open(save_feature_dir + '/test.csv', 'w') as csvfile2:
#        spamwriter = csv.writer(csvfile2, delimiter=',')
#        spamwriter.writerows(test_feat)
#        csvfile2.close()
#        print('test tl saved.')
