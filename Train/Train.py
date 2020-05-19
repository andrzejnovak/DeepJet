import sys, os
from argparse import ArgumentParser

# Options 
parser = ArgumentParser(description ='Script to run the training and evaluate it')
parser.add_argument("--decor", action='store_true', default=False, help="Serve decorrelated training targets")
parser.add_argument("--reduced", action='store_true', default=False, help="reduced model")
parser.add_argument("--simple", action='store_true', default=False, help="simple model")
parser.add_argument("--loss", default="loss_kldiv", choices=['loss_kldiv', 'loss_kldiv_3class', 'loss_reg', 'loss_jsdiv'],
                    help="loss to use for decorrelated training")
parser.add_argument("--disco", default=False, action='store_true')
parser.add_argument("--lambda-adv", default='15', help="lambda for adversarial training", type=str)
parser.add_argument("--scale", default='1', help="(initial) scale factor (-lambda_r) for backprop during adversarial training", type=str)
parser.add_argument("--scale-callback",action='store_true', default = False, help = "scale factor (-lambda_r) callback to set it to 2/(1+exp(-10*i_epoch/n_epochs)-1")
parser.add_argument("--classes", default='2', help="Number of classes for multiclassifier", type=str)
parser.add_argument("-g", "--gpu", default=-1, help="Use 1 specific gpu (Need to be in deepjetLinux3_gpu env)", type=int)
parser.add_argument("-m", "--multi-gpu", action='store_true', default=False, help="Use all visible gpus (Need to be in deepjetLinux3_gpu env)")
parser.add_argument("-i", help="Training dataCollection.dc", default=None, metavar="FILE")
parser.add_argument("-o",  help="Training output dir", default=None, metavar="PATH")
parser.add_argument("--batch",  help="Batch size, default = 2000", default=2000, metavar="INT")
parser.add_argument("--epochs",  help="Epochs, default = 50", default=50, metavar="INT")
parser.add_argument("--resume", action='store_true', default=False, help="Continue previous")
parser.add_argument("--pretrained-model", default=None, help="start from specified pretrained model", type=str)
parser.add_argument("--datasets", default=None, choices=['db', 'db_pf_cpf_sv', 'db_cpf_sv'])
opts=parser.parse_args()
if opts.decor:  os.environ['DECORRELATE'] = "True"
else:  os.environ['DECORRELATE'] = "False"
if opts.disco:  os.environ['DISCO'] = "True"
else:  os.environ['DISCO'] = "False"

os.environ['LAMBDA_ADV'] = opts.lambda_adv
os.environ['NCLASSES'] = opts.classes

# Some used default settings
class MyClass:
    """A simple example class"""
    def __init__(self):
	self.gpu = -1
	self.gpufraction = -1
	self.modelMethod = ''
        self.inputDataCollection = ''
        self.outputDir = ''

if opts.datasets is not None:
    sampleDatasets_cpf_sv = opts.datasets.split("_")

else:        
    sampleDatasets_cpf_sv = ["db","cpf","sv"]
    #sampleDatasets_cpf_sv = ["db"]
        
#select model and eval functions
if opts.loss=='loss_reg':
    from models.convolutional import model_DeepDoubleXAdversarial as trainingModel
elif opts.reduced:
    from models.convolutional import model_DeepDoubleXReduced as trainingModel
elif opts.simple:
    from models.convolutional import model_DeepDoubleXSimple  as trainingModel
else:
    from models.convolutional import model_DeepDoubleXReference  as trainingModel

    
from DeepJetCore.training.training_base import training_base

from Losses import loss_NLL, loss_meansquared, loss_kldiv, loss_kldiv_3class, global_loss_list, custom_crossentropy, loss_jsdiv, loss_reg, NBINS, NCLASSES, loss_disc, loss_adv, LAMBDA_ADV, loss_disc_kldiv, loss_disco
from DeepJetCore.modeltools import fixLayersContaining,printLayerInfosAndWeights
from Layers import global_layers_list
from Metrics import global_metrics_list, acc_kldiv, acc_disco, val_disco, mass_kldiv_q, mass_kldiv_h, acc_reg, mass_jsdiv_q
from keras import backend as K
custom_objects_list = {}
custom_objects_list.update(global_loss_list)
custom_objects_list.update(global_layers_list)
custom_objects_list.update(global_metrics_list)

inputDataset = sampleDatasets_cpf_sv

if True:  # Should probably fix
    args = MyClass()
    args.inputDataCollection = opts.i
    args.outputDir = opts.o

    multi_gpu = 1
    if opts.multi_gpu:
        # use all visible GPUs
        multi_gpu = len([x for x in os.popen("nvidia-smi -L").read().split("\n") if "GPU" in x])
        args.gpu = ','.join([str(i) for i in range(multi_gpu)])
        print(args.gpu)
    
    # Separate losses and metrics for training and decorrelation
    if opts.disco:
        loss = loss_disco
        metrics=[acc_disco, val_disco]
    elif opts.decor and opts.loss=='loss_reg':
        print("INFO: using LAMBDA_ADV = %f"%LAMBDA_ADV)
        print("INFO: using NCLASSES = %d"%NCLASSES)
        loss = loss_reg
        metrics = [acc_reg, mass_kldiv_q, mass_kldiv_h, loss_disc, loss_adv]
    elif opts.decor and opts.loss=='loss_kldiv': 
        loss = loss_kldiv
        metrics=[acc_kldiv, mass_kldiv_q, mass_kldiv_h, loss_disc_kldiv]
    elif opts.decor and opts.loss=='loss_kldiv_3class':
        loss = loss_kldiv_3class
        metrics=[acc_kldiv]
    elif opts.decor and opts.loss=='loss_jsdiv':
        loss = loss_jsdiv
        metrics=[acc_kldiv, mass_jsdiv_q, loss_disc_kldiv]
    else: 
        loss = 'categorical_crossentropy'
        metrics=['accuracy']

    # Set up training
    train=training_base(splittrainandtest=0.9,testrun=False, useweights=True, resumeSilently=opts.resume, renewtokens=False, parser=args)
    if not train.modelSet():
        modelargs = {}
        if opts.loss=='loss_reg':
            lambda_variable = K.variable(abs(float(opts.scale)))
            modelargs.update({'nRegTargets':NBINS,                  
                              'discTrainable': True,
                              'advTrainable':True,
                              'scale': lambda_variable})
            
        train.setModel(trainingModel, 
		       datasets=inputDataset, 
		       multi_gpu=multi_gpu,
                       **modelargs)
        train.compileModel(learningrate=0.001,
                           loss=[loss],
	                       metrics=metrics,
			               loss_weights=[1.])
        
        if os.path.isfile(opts.o+"/KERAS_check_best_model.h5"): # To make sure no training is lost on restart
            train.loadWeights(opts.o+"/KERAS_check_best_model.h5")
            print "XXXXXXX loading Weights XXXXXX"
            print "Loaded weights from existing model in output directory."

    # Pretrain and fix batch normalization weights to be consistent
    if not opts.decor:
	model, history = train.trainModel(nepochs=1, 
                                          batchsize=int(opts.batch), 
                                          stop_patience=50,
                                          lr_factor=0.7, 
                                          lr_patience=10, 
                                          lr_epsilon=0.00000001, 
                                          lr_cooldown=2, 
                                          lr_minimum=0.00000001, 
                                          maxqsize=100)

        #train.keras_model=fixLayersContaining(train.keras_model, 'input_batchnorm')
	
        # Need to recompile after fixing batchnorm weights or loading model and changing loss to decorrelate
        train.compileModel(learningrate=0.001,
                           loss=[loss],
	                   metrics=metrics,
		           loss_weights=[1.])
        # Load best weights
        try: # To make sure no training is lost on restart
	    train.loadWeights(opts.o+"/KERAS_check_best_model.h5")
        except:
	    pass

    # Override by loading pretrained model
    if opts.pretrained_model is not None:
        train.loadWeights(opts.pretrained_model)
        
    # Train
    lambda_callback = None
    if opts.decor and opts.loss=='loss_reg' and opts.scale_callback:
        lambda_callback = lambda_variable
    model, history = train.trainModel(nepochs=int(opts.epochs),
				      batchsize=int(opts.batch),
                                      stop_patience=50,
                                      lr_factor=0.7,
                                      lr_patience=10,
                                      lr_epsilon=0.00000001,
                                      lr_cooldown=2,
                                      lr_minimum=0.00000001,
                                      maxqsize=100,
                                      lambda_callback=lambda_callback)

    
