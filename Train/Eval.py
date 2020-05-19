import sys, os
from argparse import ArgumentParser
import setGPU
#os.environ['CUDA_VISIBLE_DEVICES']=''
# Options
parser = ArgumentParser(description ='Script to run the training and evaluate it')
parser.add_argument("--adv", action='store_true', default=False, help="Load adversarial model")
parser.add_argument("--decor", action='store_true', default=False, help="Serve decorrelated training targets")
parser.add_argument("--reduced", action='store_true', default=False, help="reduced model")
parser.add_argument("--simple", action='store_true', default=False, help="simple model")
parser.add_argument("--multi", action='store_true', default=False, help="Binary or categorical crossentropy")
parser.add_argument("-i", help="Training dataCollection.dc", default=None, metavar="FILE")
parser.add_argument("-t", help="Testing dataCollection.dc", default=None, metavar="FILE")
parser.add_argument("-d",  help="Training output dir", default=None, metavar="PATH")
parser.add_argument("-o",  help="Eval output dir", default=None, metavar="PATH")
parser.add_argument("-l",  help="Limit inference time, terminate after X minutes", default=120, metavar="INT")
parser.add_argument("-p",  help="Plot output dir within Eval output dir", default="Plots", metavar="PATH")
parser.add_argument("--storeInputs", action='store_true', help="Store inputs in df", default=False)
parser.add_argument("--taggerName",  help="DeepDouble{input} name in ROC plots", default="X")
parser.add_argument("--era",  help="Era/Year label in plots", default="2016")
parser.add_argument("--datasets", default=None, choices=['db', 'db_pf_cpf_sv', 'db_cpf_sv'])
opts=parser.parse_args()
if opts.decor:  os.environ['DECORRELATE'] = "True"
else:  os.environ['DECORRELATE'] = "False"

if opts.datasets is not None:
    sampleDatasets_cpf_sv = opts.datasets.split("_")

else:
    sampleDatasets_cpf_sv = ["db","cpf","sv"]
    #sampleDatasets_cpf_sv = ["db"]

#select model and eval functions
if opts.adv:
    from models import model_DeepDoubleXAdversarial as trainingModel
elif opts.reduced:
    from models import model_DeepDoubleXReduced as trainingModel
elif opts.simple:
    from models import model_DeepDoubleXSimple as trainingModel
else:
    from models import model_DeepDoubleXReference as trainingModel
from DeepJetCore.training.training_base import training_base
from eval_functions import loadModel, evaluate
from plots_from_df import make_plots
from Metrics import global_metrics_list, acc_kldiv


inputDataset = sampleDatasets_cpf_sv
trainDir = opts.d
inputTrainDataCollection = opts.t
inputTestDataCollection = opts.i
LoadModel = False  # If false, loads weights, loading model can crash when using decorrelation
removedVars = None
if opts.era=="2016":
    eraText=r'2016 (13 TeV)'
elif opts.era=="2017":
    eraText=r'2017 (13 TeV)'

if True:
    evalModel = loadModel(trainDir,inputTestDataCollection,trainingModel,LoadModel,inputDataset,removedVars,adv=opts.adv)
    evalDir = opts.o
    print(evalModel.summary())
    from DeepJetCore.DataCollection import DataCollection
    testd=DataCollection()
    testd.readFromFile(inputTestDataCollection)

    if os.path.isdir(evalDir):
        raise Exception('output directory: %s must not exists yet' % evalDir)
    else:
        os.mkdir(evalDir)

    df = evaluate(testd, inputTestDataCollection, evalModel, evalDir, storeInputs=opts.storeInputs, adv=opts.adv, tlimit=opts.l)
    make_plots(evalDir, savedir=opts.p, taggerName=opts.taggerName, eraText=eraText)


