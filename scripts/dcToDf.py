import sys
import os

from argparse import ArgumentParser
import numpy as np
from root_numpy import array2root
import pandas as pd
import h5py

from DeepJetCore.DataCollection import DataCollection


def dcToDf(dc_file, df_out):
	dc=DataCollection()
	dc.readFromFile(dc_file)
    	
	NENT = 1  # Can skip some events
    	filelist=[]
        i=0
	storeInputs = True
	count = 0

	feature_names = dc.dataclass.branches[1]
        spectator_names = dc.dataclass.branches[0]
	labels_names = dc.getUsedTruth()
	labels_names = ['truth'+l for l in labels_names]

	for s in dc.samples:
		if count > 1000000: break
        	spath = dc.getSamplePath(s)
            	filelist.append(spath)
	        h5File = h5py.File(spath)
        	f = h5File
                features_val_i = [h5File['x%i'%j][()] for j in range(0, h5File['x_listlength'][()][0])]
		features_val_i = features_val_i[0][::NENT,0,:]
                #predict_test_i = model.predict(features_val)
                weights_val_i = h5File['w0'][()]
                labels_val_i = h5File['y0'][()][::NENT,:]
                spectators_val_i = h5File['z0'][()][::NENT,0,:]
                if storeInputs: raw_features_val_i = h5File['z1'][()][::NENT,0,:]
                if i==0:
        	        #predict_test = predict_test_i
			weights_val = weights_val_i
                	labels_val = labels_val_i
	                spectators_val = spectators_val_i
			features_val = features_val_i
        	        if storeInputs: raw_features_val = raw_features_val_i
                else:
                	#predict_test = np.concatenate((predict_test,predict_test_i))
			weights_val =  np.concatenate((weights_val, weights_val_i))
	                labels_val = np.concatenate((labels_val, labels_val_i))
	                features_val = np.concatenate((features_val, features_val_i))
        	        spectators_val = np.concatenate((spectators_val, spectators_val_i))
                	if storeInputs: raw_features_val = np.concatenate((raw_features_val, raw_features_val_i))
                i+=1
		count += labels_val.shape[0]

	entries = np.hstack((raw_features_val,spectators_val, labels_val, weights_val.reshape((len(weights_val), 1)) ))
        df = pd.DataFrame(entries , columns = feature_names+spectator_names+labels_names+['weight'])
        #df = pd.DataFrame(raw_features_val+spectators_val , columns = feature_names+spectator_names)
        #print df 
	if df_out != None:
	        df.to_pickle(df_out) 
		print "Saved df to", df_out


if __name__ == '__main__':
	parser = ArgumentParser(description ='todf')
	parser.add_argument("-i", help="Training dataCollection.dc", default=None, metavar="FILE")
	parser.add_argument("-o", help="DF save name", default=None, metavar="FILE")
	opts=parser.parse_args()
	if opts.o != None and not opts.o.endswith('.pkl'): opts.o = opts.o + '.pkl'
        
	if opts.i != None:
		dcToDf(opts.i, opts.o)
	else:	
		dcToDf('/home/anovak/data/dev/80x/DDB/dctrain/dataCollection.dc', opts.o)
	
