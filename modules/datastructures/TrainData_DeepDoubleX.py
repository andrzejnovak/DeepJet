from __future__ import print_function

from DeepJetCore.TrainData import TrainData, fileTimeOut
import numpy


class TrainData_DeepDoubleX(TrainData):
    def __init__(self):
        '''
        This class is meant as a base class for the FatJet studies
        You will not need to edit it for trying out things
        '''
        TrainData.__init__(self)

        #define truth:
        self.treename = "deepntuplizer/tree"
        self.undefTruth = ['isUndefined']
        #self.truthclasses=["fj_isH", "fj_isCC", "fj_isBB", "fj_isNonCC" , "fj_isNonBB", "fj_isZ", "fj_isQCD" , "sample_isQCD"]
        self.truthclasses = [
            "fj_isH", "fj_isBB", "fj_isNonBB", "fj_isZ", "fj_isQCD", "sample_isQCD", "fj_isCC", "fj_isNonCC"
        ]

        self.referenceclass = 'lowest'  ## used for pt reshaping options=['lowest', 'flatten', '<class_name>']
        self.weightbranchX = 'fj_pt'
        self.weightbranchY = 'fj_sdmass'

        self.weight_binX = numpy.array(range(300, 1000, 50) + range(1000, 2600, 200),
                                       dtype=float)

        self.weight_binY = numpy.array([40, 200], dtype=float)
        # Mask bins (pass False to skip)
        self.mask_binX = None
        self.mask_binY = None

        self.weight = True
        self.remove = False
        self.removeUnderOverflow = True

        #this is only needed because the truth definitions are different from deepFlavour
        self.allbranchestoberead = []
        self.registerBranches(self.undefTruth)
        self.registerBranches(self.truthclasses)
        self.registerBranches(['fj_pt', 'fj_sdmass'])
        self.registerBranches([
            "label_H_bb",
            "label_H_cc",
            "label_QCD_bb",
            "label_QCD_cc",
            "label_QCD_others",
            "label_Z_bb",
            "label_Z_cc",
        ])
        print("Branches read:", self.allbranchestoberead)

    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses = ['fj_isNonCC', 'fj_isCC']
        if tuple_in is not None:
            q = tuple_in['fj_isNonCC']
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isCC']
            h = h.view(numpy.ndarray)
            return numpy.vstack((q, h)).transpose()


class TrainData_DeepDoubleX_db(TrainData_DeepDoubleX):
    def __init__(self):
        '''
        This is an example data format description for FatJet studies
        '''
        TrainData_DeepDoubleX.__init__(self)

        self.remove = False
        self.weight = True
        #example of how to register global branches
        self.addBranches([
            'fj_pt', 'fj_eta', 'fj_sdmass', 'fj_n_sdsubjets', 'fj_doubleb', 'fj_tau21',
            'fj_tau32', 'npv', 'npfcands', 'ntracks', 'nsv'
        ])

        self.addBranches([
            'fj_jetNTracks',
            'fj_nSV',
            'fj_tau0_trackEtaRel_0',
            'fj_tau0_trackEtaRel_1',
            'fj_tau0_trackEtaRel_2',
            'fj_tau1_trackEtaRel_0',
            'fj_tau1_trackEtaRel_1',
            'fj_tau1_trackEtaRel_2',
            'fj_tau_flightDistance2dSig_0',
            'fj_tau_flightDistance2dSig_1',
            'fj_tau_vertexDeltaR_0',
            'fj_tau_vertexEnergyRatio_0',
            'fj_tau_vertexEnergyRatio_1',
            'fj_tau_vertexMass_0',
            'fj_tau_vertexMass_1',
            'fj_trackSip2dSigAboveBottom_0',
            'fj_trackSip2dSigAboveBottom_1',
            'fj_trackSip2dSigAboveCharm_0',
            'fj_trackSipdSig_0',
            'fj_trackSipdSig_0_0',
            'fj_trackSipdSig_0_1',
            'fj_trackSipdSig_1',
            'fj_trackSipdSig_1_0',
            'fj_trackSipdSig_1_1',
            'fj_trackSipdSig_2',
            'fj_trackSipdSig_3',
            'fj_z_ratio',
        ])

        #branches that are used directly in the following function 'readFromRootFile'
        #this is a technical trick to speed up the conversion
        #self.registerBranches(['Cpfcan_erel','Cpfcan_eta','Cpfcan_phi',
        #                       'Npfcan_erel','Npfcan_eta','Npfcan_phi',
        #                       'nCpfcand','nNpfcand',
        #                       'jet_eta','jet_phi'])
        self.registerBranches(['sample_isQCD', 'fj_isH', 'fj_isQCD'])

    #this function describes how the branches are converted
    def readFromRootFile(self, filename, TupleMeanStd, weighter):

        #the first part is standard, no changes needed
        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles, ZeroPadParticles
        import numpy
        import ROOT

        fileTimeOut(filename, 60)  #give eos 1 minutes to recover
        rfile = ROOT.TFile.Open(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples = tree.GetEntries()

        #the definition of what to do with the branches

        # those are the global branches (jet pt etc)
        # they should be just glued to each other in one vector
        # and zero padded (and mean subtracted and normalised)
        # x_global = MeanNormZeroPad(filename,TupleMeanStd,
        #                           [self.branches[0]],
        #                           [self.branchcutoffs[0]],self.nsamples)
        # the second part (the pf candidates) should be treated particle wise
        # an array with (njets, nparticles, nproperties) is created
        #
        # MeanNormZeroPad[Particles] does preprocessing, ZeroPad[Particles] does not and we normalzie it with batch_norm layer
        # MeanNorm* does not work when putting the model into cmssw

        x_glb = ZeroPadParticles(filename, None, self.branches[0],
                                 self.branchcutoffs[0], self.nsamples)

        x_db = ZeroPadParticles(filename, None, self.branches[1], self.branchcutoffs[1],
                                self.nsamples)

        # Load tuple
        Tuple = self.readTreeFromRootToTuple(filename)
        # Append classes constructed in reduceTruth fcn
        truth_array = Tuple[self.truthclasses]
        import numpy.lib.recfunctions as rfn
        reduced_truth = self.reduceTruth(truth_array).transpose()
        for i, label in enumerate(self.reducedtruthclasses):
            Tuple = rfn.append_fields(Tuple, label, reduced_truth[i])

        if self.remove:
            notremoves = weighter.createNotRemoveIndices(Tuple)
        if self.weight:
            weights = weighter.getJetWeights(Tuple)
        elif self.remove:
            weights = notremoves  #weighter.createNotRemoveIndices(Tuple)
        else:
            print('neither remove nor weight')
            weights = numpy.empty(self.nsamples)
            weights.fill(1.)

        used_truth = self.reduceTruth(truth_array)
        undef = numpy.sum(used_truth, axis=1)

        if self.remove:
            print('Removing to match weighting')
            notremoves = notremoves[undef > 0]
            weights = weights[notremoves > 0]
            x_glb = x_glb[notremoves > 0]
            x_db = x_db[notremoves > 0]
            used_truth = used_truth[notremoves > 0]

        if self.weight:
            print('Adding weights, removing events with 0 weight')
            x_glb = x_glb[weights > 0]
            x_db = x_db[weights > 0]
            used_truth = used_truth[weights > 0]
            # Weights get adjusted last so they can be used as an index
            weights = weights[weights > 0]

        newnsamp = x_glb.shape[0]
        print('Keeping {}% of input events in the dataCollection'.format(
            int(float(newnsamp) / float(self.nsamples) * 100)))
        self.nsamples = newnsamp

        # fill everything
        self.w = [weights]
        self.x = [x_db]
        self.z = [x_glb]
        self.y = [used_truth]


#######################################


class TrainData_DeepDoubleX_db_pf_cpf_sv(TrainData_DeepDoubleX):
    def __init__(self):
        '''
        This is an example data format description for FatJet studies
        '''
        TrainData_DeepDoubleX.__init__(self)

        #example of how to register global branches
        self.addBranches([
            'fj_pt', 'fj_eta', 'fj_sdmass', 'fj_n_sdsubjets', 'fj_doubleb', 'fj_tau21',
            'fj_tau32', 'npv', 'npfcands', 'ntracks', 'nsv', 'ntrueInt'
        ])

        self.addBranches([
            'fj_jetNTracks', 'fj_nSV', 'fj_tau0_trackEtaRel_0', 'fj_tau0_trackEtaRel_1',
            'fj_tau0_trackEtaRel_2', 'fj_tau1_trackEtaRel_0', 'fj_tau1_trackEtaRel_1',
            'fj_tau1_trackEtaRel_2', 'fj_tau_flightDistance2dSig_0',
            'fj_tau_flightDistance2dSig_1', 'fj_tau_vertexDeltaR_0',
            'fj_tau_vertexEnergyRatio_0', 'fj_tau_vertexEnergyRatio_1',
            'fj_tau_vertexMass_0', 'fj_tau_vertexMass_1',
            'fj_trackSip2dSigAboveBottom_0', 'fj_trackSip2dSigAboveBottom_1',
            'fj_trackSip2dSigAboveCharm_0', 'fj_trackSipdSig_0', 'fj_trackSipdSig_0_0',
            'fj_trackSipdSig_0_1', 'fj_trackSipdSig_1', 'fj_trackSipdSig_1_0',
            'fj_trackSipdSig_1_1', 'fj_trackSipdSig_2', 'fj_trackSipdSig_3',
            'fj_z_ratio'
        ])

        #example of pf candidate branches
        self.addBranches([
            'pfcand_ptrel', 'pfcand_erel', 'pfcand_phirel', 'pfcand_etarel',
            'pfcand_deltaR', 'pfcand_puppiw', 'pfcand_drminsv', 'pfcand_drsubjet1',
            'pfcand_drsubjet2', 'pfcand_hcalFrac'
        ], 100)

        self.addBranches([
            'track_ptrel', 'track_erel', 'track_phirel', 'track_etarel', 'track_deltaR',
            'track_drminsv', 'track_drsubjet1', 'track_drsubjet2', 'track_dz',
            'track_dzsig', 'track_dxy', 'track_dxysig', 'track_normchi2',
            'track_quality', 'track_dptdpt', 'track_detadeta', 'track_dphidphi',
            'track_dxydxy', 'track_dzdz', 'track_dxydz', 'track_dphidxy',
            'track_dlambdadz', 'trackBTag_EtaRel', 'trackBTag_PtRatio',
            'trackBTag_PParRatio', 'trackBTag_Sip2dVal', 'trackBTag_Sip2dSig',
            'trackBTag_Sip3dVal', 'trackBTag_Sip3dSig', 'trackBTag_JetDistVal'
        ], 60)

        self.addBranches([
            'sv_ptrel', 'sv_erel', 'sv_phirel', 'sv_etarel', 'sv_deltaR', 'sv_pt',
            'sv_mass', 'sv_ntracks', 'sv_normchi2', 'sv_dxy', 'sv_dxysig', 'sv_d3d',
            'sv_d3dsig', 'sv_costhetasvpv'
        ], 5)

#this function describes how the branches are converted

    def readFromRootFile(self, filename, TupleMeanStd, weighter):

        #the first part is standard, no changes needed
        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles, ZeroPadParticles
        import numpy
        import ROOT
        print("accessing %s in %s" % (filename, "TrainData_DeepDoubleX.py"))
        fileTimeOut(filename, 240)  #give eos 4 minutes to recover
        rfile = ROOT.TFile.Open(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples = tree.GetEntries()

        x_glb = ZeroPadParticles(filename, None, self.branches[0],
                                 self.branchcutoffs[0], self.nsamples)

        x_db = ZeroPadParticles(filename, None, self.branches[1], self.branchcutoffs[1],
                                self.nsamples)

        x_pf = ZeroPadParticles(filename, None, self.branches[2], self.branchcutoffs[2],
                                self.nsamples)

        x_cpf = ZeroPadParticles(filename, None, self.branches[3],
                                 self.branchcutoffs[3], self.nsamples)

        x_sv = ZeroPadParticles(filename, None, self.branches[4], self.branchcutoffs[4],
                                self.nsamples)

        # Load tuple
        Tuple = self.readTreeFromRootToTuple(filename)
        # Append classes constructed in reduceTruth fcn
        truth_array = Tuple[self.truthclasses]
        import numpy.lib.recfunctions as rfn
        reduced_truth = self.reduceTruth(truth_array).transpose()
        for i, label in enumerate(self.reducedtruthclasses):
            Tuple = rfn.append_fields(Tuple, label, reduced_truth[i])

        if self.remove:
            notremoves = weighter.createNotRemoveIndices(Tuple)
        if self.weight:
            weights = weighter.getJetWeights(Tuple)
        elif self.remove:
            weights = notremoves  #weighter.createNotRemoveIndices(Tuple)
        else:
            print('neither remove nor weight')
            weights = numpy.empty(self.nsamples)
            weights.fill(1.)

        used_truth = self.reduceTruth(truth_array)
        undef = numpy.sum(used_truth, axis=1)

        if self.remove:
            print('Removing to match weighting')
            notremoves = notremoves[undef > 0]
            weights = weights[notremoves > 0]
            x_glb = x_glb[notremoves > 0]
            x_pf = x_pf[notremoves > 0]
            x_db = x_db[notremoves > 0]
            x_sv = x_sv[notremoves > 0]
            x_cpf = x_cpf[notremoves > 0]
            used_truth = used_truth[notremoves > 0]

        if self.weight:
            print('Adding weights, removing events with 0 weight')
            x_glb = x_glb[weights > 0]
            x_pf = x_pf[weights > 0]
            x_db = x_db[weights > 0]
            x_sv = x_sv[weights > 0]
            x_cpf = x_cpf[weights > 0]
            used_truth = used_truth[weights > 0]
            # Weights get adjusted last so they can be used as an index
            weights = weights[weights > 0]

        newnsamp = x_glb.shape[0]
        print('Keeping {}% of input events in the dataCollection'.format(
            int(float(newnsamp) / float(self.nsamples) * 100)))
        self.nsamples = newnsamp

        # fill everything
        self.w = [weights]
        self.x = [x_db, x_pf, x_cpf, x_sv]
        self.z = [x_glb]
        self.y = [used_truth]


#######################################


class TrainData_DeepDoubleX_pruned(TrainData_DeepDoubleX):
    def __init__(self):
        '''
        This is an example data format description for FatJet studies
        '''
        TrainData_DeepDoubleX.__init__(self)

        #example of how to register global branches
        self.addBranches([
            'fj_pt', 'fj_eta', 'fj_sdmass', 'fj_n_sdsubjets', 'fj_doubleb', 'fj_tau21',
            'fj_tau32', 'npv', 'npfcands', 'ntracks', 'nsv', 'ntrueInt'
        ])

        self.addBranches([
            #'fj_tau_flightDistance2dSig_1',
            #'fj_tau_vertexMass_1',
            #'fj_trackSipdSig_1_1',
            'fj_jetNTracks',
            'fj_nSV',
            'fj_tau0_trackEtaRel_0',
            'fj_tau0_trackEtaRel_1',
            'fj_tau0_trackEtaRel_2',
            'fj_tau1_trackEtaRel_0',
            'fj_tau1_trackEtaRel_1',
            'fj_tau1_trackEtaRel_2',
            'fj_tau_flightDistance2dSig_0',
            'fj_tau_vertexDeltaR_0',
            'fj_tau_vertexEnergyRatio_0',
            'fj_tau_vertexEnergyRatio_1',
            'fj_tau_vertexMass_0',
            'fj_trackSip2dSigAboveBottom_0',
            'fj_trackSip2dSigAboveBottom_1',
            'fj_trackSip2dSigAboveCharm_0',
            'fj_trackSipdSig_0',
            'fj_trackSipdSig_0_0',
            'fj_trackSipdSig_0_1',
            'fj_trackSipdSig_1',
            'fj_trackSipdSig_1_0',
            'fj_trackSipdSig_2',
            'fj_trackSipdSig_3',
            'fj_z_ratio'
        ])

        #example of pf candidate branches
        self.addBranches(
            [
                'pfcand_deltaR',
                'pfcand_drminsv',
                'pfcand_drsubjet1',
                'pfcand_drsubjet2',
                # 'pfcand_erel',
                # 'pfcand_etarel',
                # 'pfcand_hcalFrac',
                # 'pfcand_phirel',
                # 'pfcand_ptrel',
                'pfcand_puppiw'
            ],
            100)

        self.addBranches(
            [
                # 'trackBTag_Sip2dVal', 'trackBTag_Sip2dSig',
                # 'trackBTag_Sip3dVal', 'trackBTag_Sip3dSig',
                'trackBTag_EtaRel',
                'trackBTag_JetDistVal',
                'trackBTag_PParRatio',
                'trackBTag_PtRatio',
                'track_deltaR',
                # 'track_detadeta',
                # 'track_dlambdadz',
                # 'track_dphidphi',
                # 'track_dphidxy',
                # 'track_dptdpt',
                'track_drminsv',
                'track_drsubjet1',
                'track_drsubjet2',
                # 'track_dxy',
                # 'track_dxydxy',
                # 'track_dxydz',
                'track_dxysig',
                # 'track_dz',
                # 'track_dzdz',
                # 'track_dzsig',
                # 'track_erel',
                'track_etarel',
                # 'track_normchi2',
                # 'track_phirel',
                'track_ptrel',
                'track_quality'
            ],
            60)

        self.addBranches([
            'sv_costhetasvpv',
            'sv_d3d',
            'sv_d3dsig',
            'sv_deltaR',
            'sv_dxy',
            'sv_dxysig',
            'sv_erel',
            #'sv_etarel',
            'sv_mass',
            #'sv_normchi2',
            'sv_ntracks',
            #'sv_phirel',
            'sv_pt',
            'sv_ptrel'
        ], 5)

#this function describes how the branches are converted

    def readFromRootFile(self, filename, TupleMeanStd, weighter):

        #the first part is standard, no changes needed
        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles, ZeroPadParticles
        import numpy
        import ROOT
        print("accessing %s in %s" % (filename, "TrainData_DeepDoubleX.py"))
        fileTimeOut(filename, 240)  #give eos 4 minutes to recover
        rfile = ROOT.TFile.Open(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples = tree.GetEntries()

        x_glb = ZeroPadParticles(filename, None, self.branches[0],
                                 self.branchcutoffs[0], self.nsamples)

        x_db = ZeroPadParticles(filename, None, self.branches[1], self.branchcutoffs[1],
                                self.nsamples)

        x_pf = ZeroPadParticles(filename, None, self.branches[2], self.branchcutoffs[2],
                                self.nsamples)

        x_cpf = ZeroPadParticles(filename, None, self.branches[3],
                                 self.branchcutoffs[3], self.nsamples)

        x_sv = ZeroPadParticles(filename, None, self.branches[4], self.branchcutoffs[4],
                                self.nsamples)

        # Load tuple
        Tuple = self.readTreeFromRootToTuple(filename)
        # Append classes constructed in reduceTruth fcn
        truth_array = Tuple[self.truthclasses]
        import numpy.lib.recfunctions as rfn
        reduced_truth = self.reduceTruth(truth_array).transpose()
        for i, label in enumerate(self.reducedtruthclasses):
            Tuple = rfn.append_fields(Tuple, label, reduced_truth[i])

        if self.remove:
            notremoves = weighter.createNotRemoveIndices(Tuple)
        if self.weight:
            weights = weighter.getJetWeights(Tuple)
        elif self.remove:
            weights = notremoves  #weighter.createNotRemoveIndices(Tuple)
        else:
            print('neither remove nor weight')
            weights = numpy.empty(self.nsamples)
            weights.fill(1.)

        used_truth = self.reduceTruth(truth_array)
        undef = numpy.sum(used_truth, axis=1)

        if self.remove:
            print('Removing to match weighting')
            notremoves = notremoves[undef > 0]
            weights = weights[notremoves > 0]
            x_glb = x_glb[notremoves > 0]
            x_pf = x_pf[notremoves > 0]
            x_db = x_db[notremoves > 0]
            x_sv = x_sv[notremoves > 0]
            x_cpf = x_cpf[notremoves > 0]
            used_truth = used_truth[notremoves > 0]

        if self.weight:
            print('Adding weights, removing events with 0 weight')
            x_glb = x_glb[weights > 0]
            x_pf = x_pf[weights > 0]
            x_db = x_db[weights > 0]
            x_sv = x_sv[weights > 0]
            x_cpf = x_cpf[weights > 0]
            used_truth = used_truth[weights > 0]
            # Weights get adjusted last so they can be used as an index
            weights = weights[weights > 0]

        newnsamp = x_glb.shape[0]
        print('Keeping {}% of input events in the dataCollection'.format(
            int(float(newnsamp) / float(self.nsamples) * 100)))
        self.nsamples = newnsamp

        # fill everything
        self.w = [weights]
        self.x = [x_db, x_pf, x_cpf, x_sv]
        self.z = [x_glb]
        self.y = [used_truth]


#######################################


class TrainData_DeepDoubleX_apruned(TrainData_DeepDoubleX):
    def __init__(self):
        '''
        This is an example data format description for FatJet studies
        '''
        TrainData_DeepDoubleX.__init__(self)

        #example of how to register global branches
        self.addBranches([
            'fj_pt', 'fj_eta', 'fj_sdmass', 'fj_n_sdsubjets', 'fj_doubleb', 'fj_tau21',
            'fj_tau32', 'npv', 'npfcands', 'ntracks', 'nsv', 'ntrueInt'
        ])

        self.addBranches([
            'fj_jetNTracks',
            'fj_nSV',
            'fj_tau0_trackEtaRel_0',
            'fj_tau0_trackEtaRel_1',
            'fj_tau0_trackEtaRel_2',
            'fj_tau1_trackEtaRel_0',
            'fj_tau1_trackEtaRel_1',
            'fj_tau1_trackEtaRel_2',
            'fj_tau_flightDistance2dSig_0',
            # 'fj_tau_flightDistance2dSig_1',
            'fj_tau_vertexDeltaR_0',
            'fj_tau_vertexEnergyRatio_0',
            'fj_tau_vertexEnergyRatio_1',
            # 'fj_tau_vertexMass_0',
            # 'fj_tau_vertexMass_1',
            'fj_trackSip2dSigAboveBottom_0',
            'fj_trackSip2dSigAboveBottom_1',
            'fj_trackSip2dSigAboveCharm_0',
            'fj_trackSipdSig_0',
            'fj_trackSipdSig_0_0',
            'fj_trackSipdSig_0_1',
            'fj_trackSipdSig_1',
            'fj_trackSipdSig_1_0',
            # 'fj_trackSipdSig_1_1',
            # 'fj_trackSipdSig_2',
            # 'fj_trackSipdSig_3',
            'fj_z_ratio'
        ])

        #example of pf candidate branches
        self.addBranches(
            [
                'pfcand_deltaR',
                'pfcand_drminsv',
                'pfcand_drsubjet1',
                'pfcand_drsubjet2',
                'pfcand_erel',
                # 'pfcand_etarel',
                'pfcand_hcalFrac',
                # 'pfcand_phirel',
                'pfcand_ptrel',
                'pfcand_puppiw'
            ],
            100)

        self.addBranches(
            [
                'trackBTag_EtaRel',
                'trackBTag_JetDistVal',
                'trackBTag_PParRatio',
                'trackBTag_PtRatio',
                'trackBTag_Sip2dSig',
                'trackBTag_Sip2dVal',
                'trackBTag_Sip3dSig',
                'trackBTag_Sip3dVal',
                'track_deltaR',
                # 'track_detadeta',
                # 'track_dlambdadz',
                # 'track_dphidphi',
                # 'track_dphidxy',
                'track_dptdpt',
                'track_drminsv',
                'track_drsubjet1',
                'track_drsubjet2',
                'track_dxy',
                'track_dxydxy',
                'track_dxydz',
                'track_dxysig',
                'track_dz',
                # 'track_dzdz',
                'track_dzsig',
                'track_erel',
                'track_etarel',
                'track_normchi2',
                'track_phirel',
                'track_ptrel',
                'track_quality'
            ],
            60)

        self.addBranches([
            'sv_costhetasvpv',
            'sv_d3d',
            'sv_d3dsig',
            'sv_deltaR',
            'sv_dxy',
            'sv_dxysig',
            'sv_erel',
            #'sv_etarel',
            'sv_mass',
            #'sv_normchi2',
            'sv_ntracks',
            #'sv_phirel',
            'sv_pt',
            'sv_ptrel'
        ], 5)

#this function describes how the branches are converted

    def readFromRootFile(self, filename, TupleMeanStd, weighter):

        #the first part is standard, no changes needed
        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles, ZeroPadParticles
        import numpy
        import ROOT
        print("accessing %s in %s" % (filename, "TrainData_DeepDoubleX.py"))
        fileTimeOut(filename, 240)  #give eos 4 minutes to recover
        rfile = ROOT.TFile.Open(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples = tree.GetEntries()

        x_glb = ZeroPadParticles(filename, None, self.branches[0],
                                 self.branchcutoffs[0], self.nsamples)

        x_db = ZeroPadParticles(filename, None, self.branches[1], self.branchcutoffs[1],
                                self.nsamples)

        x_pf = ZeroPadParticles(filename, None, self.branches[2], self.branchcutoffs[2],
                                self.nsamples)

        x_cpf = ZeroPadParticles(filename, None, self.branches[3],
                                 self.branchcutoffs[3], self.nsamples)

        x_sv = ZeroPadParticles(filename, None, self.branches[4], self.branchcutoffs[4],
                                self.nsamples)

        # Load tuple
        Tuple = self.readTreeFromRootToTuple(filename)
        # Append classes constructed in reduceTruth fcn
        truth_array = Tuple[self.truthclasses]
        import numpy.lib.recfunctions as rfn
        reduced_truth = self.reduceTruth(truth_array).transpose()
        for i, label in enumerate(self.reducedtruthclasses):
            Tuple = rfn.append_fields(Tuple, label, reduced_truth[i])

        if self.remove:
            notremoves = weighter.createNotRemoveIndices(Tuple)
        if self.weight:
            weights = weighter.getJetWeights(Tuple)
        elif self.remove:
            weights = notremoves  #weighter.createNotRemoveIndices(Tuple)
        else:
            print('neither remove nor weight')
            weights = numpy.empty(self.nsamples)
            weights.fill(1.)

        used_truth = self.reduceTruth(truth_array)
        undef = numpy.sum(used_truth, axis=1)

        if self.remove:
            print('Removing to match weighting')
            notremoves = notremoves[undef > 0]
            weights = weights[notremoves > 0]
            x_glb = x_glb[notremoves > 0]
            x_pf = x_pf[notremoves > 0]
            x_db = x_db[notremoves > 0]
            x_sv = x_sv[notremoves > 0]
            x_cpf = x_cpf[notremoves > 0]
            used_truth = used_truth[notremoves > 0]

        if self.weight:
            print('Adding weights, removing events with 0 weight')
            x_glb = x_glb[weights > 0]
            x_pf = x_pf[weights > 0]
            x_db = x_db[weights > 0]
            x_sv = x_sv[weights > 0]
            x_cpf = x_cpf[weights > 0]
            used_truth = used_truth[weights > 0]
            # Weights get adjusted last so they can be used as an index
            weights = weights[weights > 0]

        newnsamp = x_glb.shape[0]
        print('Keeping {}% of input events in the dataCollection'.format(
            int(float(newnsamp) / float(self.nsamples) * 100)))
        self.nsamples = newnsamp

        # fill everything
        self.w = [weights]
        self.x = [x_db, x_pf, x_cpf, x_sv]
        self.z = [x_glb]
        self.y = [used_truth]


#######################################


class TrainData_DeepDoubleX_bpruned(TrainData_DeepDoubleX):
    def __init__(self):
        '''
        This is an example data format description for FatJet studies
        '''
        TrainData_DeepDoubleX.__init__(self)

        #example of how to register global branches
        self.addBranches([
            'fj_pt', 'fj_eta', 'fj_sdmass', 'fj_n_sdsubjets', 'fj_doubleb', 'fj_tau21',
            'fj_tau32', 'npv', 'npfcands', 'ntracks', 'nsv', 'ntrueInt'
        ])

        self.addBranches([
            'fj_jetNTracks',
            # 'fj_nSV',
            'fj_tau0_trackEtaRel_0',
            'fj_tau0_trackEtaRel_1',
            # 'fj_tau0_trackEtaRel_2',
            # 'fj_tau1_trackEtaRel_0',
            # 'fj_tau1_trackEtaRel_1',
            # 'fj_tau1_trackEtaRel_2',
            # 'fj_tau_flightDistance2dSig_0',
            # 'fj_tau_flightDistance2dSig_1',
            'fj_tau_vertexDeltaR_0',
            'fj_tau_vertexEnergyRatio_0',
            # 'fj_tau_vertexEnergyRatio_1',
            # 'fj_tau_vertexMass_0',
            # 'fj_tau_vertexMass_1',
            # 'fj_trackSip2dSigAboveBottom_0',
            # 'fj_trackSip2dSigAboveBottom_1',
            # 'fj_trackSip2dSigAboveCharm_0',
            # 'fj_trackSipdSig_0',
            # 'fj_trackSipdSig_0_0',
            # 'fj_trackSipdSig_0_1',
            # 'fj_trackSipdSig_1',
            # 'fj_trackSipdSig_1_0',
            # 'fj_trackSipdSig_1_1',
            # 'fj_trackSipdSig_2',
            # 'fj_trackSipdSig_3',
            # 'fj_z_ratio'
        ])

        #example of pf candidate branches
        self.addBranches(
            [
                'pfcand_deltaR',
                'pfcand_drminsv',
                'pfcand_drsubjet1',
                'pfcand_drsubjet2',
                'pfcand_erel',
                # 'pfcand_etarel',
                'pfcand_hcalFrac',
                # 'pfcand_phirel',
                'pfcand_ptrel',
                'pfcand_puppiw'
            ],
            100)

        self.addBranches(
            [
                'trackBTag_EtaRel',
                'trackBTag_JetDistVal',
                'trackBTag_PParRatio',
                'trackBTag_PtRatio',
                'trackBTag_Sip2dSig',
                'trackBTag_Sip2dVal',
                'trackBTag_Sip3dSig',
                'trackBTag_Sip3dVal',
                'track_deltaR',
                # 'track_detadeta',
                # 'track_dlambdadz',
                # 'track_dphidphi',
                # 'track_dphidxy',
                # 'track_dptdpt',
                'track_drminsv',
                'track_drsubjet1',
                'track_drsubjet2',
                'track_dxy',
                # 'track_dxydxy',
                # 'track_dxydz',
                'track_dxysig',
                'track_dz',
                # 'track_dzdz',
                'track_dzsig',
                'track_erel',
                'track_etarel',
                'track_normchi2',
                # 'track_phirel',
                'track_ptrel',
                'track_quality'
            ],
            60)

        self.addBranches([
            'sv_costhetasvpv',
            # 'sv_d3d',
            # 'sv_d3dsig',
            'sv_deltaR',
            # 'sv_dxy',
            'sv_dxysig',
            # 'sv_erel',
            #'sv_etarel',
            'sv_mass',
            #'sv_normchi2',
            'sv_ntracks',
            #'sv_phirel',
            'sv_pt',
            'sv_ptrel'
        ], 5)

#this function describes how the branches are converted

    def readFromRootFile(self, filename, TupleMeanStd, weighter):

        #the first part is standard, no changes needed
        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles, ZeroPadParticles
        import numpy
        import ROOT
        print("accessing %s in %s" % (filename, "TrainData_DeepDoubleX.py"))
        fileTimeOut(filename, 240)  #give eos 4 minutes to recover
        rfile = ROOT.TFile.Open(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples = tree.GetEntries()

        x_glb = ZeroPadParticles(filename, None, self.branches[0],
                                 self.branchcutoffs[0], self.nsamples)

        x_db = ZeroPadParticles(filename, None, self.branches[1], self.branchcutoffs[1],
                                self.nsamples)

        x_pf = ZeroPadParticles(filename, None, self.branches[2], self.branchcutoffs[2],
                                self.nsamples)

        x_cpf = ZeroPadParticles(filename, None, self.branches[3],
                                 self.branchcutoffs[3], self.nsamples)

        x_sv = ZeroPadParticles(filename, None, self.branches[4], self.branchcutoffs[4],
                                self.nsamples)

        # Load tuple
        Tuple = self.readTreeFromRootToTuple(filename)
        # Append classes constructed in reduceTruth fcn
        truth_array = Tuple[self.truthclasses]
        import numpy.lib.recfunctions as rfn
        reduced_truth = self.reduceTruth(truth_array).transpose()
        for i, label in enumerate(self.reducedtruthclasses):
            Tuple = rfn.append_fields(Tuple, label, reduced_truth[i])

        if self.remove:
            notremoves = weighter.createNotRemoveIndices(Tuple)
        if self.weight:
            weights = weighter.getJetWeights(Tuple)
        elif self.remove:
            weights = notremoves  #weighter.createNotRemoveIndices(Tuple)
        else:
            print('neither remove nor weight')
            weights = numpy.empty(self.nsamples)
            weights.fill(1.)

        used_truth = self.reduceTruth(truth_array)
        undef = numpy.sum(used_truth, axis=1)

        if self.remove:
            print('Removing to match weighting')
            notremoves = notremoves[undef > 0]
            weights = weights[notremoves > 0]
            x_glb = x_glb[notremoves > 0]
            x_pf = x_pf[notremoves > 0]
            x_db = x_db[notremoves > 0]
            x_sv = x_sv[notremoves > 0]
            x_cpf = x_cpf[notremoves > 0]
            used_truth = used_truth[notremoves > 0]

        if self.weight:
            print('Adding weights, removing events with 0 weight')
            x_glb = x_glb[weights > 0]
            x_pf = x_pf[weights > 0]
            x_db = x_db[weights > 0]
            x_sv = x_sv[weights > 0]
            x_cpf = x_cpf[weights > 0]
            used_truth = used_truth[weights > 0]
            # Weights get adjusted last so they can be used as an index
            weights = weights[weights > 0]

        newnsamp = x_glb.shape[0]
        print('Keeping {}% of input events in the dataCollection'.format(
            int(float(newnsamp) / float(self.nsamples) * 100)))
        self.nsamples = newnsamp

        # fill everything
        self.w = [weights]
        self.x = [x_db, x_pf, x_cpf, x_sv]
        self.z = [x_glb]
        self.y = [used_truth]


#######################################



class TrainData_DeepDoubleX_bprunedshort(TrainData_DeepDoubleX):
    def __init__(self):
        '''
        This is an example data format description for FatJet studies
        '''
        TrainData_DeepDoubleX.__init__(self)

        #example of how to register global branches
        self.addBranches([
            'fj_pt', 'fj_eta', 'fj_sdmass', 'fj_n_sdsubjets', 'fj_doubleb', 'fj_tau21',
            'fj_tau32', 'npv', 'npfcands', 'ntracks', 'nsv', 'ntrueInt'
        ])

        self.addBranches([
            'fj_jetNTracks',
            # 'fj_nSV',
            'fj_tau0_trackEtaRel_0',
            'fj_tau0_trackEtaRel_1',
            # 'fj_tau0_trackEtaRel_2',
            # 'fj_tau1_trackEtaRel_0',
            # 'fj_tau1_trackEtaRel_1',
            # 'fj_tau1_trackEtaRel_2',
            # 'fj_tau_flightDistance2dSig_0',
            # 'fj_tau_flightDistance2dSig_1',
            'fj_tau_vertexDeltaR_0',
            'fj_tau_vertexEnergyRatio_0',
            # 'fj_tau_vertexEnergyRatio_1',
            # 'fj_tau_vertexMass_0',
            # 'fj_tau_vertexMass_1',
            # 'fj_trackSip2dSigAboveBottom_0',
            # 'fj_trackSip2dSigAboveBottom_1',
            # 'fj_trackSip2dSigAboveCharm_0',
            # 'fj_trackSipdSig_0',
            # 'fj_trackSipdSig_0_0',
            # 'fj_trackSipdSig_0_1',
            # 'fj_trackSipdSig_1',
            # 'fj_trackSipdSig_1_0',
            # 'fj_trackSipdSig_1_1',
            # 'fj_trackSipdSig_2',
            # 'fj_trackSipdSig_3',
            # 'fj_z_ratio'
        ])

        #example of pf candidate branches
        self.addBranches(
            [
                'pfcand_deltaR',
                'pfcand_drminsv',
                'pfcand_drsubjet1',
                'pfcand_drsubjet2',
                'pfcand_erel',
                # 'pfcand_etarel',
                'pfcand_hcalFrac',
                # 'pfcand_phirel',
                'pfcand_ptrel',
                'pfcand_puppiw'
            ],
            60)

        self.addBranches(
            [
                'trackBTag_EtaRel',
                'trackBTag_JetDistVal',
                'trackBTag_PParRatio',
                'trackBTag_PtRatio',
                'trackBTag_Sip2dSig',
                'trackBTag_Sip2dVal',
                'trackBTag_Sip3dSig',
                'trackBTag_Sip3dVal',
                'track_deltaR',
                # 'track_detadeta',
                # 'track_dlambdadz',
                # 'track_dphidphi',
                # 'track_dphidxy',
                # 'track_dptdpt',
                'track_drminsv',
                'track_drsubjet1',
                'track_drsubjet2',
                'track_dxy',
                # 'track_dxydxy',
                # 'track_dxydz',
                'track_dxysig',
                'track_dz',
                # 'track_dzdz',
                'track_dzsig',
                'track_erel',
                'track_etarel',
                'track_normchi2',
                # 'track_phirel',
                'track_ptrel',
                'track_quality'
            ],
            40)

        self.addBranches([
            'sv_costhetasvpv',
            # 'sv_d3d',
            # 'sv_d3dsig',
            'sv_deltaR',
            # 'sv_dxy',
            'sv_dxysig',
            # 'sv_erel',
            #'sv_etarel',
            'sv_mass',
            #'sv_normchi2',
            'sv_ntracks',
            #'sv_phirel',
            'sv_pt',
            'sv_ptrel'
        ], 5)

#this function describes how the branches are converted

    def readFromRootFile(self, filename, TupleMeanStd, weighter):

        #the first part is standard, no changes needed
        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles, ZeroPadParticles
        import numpy
        import ROOT
        print("accessing %s in %s" % (filename, "TrainData_DeepDoubleX.py"))
        fileTimeOut(filename, 240)  #give eos 4 minutes to recover
        rfile = ROOT.TFile.Open(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples = tree.GetEntries()

        x_glb = ZeroPadParticles(filename, None, self.branches[0],
                                 self.branchcutoffs[0], self.nsamples)

        x_db = ZeroPadParticles(filename, None, self.branches[1], self.branchcutoffs[1],
                                self.nsamples)

        x_pf = ZeroPadParticles(filename, None, self.branches[2], self.branchcutoffs[2],
                                self.nsamples)

        x_cpf = ZeroPadParticles(filename, None, self.branches[3],
                                 self.branchcutoffs[3], self.nsamples)

        x_sv = ZeroPadParticles(filename, None, self.branches[4], self.branchcutoffs[4],
                                self.nsamples)

        # Load tuple
        Tuple = self.readTreeFromRootToTuple(filename)
        # Append classes constructed in reduceTruth fcn
        truth_array = Tuple[self.truthclasses]
        import numpy.lib.recfunctions as rfn
        reduced_truth = self.reduceTruth(truth_array).transpose()
        for i, label in enumerate(self.reducedtruthclasses):
            Tuple = rfn.append_fields(Tuple, label, reduced_truth[i])

        if self.remove:
            notremoves = weighter.createNotRemoveIndices(Tuple)
        if self.weight:
            weights = weighter.getJetWeights(Tuple)
        elif self.remove:
            weights = notremoves  #weighter.createNotRemoveIndices(Tuple)
        else:
            print('neither remove nor weight')
            weights = numpy.empty(self.nsamples)
            weights.fill(1.)

        used_truth = self.reduceTruth(truth_array)
        undef = numpy.sum(used_truth, axis=1)

        if self.remove:
            print('Removing to match weighting')
            notremoves = notremoves[undef > 0]
            weights = weights[notremoves > 0]
            x_glb = x_glb[notremoves > 0]
            x_pf = x_pf[notremoves > 0]
            x_db = x_db[notremoves > 0]
            x_sv = x_sv[notremoves > 0]
            x_cpf = x_cpf[notremoves > 0]
            used_truth = used_truth[notremoves > 0]

        if self.weight:
            print('Adding weights, removing events with 0 weight')
            x_glb = x_glb[weights > 0]
            x_pf = x_pf[weights > 0]
            x_db = x_db[weights > 0]
            x_sv = x_sv[weights > 0]
            x_cpf = x_cpf[weights > 0]
            used_truth = used_truth[weights > 0]
            # Weights get adjusted last so they can be used as an index
            weights = weights[weights > 0]

        newnsamp = x_glb.shape[0]
        print('Keeping {}% of input events in the dataCollection'.format(
            int(float(newnsamp) / float(self.nsamples) * 100)))
        self.nsamples = newnsamp

        # fill everything
        self.w = [weights]
        self.x = [x_db, x_pf, x_cpf, x_sv]
        self.z = [x_glb]
        self.y = [used_truth]


#######################################



class TrainData_DeepDoubleX_db_cpf_sv_reduced(TrainData_DeepDoubleX):
    def __init__(self):
        '''
        This is an example data format description for FatJet studies
        '''
        TrainData_DeepDoubleX.__init__(self)

        #example of how to register global branches
        # Only takes floats, not ints
        self.addBranches([
            'fj_pt', 'fj_eta', 'fj_sdmass', 'fj_n_sdsubjets', 'fj_doubleb', 'fj_tau21',
            'fj_tau32', 'npv', 'npfcands', 'ntracks', 'nsv', 'ntrueInt'
        ])

        self.addBranches([
            'fj_jetNTracks', 'fj_nSV', 'fj_tau0_trackEtaRel_0', 'fj_tau0_trackEtaRel_1',
            'fj_tau0_trackEtaRel_2', 'fj_tau1_trackEtaRel_0', 'fj_tau1_trackEtaRel_1',
            'fj_tau1_trackEtaRel_2', 'fj_tau_flightDistance2dSig_0',
            'fj_tau_flightDistance2dSig_1', 'fj_tau_vertexDeltaR_0',
            'fj_tau_vertexEnergyRatio_0', 'fj_tau_vertexEnergyRatio_1',
            'fj_tau_vertexMass_0', 'fj_tau_vertexMass_1',
            'fj_trackSip2dSigAboveBottom_0', 'fj_trackSip2dSigAboveBottom_1',
            'fj_trackSip2dSigAboveCharm_0', 'fj_trackSipdSig_0', 'fj_trackSipdSig_0_0',
            'fj_trackSipdSig_0_1', 'fj_trackSipdSig_1', 'fj_trackSipdSig_1_0',
            'fj_trackSipdSig_1_1', 'fj_trackSipdSig_2', 'fj_trackSipdSig_3',
            'fj_z_ratio'
        ])

        self.addBranches([
            'trackBTag_EtaRel', 'trackBTag_PtRatio', 'trackBTag_PParRatio',
            'trackBTag_Sip2dVal', 'trackBTag_Sip2dSig', 'trackBTag_Sip3dVal',
            'trackBTag_Sip3dSig', 'trackBTag_JetDistVal'
        ], 60)

        self.addBranches([
            'sv_d3d',
            'sv_d3dsig',
        ], 5)

        #branches that are used directly in the following function 'readFromRootFile'
        #this is a technical trick to speed up the conversion
        #self.registerBranches(['Cpfcan_erel','Cpfcan_eta','Cpfcan_phi',
        #                       'Npfcan_erel','Npfcan_eta','Npfcan_phi',
        #                       'nCpfcand','nNpfcand',
        #                       'jet_eta','jet_phi'])

    #this function describes how the branches are converted
    def readFromRootFile(self, filename, TupleMeanStd, weighter):

        #the first part is standard, no changes needed
        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles, ZeroPadParticles
        import numpy
        import ROOT

        fileTimeOut(filename, 60)  #give eos 1 minutes to recover
        rfile = ROOT.TFile.Open(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples = tree.GetEntries()

        #the definition of what to do with the branches

        # those are the global branches (jet pt etc)
        # they should be just glued to each other in one vector
        # and zero padded (and mean subtracted and normalised)
        # x_global = MeanNormZeroPad(filename,TupleMeanStd,
        #                           [self.branches[0]],
        #                           [self.branchcutoffs[0]],self.nsamples)
        # the second part (the pf candidates) should be treated particle wise
        # an array with (njets, nparticles, nproperties) is created
        #
        # MeanNormZeroPad[Particles] does preprocessing, ZeroPad[Particles] does not and we normalzie it with batch_norm layer
        # MeanNorm* does not work when putting the model into cmssw

        x_glb = ZeroPadParticles(filename, None, self.branches[0],
                                 self.branchcutoffs[0], self.nsamples)

        x_db = ZeroPadParticles(filename, None, self.branches[1], self.branchcutoffs[1],
                                self.nsamples)

        x_cpf = ZeroPadParticles(filename, None, self.branches[2],
                                 self.branchcutoffs[2], self.nsamples)

        x_sv = ZeroPadParticles(filename, None, self.branches[3], self.branchcutoffs[3],
                                self.nsamples)

        # Load tuple
        Tuple = self.readTreeFromRootToTuple(filename)
        # Append classes constructed in reduceTruth fcn
        truth_array = Tuple[self.truthclasses]
        import numpy.lib.recfunctions as rfn
        reduced_truth = self.reduceTruth(truth_array).transpose()
        for i, label in enumerate(self.reducedtruthclasses):
            Tuple = rfn.append_fields(Tuple, label, reduced_truth[i])

        if self.remove:
            notremoves = weighter.createNotRemoveIndices(Tuple)
        if self.weight:
            weights = weighter.getJetWeights(Tuple)
        elif self.remove:
            weights = notremoves  #weighter.createNotRemoveIndices(Tuple)
        else:
            print('neither remove nor weight')
            weights = numpy.empty(self.nsamples)
            weights.fill(1.)

        used_truth = self.reduceTruth(truth_array)
        undef = numpy.sum(used_truth, axis=1)

        if self.remove:
            print('Removing to match weighting')
            notremoves = notremoves[undef > 0]
            weights = weights[notremoves > 0]
            x_glb = x_glb[notremoves > 0]
            x_db = x_db[notremoves > 0]
            x_sv = x_sv[notremoves > 0]
            x_cpf = x_cpf[notremoves > 0]
            used_truth = used_truth[notremoves > 0]

        if self.weight:
            print('Adding weights, removing events with 0 weight')
            x_glb = x_glb[weights > 0]
            x_db = x_db[weights > 0]
            x_sv = x_sv[weights > 0]
            x_cpf = x_cpf[weights > 0]
            used_truth = used_truth[weights > 0]
            # Weights get adjusted last so they can be used as an index
            weights = weights[weights > 0]

        newnsamp = x_glb.shape[0]
        print('Keeping {}% of input events in the dataCollection'.format(
            int(float(newnsamp) / float(self.nsamples) * 100)))
        self.nsamples = newnsamp

        # fill everything
        self.w = [weights]
        self.x = [x_db, x_cpf, x_sv]
        self.z = [x_glb]
        self.y = [used_truth]


class TrainData_DeepDoubleC_db_cpf_sv_reduced(TrainData_DeepDoubleX_db_cpf_sv_reduced):
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses = ['QCD', 'Hcc']
        if tuple_in is not None:
            q = tuple_in['fj_isQCD'] * tuple_in['sample_isQCD']
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isCC'] * tuple_in['fj_isH']
            h = h.view(numpy.ndarray)
            return numpy.vstack((q, h)).transpose()


class TrainData_DeepDoubleC_db_cpf_sv_reduced_incglu(
        TrainData_DeepDoubleX_db_cpf_sv_reduced):
    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses = ['non-cc', 'cc']
        if tuple_in is not None:
            q = tuple_in['fj_isNonCC'] * tuple_in['fj_isQCD'] * tuple_in['sample_isQCD']
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isCC']
            h = h.view(numpy.ndarray)
            return numpy.vstack((q, h)).transpose()


#######################################
#             DeepDoubleCvB           #
#######################################
class TrainData_DeepDoubleCvB_db(TrainData_DeepDoubleX_db):
    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses = ['Hbb', 'Hcc']
        if tuple_in is not None:
            q = tuple_in['fj_isBB'] * tuple_in['fj_isH']
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isCC'] * tuple_in['fj_isH']
            h = h.view(numpy.ndarray)
            return numpy.vstack((q, h)).transpose()


######################################
class TrainData_DeepDoubleCvB_db_pf_cpf_sv(TrainData_DeepDoubleX_db_pf_cpf_sv):
    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses = ['Hbb', 'Hcc']
        if tuple_in is not None:
            q = tuple_in['fj_isBB'] * tuple_in['fj_isH']
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isCC'] * tuple_in['fj_isH']
            h = h.view(numpy.ndarray)
            return numpy.vstack((q, h)).transpose()


#####################################
class TrainData_DeepDoubleCvB_db_cpf_sv_reduced(TrainData_DeepDoubleX_db_cpf_sv_reduced
                                                ):
    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses = ['Hbb', 'Hcc']
        if tuple_in is not None:
            q = tuple_in['fj_isBB'] * tuple_in['fj_isH']
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isCC'] * tuple_in['fj_isH']
            h = h.view(numpy.ndarray)
            return numpy.vstack((q, h)).transpose()


class TrainData_DeepDoubleCvB_db_cpf_sv_reduced_incglu(
        TrainData_DeepDoubleX_db_cpf_sv_reduced):
    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses = ['bb', 'cc']
        if tuple_in is not None:
            q = tuple_in['fj_isBB']
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isCC']
            h = h.view(numpy.ndarray)
            return numpy.vstack((q, h)).transpose()


#####################################
#          DeepDoubleBvL            #
#####################################
class TrainData_DeepDoubleB_db(TrainData_DeepDoubleX_db):
    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses = ['QCD', 'Hbb']
        if tuple_in is not None:
            q = tuple_in['fj_isQCD'] * tuple_in['sample_isQCD']
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isBB'] * tuple_in['fj_isH']
            h = h.view(numpy.ndarray)

            return numpy.vstack((q, h)).transpose()


#####################################
#          DeepDoubleBvL            #
#####################################
class TrainData_DeepDoubleB_db_cpf_sv_reduced(TrainData_DeepDoubleX_db_cpf_sv_reduced):
    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses = ['QCD', 'Hbb']
        if tuple_in is not None:
            q = tuple_in['fj_isQCD'] * tuple_in['sample_isQCD']
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isBB'] * tuple_in['fj_isH']
            h = h.view(numpy.ndarray)

            return numpy.vstack((q, h)).transpose()

        
class TrainData_DeepDoubleXC_db_pf_cpf_sv(TrainData_DeepDoubleX_db_pf_cpf_sv):
    def __init__(self):
        TrainData_DeepDoubleX_db_pf_cpf_sv.__init__(self)
        self.weight = True
        self.remove = False
        self.referenceclass = 'QCD'

        self.weight_binY = numpy.arange(20, 251, 1, dtype=float)
    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses = ['QCD', 'Hcc']
        if tuple_in is not None:
            q = tuple_in['fj_isQCD'] * tuple_in['sample_isQCD']
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isCC'] * tuple_in['fj_isH']
            h = h.view(numpy.ndarray)

            return numpy.vstack((q, h)).transpose()
        
class TrainData_DeepDoubleXC_pruned(TrainData_DeepDoubleX_bprunedshort):
    def __init__(self):
        TrainData_DeepDoubleX_bprunedshort.__init__(self)
        self.weight = True
        self.remove = False
        self.referenceclass = 'QCD'

        self.weight_binY = numpy.arange(20, 251, 1, dtype=float)
    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses = ['QCD', 'Hcc']
        if tuple_in is not None:
            q = tuple_in['fj_isQCD'] * tuple_in['sample_isQCD']
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isCC'] * tuple_in['fj_isH']
            h = h.view(numpy.ndarray)

            return numpy.vstack((q, h)).transpose()


class TrainData_DeepDoubleXCvB_db_pf_cpf_sv(TrainData_DeepDoubleX_db_pf_cpf_sv):
    def __init__(self):
        TrainData_DeepDoubleX_db_pf_cpf_sv.__init__(self)
        self.weight = True
        self.remove = False
        self.referenceclass = 'Hbb'

        self.weight_binY = numpy.arange(20, 251, 1, dtype=float)
    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses = ['Hbb', 'Hcc']
        if tuple_in is not None:
            q = tuple_in['fj_isBB'] * tuple_in['fj_isH']
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isCC'] * tuple_in['fj_isH']
            h = h.view(numpy.ndarray)

            return numpy.vstack((q, h)).transpose()
        

class TrainData_DeepDoubleXCvB_pruned(TrainData_DeepDoubleX_bprunedshort):
    def __init__(self):
        TrainData_DeepDoubleX_bprunedshort.__init__(self)
        self.weight = True
        self.remove = False
        self.referenceclass = 'Hbb'

        self.weight_binY = numpy.arange(20, 251, 1, dtype=float)
    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses = ['Hbb', 'Hcc']
        if tuple_in is not None:
            q = tuple_in['fj_isBB'] * tuple_in['fj_isH']
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isCC'] * tuple_in['fj_isH']
            h = h.view(numpy.ndarray)

            return numpy.vstack((q, h)).transpose()


class TrainData_DeepDoubleX3_db_pf_cpf_sv(TrainData_DeepDoubleX_db_pf_cpf_sv):
    def __init__(self):
        TrainData_DeepDoubleX_db_pf_cpf_sv.__init__(self)
        self.weight = True
        self.remove = False
        self.referenceclass = 'QCD'

        self.weight_binY = numpy.arange(20, 251, 1, dtype=float)
    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses = ["QCD", 'Hbb', 'Hcc']
        if tuple_in is not None:
            q = tuple_in['fj_isQCD'] * tuple_in['sample_isQCD']
            q = q.view(numpy.ndarray)
            h1 = tuple_in['fj_isBB'] * tuple_in['fj_isH']
            h1 = h1.view(numpy.ndarray)
            h2 = tuple_in['fj_isCC'] * tuple_in['fj_isH']
            h2 = h2.view(numpy.ndarray)

            return numpy.vstack((q, h1, h2)).transpose()
        

#####################################
#          DeepDoubleBvL            #
#####################################
class TrainData_DeepDoubleB_db_pf_cpf_sv(TrainData_DeepDoubleX_db_pf_cpf_sv):
    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses = ['QCD', 'Hbb']
        if tuple_in is not None:
            q = tuple_in['fj_isQCD'] * tuple_in['sample_isQCD']
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isBB'] * tuple_in['fj_isH']
            h = h.view(numpy.ndarray)

            return numpy.vstack((q, h)).transpose()


class TrainData_DeepDoubleB_pruned(TrainData_DeepDoubleX_pruned):
    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses = ['QCD', 'Hbb']
        if tuple_in is not None:
            q = tuple_in['fj_isQCD'] * tuple_in['sample_isQCD']
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isBB'] * tuple_in['fj_isH']
            h = h.view(numpy.ndarray)

            return numpy.vstack((q, h)).transpose()


class TrainData_DeepDoubleB_apruned(TrainData_DeepDoubleX_apruned):
    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses = ['QCD', 'Hbb']
        if tuple_in is not None:
            q = tuple_in['fj_isQCD'] * tuple_in['sample_isQCD']
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isBB'] * tuple_in['fj_isH']
            h = h.view(numpy.ndarray)

            return numpy.vstack((q, h)).transpose()


class TrainData_DeepDoubleB_bpruned(TrainData_DeepDoubleX_bpruned):
    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses = ['QCD', 'Hbb']
        if tuple_in is not None:
            q = tuple_in['fj_isQCD'] * tuple_in['sample_isQCD']
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isBB'] * tuple_in['fj_isH']
            h = h.view(numpy.ndarray)

            return numpy.vstack((q, h)).transpose()
        
        
class TrainData_DeepDoubleB_bprunedshort(TrainData_DeepDoubleX_bprunedshort):
    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses = ['QCD', 'Hbb']
        if tuple_in is not None:
            q = tuple_in['fj_isQCD'] * tuple_in['sample_isQCD']
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isBB'] * tuple_in['fj_isH']
            h = h.view(numpy.ndarray)

            return numpy.vstack((q, h)).transpose()



#######################################
#          Multi-Classifier           #
#######################################
class TrainData_DeepDoubleX_db_cpf_sv_reduced_3lab(
        TrainData_DeepDoubleX_db_cpf_sv_reduced):
    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses = ['QCD', 'Hcc', 'Hbb']
        if tuple_in is not None:
            q = tuple_in["fj_isQCD"] * tuple_in['sample_isQCD']
            q = q.view(numpy.ndarray)
            h1 = tuple_in['fj_isCC'] * tuple_in['fj_isH']
            h2 = tuple_in['fj_isBB'] * tuple_in['fj_isH']
            h1 = h1.view(numpy.ndarray)
            h2 = h2.view(numpy.ndarray)
            return numpy.vstack((q, h1, h2)).transpose()


#######################################
class TrainData_DeepDoubleX_db_cpf_sv_reduced_5lab(
        TrainData_DeepDoubleX_db_cpf_sv_reduced):
    def __init__(self):
        TrainData_DeepDoubleX_db_cpf_sv_reduced.__init__(self)
        self.ignore_when_weighting = [
            'gbb'
        ]  # to be used only with weight reference 'lowest'

    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses = ['Light', 'Hcc', 'Hbb', 'gcc', 'gbb']
        if tuple_in is not None:
            q = tuple_in["fj_isQCD"] * tuple_in["fj_isNonCC"] * tuple_in[
                "fj_isNonBB"] * tuple_in['sample_isQCD']
            q = q.view(numpy.ndarray)
            h1 = tuple_in['fj_isCC'] * tuple_in['fj_isH']
            h2 = tuple_in['fj_isBB'] * tuple_in['fj_isH']
            q1 = tuple_in['fj_isCC'] * tuple_in['fj_isQCD'] * tuple_in['sample_isQCD']
            q2 = tuple_in['fj_isBB'] * tuple_in['fj_isQCD'] * tuple_in['sample_isQCD']
            h1 = h1.view(numpy.ndarray)
            h2 = h2.view(numpy.ndarray)
            q1 = q1.view(numpy.ndarray)
            q2 = q2.view(numpy.ndarray)
            return numpy.vstack((q, h1, h2, q1, q2)).transpose()


class TrainData_DeepDoubleX_db_cpf_sv_reduced_7lab(
        TrainData_DeepDoubleX_db_cpf_sv_reduced):
    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses = ['Light', 'Hcc', 'Hbb', 'Zcc', 'Zbb', 'gcc', 'gbb']
        if tuple_in is not None:
            q = tuple_in["fj_isQCD"] * tuple_in["fj_isNonCC"] * tuple_in[
                "fj_isNonBB"] * tuple_in['sample_isQCD']
            q = q.view(numpy.ndarray)
            h1 = tuple_in['fj_isCC'] * tuple_in['fj_isH']
            h2 = tuple_in['fj_isBB'] * tuple_in['fj_isH']
            z1 = tuple_in['fj_isCC'] * tuple_in['fj_isZ']
            z2 = tuple_in['fj_isBB'] * tuple_in['fj_isZ']
            q1 = tuple_in['fj_isCC'] * tuple_in['fj_isQCD'] * tuple_in['sample_isQCD']
            q2 = tuple_in['fj_isBB'] * tuple_in['fj_isQCD'] * tuple_in['sample_isQCD']
            h1 = h1.view(numpy.ndarray)
            h2 = h2.view(numpy.ndarray)
            z1 = z1.view(numpy.ndarray)
            z2 = z2.view(numpy.ndarray)
            q1 = q1.view(numpy.ndarray)
            q2 = q2.view(numpy.ndarray)
            return numpy.vstack((q, h1, h2, z1, z2, q1, q2)).transpose()


class TrainData_DeepDoubleB_reference(TrainData_DeepDoubleB_db_cpf_sv_reduced):
    def __init__(self):
        TrainData_DeepDoubleB_db_cpf_sv_reduced.__init__(self)
        self.weight = True
        self.remove = False
        self.referenceclass = 'lowest'


class TrainData_DeepDoubleB_r_massrw(TrainData_DeepDoubleB_db_cpf_sv_reduced):
    def __init__(self):
        TrainData_DeepDoubleB_db_cpf_sv_reduced.__init__(self)
        self.weight = True
        self.remove = False
        self.referenceclass = 'lowest'

        self.weight_binY = numpy.arange(20, 254, 4, dtype=float)


class TrainData_DeepDoubleB_fullinputs(TrainData_DeepDoubleB_db_pf_cpf_sv):
    def __init__(self):
        TrainData_DeepDoubleB_db_pf_cpf_sv.__init__(self)
        self.weight = True
        self.remove = False
        self.referenceclass = 'lowest'

        self.weight_binY = numpy.arange(20, 254, 4, dtype=float)


class TrainData_2prong(TrainData_DeepDoubleB_db_cpf_sv_reduced):
    def __init__(self):
        TrainData_DeepDoubleB_db_cpf_sv_reduced.__init__(self)
        self.weight = True
        self.remove = False
        self.referenceclass = 'lowest'

        self.weight_binY = numpy.array([
            20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180,
            190, 200, 210, 220, 230, 240, 250
        ],
                                       dtype=float)

    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses = ['QCD', 'H']
        if tuple_in is not None:
            q = tuple_in['fj_isQCD'] * tuple_in['sample_isQCD']
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isH']
            h = h.view(numpy.ndarray)

            return numpy.vstack((q, h)).transpose()


class TrainData_DeepDoubleB_fullinputs_massrw(TrainData_DeepDoubleB_db_pf_cpf_sv):
    def __init__(self):
        TrainData_DeepDoubleB_db_cpf_sv_reduced.__init__(self)
        self.weight = True
        self.remove = False
        self.referenceclass = 'lowest'


class TrainData_DeepDoubleB_fullinputs_wq(TrainData_DeepDoubleB_db_pf_cpf_sv):
    def __init__(self):
        TrainData_DeepDoubleB_db_pf_cpf_sv.__init__(self)
        self.weight = True
        self.remove = False
        self.referenceclass = 'QCD'

        self.weight_binY = numpy.arange(20, 251, 1, dtype=float)
        

class TrainData_DeepDoubleB_pruned1(TrainData_DeepDoubleB_pruned):
    def __init__(self):
        TrainData_DeepDoubleB_pruned.__init__(self)
        self.weight = True
        self.remove = False
        self.referenceclass = 'QCD'

        self.weight_binY = numpy.arange(20, 251, 1, dtype=float)


class TrainData_DeepDoubleB_pruned2(TrainData_DeepDoubleB_apruned):
    def __init__(self):
        TrainData_DeepDoubleB_apruned.__init__(self)
        self.weight = True
        self.remove = False
        self.referenceclass = 'QCD'

        self.weight_binY = numpy.arange(20, 251, 1, dtype=float)


class TrainData_DeepDoubleB_pruned3(TrainData_DeepDoubleB_bpruned):
    def __init__(self):
        TrainData_DeepDoubleB_bpruned.__init__(self)
        self.weight = True
        self.remove = False
        self.referenceclass = 'QCD'

        self.weight_binY = numpy.arange(20, 251, 1, dtype=float)


class  TrainData_DeepDoubleB_prunedshort(TrainData_DeepDoubleB_bprunedshort):
    def __init__(self):
        TrainData_DeepDoubleB_bprunedshort.__init__(self)
        self.weight = True
        self.remove = False
        self.referenceclass = 'QCD'

        self.weight_binY = numpy.arange(20, 251, 1, dtype=float)


class TrainData_DeepDoubleB_fullinputs_mask(TrainData_DeepDoubleB_db_pf_cpf_sv):
    def __init__(self):
        TrainData_DeepDoubleB_db_pf_cpf_sv.__init__(self)
        self.weight = True
        self.remove = False
        self.referenceclass = 'QCD'

        self.weight_binY = numpy.arange(20, 251, 1, dtype=float)
        self.mask_binY = numpy.ones_like(self.weight_binY[:-1]).astype(bool)
        self.mask_binX = numpy.ones_like(self.weight_binX[:-1]).astype(bool)
        self.mask_binY[110:130] = False


class TrainData_DeepDoubleC_reference(TrainData_DeepDoubleC_db_cpf_sv_reduced):
    def __init__(self):
        TrainData_DeepDoubleC_db_cpf_sv_reduced.__init__(self)
        self.weight = True
        self.remove = False
        self.referenceclass = 'lowest'


class TrainData_DeepDoubleX_3lab(TrainData_DeepDoubleX_db_cpf_sv_reduced_3lab):
    def __init__(self):
        TrainData_DeepDoubleX_db_cpf_sv_reduced_3lab.__init__(self)
        self.weight = True
        self.remove = False
        self.referenceclass = 'lowest'


class TrainData_DeepDoubleCvB_reference(TrainData_DeepDoubleCvB_db_cpf_sv_reduced):
    def __init__(self):
        TrainData_DeepDoubleCvB_db_cpf_sv_reduced.__init__(self)
        self.weight = True
        self.remove = False
        self.referenceclass = 'lowest'


class TrainData_DeepDoubleX_reference(TrainData_DeepDoubleX_db_cpf_sv_reduced_5lab):
    def __init__(self):
        TrainData_DeepDoubleX_db_cpf_sv_reduced_5lab.__init__(self)
        self.ignore_when_weighting = [
            'gbb'
        ]  # to be used only with weight reference 'lowest'
        self.weight = True
        self.remove = False
        self.referenceclass = 'lowest'


# To create Z labels
class TrainData_DeepDoubleX_all(TrainData_DeepDoubleX_db_cpf_sv_reduced_7lab):
    def __init__(self):
        TrainData_DeepDoubleX_db_cpf_sv_reduced_7lab.__init__(self)
        self.weight = True
        self.remove = False
        self.referenceclass = 'lowest'
