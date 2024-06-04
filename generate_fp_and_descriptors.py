from rdkit.Chem import AllChem, Descriptors
from rdkit import Chem
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import sys
import pandas as pd
import numpy as np
from rdkit.Chem import MACCSkeys
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import RDKFingerprint
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
#from rdkit.Chem.Fingerprints import FingerprintMols
import os
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import MinMaxScaler
#from rdkit.Chem.AtomPairs import Pairs
import tempfile
import shutil

class Features_Generations:
    def __init__(self, csv_path, features_type):
        self.csv_path = csv_path
        self.features_type = features_type
        self.temp_dir = tempfile.mkdtemp()
        
    def toSDF(self, smiles):
        """
        Converts smiles into sdf format which is used to generate fingerprints in TPATF, TPAPF, and PHYC
        """
        # Get mol format of smiles
        mol = Chem.MolFromSmiles(smiles)
        
        # Compute 2D coordinates
        AllChem.Compute2DCoords(mol)
        mol.SetProp("smiles", smiles)
        
        w = Chem.SDWriter(os.path.join(self.temp_dir, "temp.sdf"))
        w.write(mol)
        w.flush()

    def _cleanup(self):
        """
        Removes the temporary files temp.sdf and temp.csv files
        """
        shutil.rmtree(self.temp_dir)

    def toTPATF(self):
        """
        Calculates the topological pharmacophore atomic triplets fingerprints
        Parameters
        ----------
        input : sdf file
            Sdf file is created using 'to_SDF()' 
        
        return : list
            returns the features in list form
        """
        features = []
        script_path = "/Users/gvin/Downloads/mayachemtools/bin/TopologicalPharmacophoreAtomTripletsFingerprints.pl"
          
        # Now generate the TPATF features
        # Check if the sdf file exists
        
        if not os.path.isfile(os.path.join(self.temp_dir, "temp.sdf")): return None
        command = "perl " + script_path + " -r " + os.path.join(self.temp_dir, "temp") + " --AtomTripletsSetSizeToUse FixedSize -v ValuesString -o " + os.path.join(self.temp_dir, "temp.sdf")
        os.system(command)
        
        with open(os.path.join(self.temp_dir, "temp.csv"), 'r') as f:
            for line in f.readlines():
                if "Cmpd" in line:
                    line = line.split(';')[5].replace('"','')
                    features = [int(i) for i in line.split(" ")]

        return features
    
    def toTPAPF(self):
        """
        Calculates the topological pharmacophore atomic pair fingerprints
        
        Parameters
        ----------
        input : sdf file
            Sdf file is created using 'to_SDF()' 
        
        return : list
            returns the features in list form
        """
        
        script_path = "/Users/gvin/Downloads/mayachemtools/bin/TopologicalPharmacophoreAtomPairsFingerprints.pl"
        
        # Generate TPAPF features
        # Check if the sdf file exists
        if not os.path.isfile(os.path.join(self.temp_dir, "temp.sdf")): return None
        command = "perl " + script_path + " -r " + os.path.join(self.temp_dir, "temp") + " --AtomPairsSetSizeToUse FixedSize -v ValuesString -o " + os.path.join(self.temp_dir, "temp.sdf")
        os.system(command)
        
        with open(os.path.join(self.temp_dir, "temp.csv"), 'r') as f:
            for line in f.readlines():
                if "Cmpd" in line:
                    line = line.split(';')[5].replace('"','')
                    features = [int(i) for i in line.split(" ")]
        
        return features

    def toPHYC(self):
        """
        Calculates the phisiocochemical properties
        
        Parameters
        ----------
        input : sdf file
            Sdf file is created using 'to_SDF()' 
        
        return : list
            returns the features in list form
        """
        script_path = "/Users/gvin/Downloads/mayachemtools/bin/CalculatePhysicochemicalProperties.pl"
        
        # Now generate the PHYS features
        # Check if the sdf file exists
        if not os.path.isfile(os.path.join(self.temp_dir, "temp.sdf")): return None
        command = "perl " + script_path + " -r " + os.path.join(self.temp_dir, "temp")+" -o " + os.path.join(self.temp_dir,"temp.sdf")
        os.system(command)
        
        with open(os.path.join(self.temp_dir, "temp.csv"), 'r') as f:
            for line in f.readlines():

                if "Cmp" in line:
                    line = line.replace('"','')
                    line = ','.join(line.split(',')[1:])
                    features = [float(i) for i in line.split(",")]

        return features
   
    def _cleanup(self):
        """
        Removes the temporary files temp.sdf and temp.csv files
        """
        shutil.rmtree(self.temp_dir) 


    def tpatf_fp(self):
        """
        Receives the csv file which is used to generate TPATF fingerprints (2692) and saves as numpy file
        
        Parameter
        ---------
        
        input smiles : str
            Compouds in the form of smiles are used
    
        return : np.array
            Features are saved in the form of numpy files
        
        """
        df= pd.read_csv(self.csv_path)
        smiles_list = df['smiles'].tolist()
        
        fingerprints = []
        not_found = []

        for i in tqdm(range(len(smiles_list))):
            try:
                self.toSDF(smiles_list[i])  
                
                features = fg.toTPATF()
                
                fingerprints.append(features)
            except:
                
                fingerprints.append(np.nan)
                not_found.append(i)
                pass

        # Clean up the temporary files
        self._cleanup()
        
        # Drops rowns if features are not found
        df.drop(not_found, axis=0,inplace=True)
        print('Number of FPs not found: {}'.format(len(not_found)))
        '''
        df.reset_index(drop=True, inplace=True)
        
        # Encoding categorical data
        labelencoder = LabelEncoder()                       
        Y = labelencoder.fit_transform(df['Label'].values)
        Y = Y.reshape(Y.shape[0],1)
        
        print('Output shape: {}'.format(Y.shape))
        '''
        fp_array = ( np.asarray((fingerprints), dtype=object) )
        
        # Drop rows from array where FP not generated
        X = np.delete(fp_array, not_found, axis=0)
        
        X = np.vstack(X).astype(np.float64)
        
        print('Input shape: {}'.format(X.shape))
        
        #Concatenating input and output array
        final_array = X #np.concatenate((X, Y), axis=1)
        '''
        # Removing rows, from final_array, where duplicate FPs are present
        final_array_slice = final_array[:, 0:(final_array.shape[1]-1)]
        _, unq_row_indices = np.unique(final_array_slice,return_index=True,axis=0)
        final_array_unique = final_array[unq_row_indices]
        
        final_numpy_array = np.asarray((final_array_unique), dtype = np.float32)
        print('Number of Duplicate FPs: {}'.format(final_array.shape[0] - final_array_unique.shape[0]))
    
        print('Final Numpy array shape: {}'.format(final_array_unique.shape))
        '''
        print('Final Numpy array shape: {}'.format(final_array.shape))
        final_numpy_array = np.asarray((final_array), dtype=np.float64)
        return final_numpy_array  
                           
    def tpapf_fp(self):
        """
        Receives the csv file which is used to generate TPAPF fingerprints (150) and saves as numpy file
        
        Parameter
        ---------
        
        input smiles : str
            Compouds in the form of smiles are used
    
        return : np.array
            Features are saved in the form of numpy files
        
        """
        df= pd.read_csv(self.csv_path)
        smiles_list = df['smiles'].tolist()

        fingerprints = []
        not_found = []

        for i in tqdm(range(len(smiles_list))):
            try:
                print('to_sdf is called')
                self.toSDF(smiles_list[i])  
                
                print('came back from tosdf and getfeatuers_calling')
                features = fg.toTPAPF()
                
                fingerprints.append(features)
            except:
                
                fingerprints.append(np.nan)
                not_found.append(i)
                pass
            
        # Clean up the temporary files            
        self._cleanup()
        
        df.drop(not_found, axis=0,inplace=True)
        print('Number of FPs not found: {}'.format(len(not_found)))
        
        df.reset_index(drop=True, inplace=True)
        
        labelencoder = LabelEncoder() 
        Y = labelencoder.fit_transform(df['Label'].values)
        Y = Y.reshape(Y.shape[0],1)
        
        print('Output shape: {}'.format(Y.shape))
    
        fp_array = ( np.asarray((fingerprints), dtype=object) )
        
        #Drop rows from array where FP not generated
        X = np.delete(fp_array, not_found, axis=0)
        #Save as dytpe = np.float32
        X = np.vstack(X).astype(np.float32) 
        
        print('Input shape: {}'.format(X.shape))
        #Concatenating input and output array
        final_array = np.concatenate((X, Y), axis=1)
        
        # Removing rows from final_array, where duplicate FPs are present
        final_array_slice = final_array[:, 0:(final_array.shape[1]-1)]
        _, unq_row_indices = np.unique(final_array_slice,return_index=True,axis=0)
        final_array_unique = final_array[unq_row_indices]
        final_numpy_array = np.asarray((final_array_unique), dtype = np.float32)
        
        print('Number of Duplicate FPs: {}'.format(final_array.shape[0] - final_array_unique.shape[0]))
    
        print('Final Numpy array shape: {}'.format(final_array_unique.shape))
        
        return final_numpy_array     
    
    def phyc(self):
        """
        Receives the csv file which is used to generate Physicochemical properties (8) and saves as numpy file
        
        Parameter
        ---------
        
        input smiles : str
            Compouds in the form of smiles are used
    
        return : np.array
            Features are saved in the form of numpy files
        
        """
        df= pd.read_csv(self.csv_path)
        smiles_list = df['smiles'].tolist()

        fingerprints = []
        not_found = []

        for i in tqdm(range(len(smiles_list))):
            try:
                self.toSDF(smiles_list[i])  
                features = fg.toPHYC()
                fingerprints.append(features)
            
            except:
                fingerprints.append(np.nan)
                not_found.append(i)
                pass
            
        # Clean up the temporary files    
        self._cleanup()
        
        df.drop(not_found, axis=0,inplace=True)
        print('Number of FPs not found: {}'.format(len(not_found)))
        
        df.reset_index(drop=True, inplace=True)
        
        labelencoder = LabelEncoder()
        
        Y = labelencoder.fit_transform(df['Label'].values)
        Y = Y.reshape(Y.shape[0],1)
        
        print('Output shape: {}'.format(Y.shape))
        
        fp_array = ( np.asarray((fingerprints), dtype=object) )
        #Drop rows from array where FP not generated
        X = np.delete(fp_array, not_found, axis=0)   
        X = np.vstack(X).astype(np.float32)
        
        print('Input shape: {}'.format(X.shape))
        
        final_array = np.concatenate((X, Y), axis=1)
        
        # Removing rows from final_array, where duplicate FPs are present
        final_array_slice = final_array[:, 0:(final_array.shape[1]-1)]
        _, unq_row_indices = np.unique(final_array_slice,return_index=True,axis=0)
        final_array_unique = final_array[unq_row_indices]
        final_numpy_array = np.asarray((final_array_unique), dtype = np.float32)
        print('Number of Duplicate FPs: {}'.format(final_array.shape[0] - final_array_unique.shape[0]))
    
        print('Final Numpy array shape: {}'.format(final_array_unique.shape))
        
        return final_numpy_array
    
##########################< Note!!! - Morgan_fp here = ECFP4 > ##########################
########< https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints >########
#########################################################################################    
    def morgan_fp(self):
        """
        Receives the csv file which is used to generate Morgan fingerprints (1024) and saves as numpy file
        
        Parameter
        ---------
        
        input smiles : str
            Compouds in the form of smiles are used
    
        return : np.array
            Features are saved in the form of numpy files
        """
        df = pd.read_csv(self.csv_path)
        smiles_list = df['smiles'].tolist()
        
        fingerprints = []
        not_found = []
        
        for i in tqdm(range(len(smiles_list))):
            try:
                
                mol = Chem.MolFromSmiles(smiles_list[i])
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
                bits_array = (np.fromstring(fp.ToBitString(),'u1') - ord('0'))
                fingerprints.append(bits_array)
            
            except:
                
                fingerprints.append(np.nan)
                not_found.append(i)
                pass
    
        df.drop(not_found, axis=0,inplace=True)             
        
        print('Number of FPs not found: {}'.format(len(not_found)))
        '''
        df.reset_index(drop=True, inplace=True)
        labelencoder = LabelEncoder()                       
        Y = labelencoder.fit_transform(df['Label'].values)
        Y = Y.reshape(Y.shape[0],1)
        
        print('Output shape: {}'.format(Y.shape))
        '''

        fp_array = ( np.asarray((fingerprints), dtype=object) )
        X = np.delete(fp_array, not_found, axis=0)          
        X = np.vstack(X).astype(np.float64)
        
        print('Input shape: {}'.format(X.shape))
        
        final_array = X #np.concatenate((X, Y), axis=1)
        '''
        # Removing rows from final_array, where duplicate FPs are present
        final_array_slice = final_array[:, 0:(final_array.shape[1]-1)]
        _, unq_row_indices = np.unique(final_array_slice,return_index=True,axis=0)
        final_array_unique = final_array[unq_row_indices]
        final_numpy_array = np.asarray((final_array_unique), dtype = np.float32)
        print('Number of Duplicate FPs: {}'.format(final_array.shape[0] - final_array_unique.shape[0]))
        '''
        final_numpy_array = np.asarray((final_array), dtype=np.float64)
        print('Final Numpy array shape: {}'.format(final_array.shape))
        
        return final_numpy_array 
    
#    ##########################---DOCUMENTATION ECFP2 GIVEN BELOW---#####################################
#    ########< https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints >########
#    ##############################################################################################
    
    def ecfp2_fp(self):
        """
        Receives the csv file which is used to generate ecfp2 (diameter=2) fingerprints (1024) and saves as numpy file
        
        Parameter
        ---------
        
        input smiles : str
            Compouds in the form of smiles are used
    
        return : np.array
            Features are saved in the form of numpy files
        """
        df = pd.read_csv(self.csv_path)
        smiles_list = df['smiles'].tolist()
        
        fingerprints = []
        not_found = []
        
        for i in tqdm(range(len(smiles_list))):
            try:
                
                mol = Chem.MolFromSmiles(smiles_list[i])
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=1, nBits=1024)
                bits_array = (np.fromstring(fp.ToBitString(),'u1') - ord('0'))
                fingerprints.append(bits_array)
            
            except:
                
                fingerprints.append(np.nan)
                not_found.append(i)
                pass
    
        df.drop(not_found, axis=0,inplace=True)
    
        print('Number of FPs not found: {}'.format(len(not_found)))
        '''
        df.reset_index(drop=True, inplace=True)
        labelencoder = LabelEncoder()
        Y = labelencoder.fit_transform(df['Label'].values)
        Y = Y.reshape(Y.shape[0],1)
        
        print('Output shape: {}'.format(Y.shape))
        '''
        fp_array = ( np.asarray((fingerprints), dtype=object) )
        X = np.delete(fp_array, not_found, axis=0)
        X = np.vstack(X).astype(np.float64)
        
        print('Input shape: {}'.format(X.shape))
        
        final_array = X #np.concatenate((X, Y), axis=1)
        '''
        #Removing rows from final_array, where duplicate FPs are present
        final_array_slice = final_array[:, 0:(final_array.shape[1]-1)]
        _, unq_row_indices = np.unique(final_array_slice,return_index=True,axis=0)
        final_array_unique = final_array[unq_row_indices]
        final_numpy_array = np.asarray((final_array_unique), dtype = np.float32)
        print('Number of Duplicate FPs: {}'.format(final_array.shape[0] - final_array_unique.shape[0]))
        
        print('Final Numpy array shape: {}'.format(final_array_unique.shape))
        print('Type of final array: {}'.format(type(final_array_unique)))
        '''
        final_numpy_array = np.asarray((final_array), dtype=np.float64)
        print('Final Numpy array shape: {}'.format(final_array.shape))
        return final_numpy_array
    
    ##########################---DOCUMENTATION ECFP6 GIVEN BELOW---#####################################
    ########< https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints >########
    ##############################################################################################
    
    def ecfp6_fp(self):
        
        """
        Receives the csv file which is used to generate ecfp-6 (diameter = 6) fingerprints (1024) and saves as numpy file
        
        Parameter
        ---------
        
        input smiles : str
            Compouds in the form of smiles are used
    
        return : np.array
            Features are saved in the form of numpy files
        """
        df = pd.read_csv(self.csv_path)
        smiles_list = df['smiles'].tolist()
        
        fingerprints = []
        not_found = []
        
        for i in tqdm(range(len(smiles_list))):
            try:
                mol = Chem.MolFromSmiles(smiles_list[i])
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024)
                bits_array = (np.fromstring(fp.ToBitString(),'u1') - ord('0'))
                fingerprints.append(bits_array)
            
            except:
                
                fingerprints.append(np.nan)
                not_found.append(i)
                pass
    
        df.drop(not_found, axis=0,inplace=True)
    
        print('Number of FPs not found: {}'.format(len(not_found)))
        
        df.reset_index(drop=True, inplace=True)
        labelencoder = LabelEncoder()
        Y = labelencoder.fit_transform(df['Label'].values)
        Y = Y.reshape(Y.shape[0],1)
        
        print('Output shape: {}'.format(Y.shape))
        
        fp_array = ( np.asarray((fingerprints), dtype=object) )
        X = np.delete(fp_array, not_found, axis=0)  
        X = np.vstack(X).astype(np.float32)
        
        print('Input shape: {}'.format(X.shape))
        
        final_array = np.concatenate((X, Y), axis=1)
        
        # Removing rows from final_array, where duplicate FPs are present
        final_array_slice = final_array[:, 0:(final_array.shape[1]-1)]
        _, unq_row_indices = np.unique(final_array_slice,return_index=True,axis=0)
        final_array_unique = final_array[unq_row_indices]
        final_numpy_array = np.asarray((final_array_unique), dtype = np.float32)

        print('Number of Duplicate FPs: {}'.format(final_array.shape[0] - final_array_unique.shape[0]))
        
        print('Final Numpy array shape: {}'.format(final_array_unique.shape))
        print('Type of final array: {}'.format(type(final_array_unique)))
        
        return final_numpy_array 
    
    ##########################---DOCUMENTATION MACC FP GIVEN BELOW---#####################################
    ########< https://www.rdkit.org/docs/GettingStartedInPython.html#maccs-keys >########
    ##############################################################################################
    
    
    def maccs_fp(self):
        """
        Receives the csv file which is used to generate MACCS fingerprints (167) and saves as numpy file
        
        Parameter
        ---------
        
        input smiles : str
            Compouds in the form of smiles are used
    
        return : np.array
            Features are saved in the form of numpy files
        """        
        df = pd.read_csv(self.csv_path)
        smiles_list = df['smiles'].tolist()
        
        fingerprints = []
        not_found = []

        for i in tqdm(range(len(smiles_list))):
            try:
                
                mol = Chem.MolFromSmiles(smiles_list[i])
                fp = MACCSkeys.GenMACCSKeys(mol)
                bits_array = (np.fromstring(fp.ToBitString(),'u1') - ord('0'))
                fingerprints.append(bits_array)
            
            except:
                
                fingerprints.append(np.nan)
                not_found.append(i)
                pass
    
        df.drop(not_found, axis=0,inplace=True)
    
        print('Number of FPs not found: {}'.format(len(not_found)))
        '''
        df.reset_index(drop=True, inplace=True)
        labelencoder = LabelEncoder()
        Y = labelencoder.fit_transform(df['Label'].values)
        Y = Y.reshape(Y.shape[0],1)
        
        print('Output shape: {}'.format(Y.shape))
        '''
        fp_array = ( np.asarray((fingerprints), dtype=object) )
        X = np.delete(fp_array, not_found, axis=0) 
        X = np.vstack(X).astype(np.float64)
        
        print('Input shape: {}'.format(X.shape))
        
        final_array = X #np.concatenate((X, Y), axis=1)
        ''' 
        # Removing rows from final_array, where duplicate FPs are present
        final_array_slice = final_array[:, 0:(final_array.shape[1]-1)]
        _, unq_row_indices = np.unique(final_array_slice,return_index=True,axis=0)
        final_array_unique = final_array[unq_row_indices]
        
        print('Number of Duplicate FPs: {}'.format(final_array.shape[0] - final_array_unique.shape[0]))
        
        print('Final Numpy array shape: {}'.format(final_array_unique.shape))
        print('Type of final array: {}'.format(type(final_array_unique)))
        final_numpy_array = np.asarray((final_array_unique), dtype = np.float32)
        '''
        print('Final Numpy array shape: {}'.format(final_array.shape))
        final_numpy_array = np.asarray((final_array), dtype=np.float64)

        return final_numpy_array
    
    ##########################---DOCUMENTATION AVALON FP GIVEN BELOW---#####################################
    ########<https://www.rdkit.org/docs/source/rdkit.Avalon.pyAvalonTools.html>########
    ##############################################################################################
    
    def avalon_fp(self):
        """
        Receives the csv file which is used to generate avalon fingerprints (512) and saves as numpy file
        
        Parameter
        ---------
        
        input smiles : str
            Compouds in the form of smiles are used
    
        return : np.array
            Features are saved in the form of numpy files
        """
        df = pd.read_csv(self.csv_path)
        smiles_list = df['smiles'].tolist()
        
        fingerprints = []
        not_found = []
        
        for i in tqdm(range(len(smiles_list))):
            try:
                
                mol = Chem.MolFromSmiles(smiles_list[i])
                fp = pyAvalonTools.GetAvalonFP(mol, nBits=512)
                bits_array = (np.fromstring(fp.ToBitString(),'u1') - ord('0'))
                fingerprints.append(bits_array)
            
            except:
                
                fingerprints.append(np.nan)
                not_found.append(i)
                pass
    
        df.drop(not_found, axis=0,inplace=True)
    
        print('Number of FPs not found: {}'.format(len(not_found)))
        '''
        df.reset_index(drop=True, inplace=True)
        labelencoder = LabelEncoder()
        Y = labelencoder.fit_transform(df['Label'].values)
        Y = Y.reshape(Y.shape[0],1)
        
        print('Output shape: {}'.format(Y.shape))
        '''
        fp_array = ( np.asarray((fingerprints), dtype=object) )
        X = np.delete(fp_array, not_found, axis=0)   
        X = np.vstack(X).astype(np.float64)
        
        print('Input shape: {}'.format(X.shape))
        
        final_array = X #np.concatenate((X, Y), axis=1)
        '''
        # Removing rows, from final_array, where duplicate FPs are present
        final_array_slice = final_array[:, 0:(final_array.shape[1]-1)]
        _, unq_row_indices = np.unique(final_array_slice,return_index=True,axis=0)
        final_array_unique = final_array[unq_row_indices]
        
        print('Number of Duplicate FPs: {}'.format(final_array.shape[0] - final_array_unique.shape[0]))
        
        print('Final Numpy array shape: {}'.format(final_array_unique.shape))
        print('Type of final array: {}'.format(type(final_array_unique)))
        final_numpy_array = np.asarray((final_array_unique), dtype = np.float32)
        '''
        print('Final Numpy array shape: {}'.format(final_array.shape))
        final_numpy_array = np.asarray((final_array), dtype=np.float64)
        return final_numpy_array 
    
    ##########################---DOCUMENTATION RDK FP GIVEN BELOW---#####################################
    ########< https://www.rdkit.org/docs/source/rdkit.Chem.MolDb.FingerprintUtils.html >########
    ##############################################################################################
    
    def rdk_fp(self):
        """
        Receives the csv file which is used to generate rdk fingerprints (2048) and saves as numpy file
        
        Parameter
        ---------
        
        input smiles : str
            Compouds in the form of smiles are used
    
        return : np.array
            Features are saved in the form of numpy files
        """
        df = pd.read_csv(self.csv_path)
        smiles_list = df['smiles'].tolist()
    
        fingerprints = []
        not_found = []
        
        for i in tqdm(range(len(smiles_list))):
            try:
                
                mol = Chem.MolFromSmiles(smiles_list[i])
                fp = RDKFingerprint(mol, nBitsPerHash=1)
                bits_array = (np.fromstring(fp.ToBitString(),'u1') - ord('0'))
                fingerprints.append(bits_array)
            
            except:
                
                fingerprints.append(np.nan)
                not_found.append(i)
                pass
    
        df.drop(not_found, axis=0,inplace=True)
    
        print('Number of FPs not found: {}'.format(len(not_found)))
        '''
        df.reset_index(drop=True, inplace=True)
        labelencoder = LabelEncoder()
        Y = labelencoder.fit_transform(df['Label'].values)
        Y = Y.reshape(Y.shape[0],1)
    
        print('Output shape: {}'.format(Y.shape))
        '''
        fp_array = ( np.asarray((fingerprints), dtype=object) )
        X = np.delete(fp_array, not_found, axis=0)
        X = np.vstack(X).astype(np.float64)
        
        print('Input shape: {}'.format(X.shape))
        
        final_array = X #np.concatenate((X, Y), axis=1)
        '''
        # Removing rows, from final_array, where duplicate FPs are present
        final_array_slice = final_array[:, 0:(final_array.shape[1]-1)]
        _, unq_row_indices = np.unique(final_array_slice,return_index=True,axis=0)
        final_array_unique = final_array[unq_row_indices]
        
        print('Number of Duplicate FPs: {}'.format(final_array.shape[0] - final_array_unique.shape[0]))
        
        print('Final Numpy array shape: {}'.format(final_array_unique.shape))
        print('Type of final array: {}'.format(type(final_array_unique)))
        final_numpy_array = np.asarray((final_array_unique), dtype = np.float32)
        '''
        print('Final Numpy array shape: {}'.format(final_array.shape))
        final_numpy_array = np.asarray((final_array), dtype=np.float64)
        return final_numpy_array 
    
    ##########################---DOCUMENTATION ATOM PAIR FP GIVEN BELOW---#####################################
    ########<https://www.rdkit.org/docs/source/rdkit.Chem.MolDb.FingerprintUtils.html>########
    ##############################################################################################
    ## https://www.rdkit.org/docs/source/rdkit.Chem.AtomPairs.Pairs.html, But it's so hug bit vector##
    
    def atom_pair_fp(self):
        df = pd.read_csv(self.csv_path)
        smiles_list = df['smiles'].tolist()
        
        fingerprints = []
        not_found = []

        for i in tqdm(range(len(smiles_list))):
            try:
                
                mol = Chem.MolFromSmiles(smiles_list[i])
                fp = Pairs.GetAtomPairFingerprintAsIntVect(mol)
                fp._sumCache = fp.GetTotalVal() #Bit vector here will be huge, which is why taking TotalVal()
    #             bits = fp.ToBitString()
    #             bits_array = (np.fromstring(fp.ToBitString(),'u1') - ord('0'))
                fingerprints.append(fp._sumCache)
                print('fing',fingerprints)        
            except:
                
                fingerprints.append(np.nan)
                not_found.append(i)
                pass
    
        df.drop(not_found, axis=0,inplace=True)
    
        print('Number of FPs not found: {}'.format(len(not_found)))
        '''
        df.reset_index(drop=True, inplace=True)
        labelencoder = LabelEncoder()
        Y = labelencoder.fit_transform(df['Label'].values)
        Y = Y.reshape(Y.shape[0],1)
    
        print('Output shape: {}'.format(Y.shape))
        '''
        fp_array = ( np.asarray((fingerprints), dtype=object) )
        X = np.delete(fp_array, not_found, axis=0)
        X = np.vstack(X).astype(np.float64)
        print('Typeof X', type(X))
        #print(X)
        print('Input shape: {}'.format(X.shape))
        
        final_array = X #np.concatenate((X, Y), axis=1)
        '''
        # Removing rows, from final_array, where duplicate FPs are present
        final_array_slice = final_array[:, 0:(final_array.shape[1]-1)]
        _, unq_row_indices = np.unique(final_array_slice,return_index=True,axis=0)
        final_array_unique = final_array[unq_row_indices]
        
        print('Number of Duplicate FPs: {}'.format(final_array.shape[0] - final_array_unique.shape[0]))
        
        print('Final Numpy array shape: {}'.format(final_array_unique.shape))
        print('Type of final array: {}'.format(type(final_array_unique)))
        final_numpy_array = np.asarray((final_array_unique), dtype = np.float32)
        '''
        print('Final Numpy array shape: {}'.format(final_array.shape))
        final_numpy_array = np.asarray((final_array), dtype=np.float64)
        return final_numpy_array   
    
    
    ##########################---DOCUMENTATION TORSION FP GIVEN BELOW---#####################################
    ########<https://www.rdkit.org/docs/source/rdkit.Chem.MolDb.FingerprintUtils.html>########
    ##############################################################################################
    
    def torsions_fp(self):
        df = pd.read_csv(self.csv_path)
        smiles_list = df['smiles'].tolist()
        
        fingerprints = []
        not_found = []
        for i in tqdm(range(len(smiles_list))):
            try:
                
                mol = Chem.MolFromSmiles(smiles_list[i])
                fp = Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol)
                fp._sumCache = fp.GetTotalVal() #Bit vector here will be huge, which is why taking TotalVal()
                #             bits = fp.ToBitString()
                #             bits_array = (np.fromstring(fp.ToBitString(),'u1') - ord('0'))
                fingerprints.append(fp._sumCache)
            
            except:
                
                fingerprints.append(np.nan)
                not_found.append(i)
                pass
    
        df.drop(not_found, axis=0,inplace=True)
    
        print('Number of FPs not found: {}'.format(len(not_found)))
        '''
        df.reset_index(drop=True, inplace=True)
        labelencoder = LabelEncoder()
        Y = labelencoder.fit_transform(df['Label'].values)
        Y = Y.reshape(Y.shape[0],1)
    
        print('Output shape: {}'.format(Y.shape))
        '''
        fp_array = ( np.asarray((fingerprints), dtype=object) )
        X = np.delete(fp_array, not_found, axis=0)
        X = np.vstack(X).astype(np.float64)
        print('Input shape: {}'.format(X.shape))
        
        final_array = X #np.concatenate((X, Y), axis=1)
        '''
        # Removing rows, from final_array, where duplicate FPs are present
        final_array_slice = final_array[:, 0:(final_array.shape[1]-1)]
        _, unq_row_indices = np.unique(final_array_slice,return_index=True,axis=0)
        final_array_unique = final_array[unq_row_indices]
        
        print('Number of Duplicate FPs: {}'.format(final_array.shape[0] - final_array_unique.shape[0]))
        
        print('Final Numpy array shape: {}'.format(final_array_unique.shape))
        print('Type of final array: {}'.format(type(final_array_unique)))
        final_numpy_array = np.asarray((final_array_unique), dtype = np.float32)
        '''
        print('Final Numpy array shape: {}'.format(final_array.shape))
        final_numpy_array = np.asarray((final_array), dtype=np.float64)
        return final_numpy_array 
    
    ##########################---DOCUMENTATION DESCRIPTORS (physical properties) GIVEN BELOW---#####################################
    ###< https://www.rdkit.org/docs/source/rdkit.ML.Descriptors.MoleculeDescriptors.html >######
    #####<https://sourceforge.net/p/rdkit/mailman/message/30087006/>##################
    ##############################################################################################
    
    def molecule_descriptors(self):
        """
        Receives the csv file which is used to generate molecular descriptors (200) and saves as numpy file
        
        Parameter
        ---------
        
        input smiles : str
            Compouds in the form of smiles are used
    
        return : np.array
            Descriptors are saved in the form of numpy files
        """
        df = pd.read_csv(self.csv_path)
        smiles_list = df['smiles'].tolist()
        
        descriptors = []
        not_found = []
        
        for i in tqdm(range(len(smiles_list))):
            try:
                
                calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
                mol = Chem.MolFromSmiles(smiles_list[i])
                ds = calc.CalcDescriptors(mol)
                ds = np.asarray(list(ds))
                descriptors.append(ds)
            
            except:
                
                descriptors.append(np.nan)
                not_found.append(i)
                pass
    
        df.drop(not_found, axis=0,inplace=True)
    
        print('Number of Descriptors not found: {}'.format(len(not_found)))
    
        #df.reset_index(drop=True, inplace=True)
        #labelencoder = LabelEncoder()
        #Y = labelencoder.fit_transform(df['Label'].values)
        #Y = Y.reshape(Y.shape[0],1)
    
        #print('Output shape: {}'.format(Y.shape))
    
        fp_array = ( np.asarray((descriptors), dtype=object) )
        #Drop rows from array where Descriptor not generated
        X = np.delete(fp_array, not_found, axis=0)
        X = np.vstack(X).astype(np.float64)
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X) #Normalize -->( (X-X.min()) / (X.max()-X.min) ) where X.max() & X.min() are taken from within a
                                                            #column and NOT the whole numpy array
    
        print('Input shape: {}'.format(X.shape))
    
        final_array = X #np.concatenate((X, Y), axis=1)
        '''
        # Removing rows, from final_array, where duplicate Descriptors are present
        final_array_slice = final_array[:, 0:(final_array.shape[1]-1)]
        _, unq_row_indices = np.unique(final_array_slice,return_index=True,axis=0)
        final_array_unique = final_array[unq_row_indices]
        
        print('Number of Duplicate Descriptors: {}'.format(final_array.shape[0] - final_array_unique.shape[0]))
        
        print('Final Numpy array shape: {}'.format(final_array_unique.shape))
        print('Type of final array: {}'.format(type(final_array_unique)))
        final_numpy_array = np.asarray((final_array_unique), dtype = np.float64)
        '''
        print('Type of final array: {}'.format(type(final_array)))
        final_numpy_array = np.asarray((final_array), dtype=np.float64)
        return final_numpy_array

#########################----MAIN FUNCTION BELOW-------###################################


def main():
    if sys.argv[2] == 'morgan_fp':
        numpy_file = fg.morgan_fp()
    elif sys.argv[2] == 'maccs_fp':
        numpy_file = fg.maccs_fp()
    elif sys.argv[2] == 'avalon_fp':
        numpy_file = fg.avalon_fp()
    elif sys.argv[2] == 'rdk_fp':
        numpy_file = fg.rdk_fp()
    elif sys.argv[2] == 'molecule_descriptors':
        numpy_file = fg.molecule_descriptors()
    elif sys.argv[2] == 'ecfp2_fp':
        numpy_file = fg.ecfp2_fp()
    elif sys.argv[2] == 'ecfp6_fp':
        numpy_file = fg.ecfp6_fp()
    elif sys.argv[2] == 'tpatf_fp':
        numpy_file = fg.tpatf_fp()
    elif sys.argv[2] == 'tpapf_fp':
        numpy_file = fg.tpapf_fp()
    elif sys.argv[2] == 'phyc':
        numpy_file = fg.phyc()
    elif sys.argv[2] == 'atom_pair_fp':
        numpy_file = fg.atom_pair_fp()
    elif sys.argv[2] == 'torsions_fp':
        numpy_file = fg.torsions_fp()
    else:
        print('FingerPrint is not available, Please check the FingerPrint name!')
        exit()
    # Saving numpy file of fingerprints or molecule_descriptors
    np.savetxt(numpy_folder+'/'+subcell_name+'_'+sys.argv[2]+'.txt',numpy_file)

if __name__ == "__main__":
    
    numpy_folder = './smiles'
    
    # Creates folder if it does not exists
    if not os.path.isdir(numpy_folder): os.mkdir(numpy_folder)
        
    fg = Features_Generations(sys.argv[1], sys.argv[2])
   # subcell_name = sys.argv[1].split('/')[3].split('.')[0] 
    
    subcell_name = sys.argv[1].split('.')[0]
    main()

