'''
Created on May 24, 2022
@author: Ioannis Stefanou & Filippo Masi
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed 
# = '0' all messages are logged (default behavior)
# = '1' INFO messages are not printed
# = '3' INFO, WARNING, and ERROR messages are not printed
import tensorflow as tf
tf.keras.backend.set_floatx('float64') # set tensorflow floating precision

import numpy as np # manipulation of arrays

class AIUserMaterial3D():
    """
    AI Material class
    """
    def __init__(self, ANN_filename, p_nIsvars):
        """
        Load ANN network
        
        :param ANN_filename: ANN filename with path
        :type string
        :param p_nIsvars: number of internal state variables
        :type integer
        """
        
        self.model = tf.saved_model.load(ANN_filename)        
        self.p_nIsvars = p_nIsvars
            
    def predict_AI_wrapper(self,deGP,svarsGP_t):
        """
        User material at a Gauss point
        
        :param deGP: generalized deformation vector at GP - input
        :type deGP: numpy array
        :param svarsGP_t: state variables at GP - input/output
        :type svarsGP_t: numpy array
        
        :return: generalized stress at GP output, state variables at GP - output, jacobian at GP - output
        :rtype: numpy array, numpy array, numpy array
        
        
        Wrapper to predict material response at the Gauss point via an Artificial Neural Network (model)
        
        The model is called with inputs of size 
        :param inputs: state variables, generalized deformation vector at GP
        :type inputs: numpy array (concatenate)
        :shape inputs: (1, 14 + self.p_nIsvars) = (1, 36), axis=0 represents the batch size (and should not modified)
        :call self.model(inputs,training=False)
        :return call: stressGP_t, svarsGP_t, dsdeGP_t with batch_size = 1, thus [0] squeeze the arrays along axis=0
        
        Note: the material response is normalized.
        """
        
        inputs = np.expand_dims(np.hstack((svarsGP_t[:12+self.p_nIsvars],
                                           deGP)),
                                           0)  

        stressGP_t,svarsGP_t,dsdeGP_t = self.model(inputs,training=False)
        stressGP_t = stressGP_t.numpy()[0]
        svarsGP_t = svarsGP_t.numpy()[0]
        dsdeGP_t = dsdeGP_t.numpy()[0]
        return stressGP_t, svarsGP_t, dsdeGP_t

    def usermatGP(self,stressGP_t,deGP,svarsGP_t,dsdeGP_t,dt,GP_id,aux_deGP=np.zeros(1)):
        """
        User material at a Gauss point
        
        :param stressGP_t: generalized stress at GP - input/output
        :type stressGP_t: numpy array
        :param deGP: generalized deformation vector at GP - input
        :type deGP: numpy array
        :param aux_deGP: auxiliary generalized deformation vector at GP - input
        :type aux_deGP: numpy array
        :param svarsGP_t: state variables at GP - input/output
        :type svarsGP_t: numpy array
        :param dsdeGP_t: jacobian at GP - output
        :type dsde_t: numpy array
        :param dt: time increment
        :type dt: double
        :param GP_id: Gauss Point id (global numbering of all Gauss Points in the problem) - for normal materials is of no use
        :type GP_id: integer
        """       

        stressGP_t[:], svarsGP_t[:], dsdeGP_t[:] = self.predict_AI_wrapper(deGP,svarsGP_t)        
        
        return 
    