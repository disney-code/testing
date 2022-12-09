from tensorflow import keras
import tensorflow as tf
import numpy as np



class FGSM:
    """
    We use FGSM to generate a batch of adversarial examples. 
    """
    def __init__(self, model, ep=0.01, isRand=True):
        """
        isRand is set True to improve the attack success rate. 
        """
        self.isRand = isRand
        self.model = model
        self.ep = ep
        
    def generate(self, x, y, randRate=1):
        """
        x: clean inputs, shape of x: [batch_size, width, height, channel] 
        y: ground truth, one hot vectors, shape of y: [batch_size, N_classes] 
        """
        fols = []
        target = tf.constant(y)
        
        xi = x.copy()
        if self.isRand:
            x = x + np.random.uniform(-self.ep * randRate, self.ep * randRate, x.shape)
            x = np.clip(x, 0, 1)
        
        x = tf.Variable(x)
        
        grads=[]
        count = 0
        constant_var=x.shape[0]
        for i in range(int(np.ceil(x.shape[0]/500))):
            print(f"count {count} to count {count+500} and total count is {constant_var}")
            with tf.GradientTape() as tape:
                x_=tf.Variable(x[count:count+500]) #shape is 500,32,32,3
                target_=tf.constant(target[count:count+500])
                count+=500
                #print(f"count:{count}/{constant_var}")
                loss = keras.losses.categorical_crossentropy(target_, self.model(x_))
                #print("loss shape line 43 : ", loss.shape)
                grads.append(tape.gradient(loss, x_))
            del tape
#             loss = keras.losses.categorical_crossentropy(target, self.model(x))
#             grads = tape.gradient(loss, x)
        grads=tf.concat(grads,axis=0)
        delta = tf.sign(grads)
        x_adv = x + self.ep * delta
        
        x_adv = tf.clip_by_value(x_adv, clip_value_min=xi-self.ep, clip_value_max=xi+self.ep)
        x_adv = tf.clip_by_value(x_adv, clip_value_min=0, clip_value_max=1)
        
        idxs = np.where(np.argmax(self.model(x_adv), axis=1) != np.argmax(y, axis=1))[0]
        print("SUCCESS:", len(idxs))
        
        x_adv, xi, target = x_adv.numpy()[idxs], xi[idxs], target.numpy()[idxs]
        print(f"line 62 x_adv.shape: {x_adv.shape}")
        x_adv, target = tf.Variable(x_adv), tf.constant(target)
        print(f"line 64 x_adv.shape: {x_adv.shape}")
        preds = self.model(x_adv).numpy()
        print(f"line 66 preds.shape: {preds.shape}")
        ginis = np.sum(np.square(preds), axis=1)
        print("line 68")
#         grads=[]
#         count = 0
    #x_adv shape is (32289,32,32,3)
    
    
# x_=tf.Variable(x[count:count+500]) #shape is 500,32,32,3
# target_=tf.constant(y[count:count+500])
# count+=500
# print("count: ",count)
# loss = keras.losses.categorical_crossentropy(target_, self.model(x_))
# #print("loss shape line 43 : ", loss.shape)
# grads.append(tape.gradient(loss, x_))
        grads=[]
        count = 0
        constant_var=x_adv.shape[0]
        for i in range(int(np.ceil(x_adv.shape[0]/500))):
            print(f"count {count} to count {count+500} and total count is {constant_var}")
            with tf.GradientTape() as tape:
                x_adv_ = tf.Variable(x_adv[count:count+500])
                target_= tf.constant(target[count:count+500])
                loss = keras.losses.categorical_crossentropy(target_, self.model(x_adv_))
                grads.append(tape.gradient(loss, x_adv_))
                count+=500
        grads=tf.concat(grads,axis=0)
            
        grad_norm = np.linalg.norm(grads.numpy().reshape(x_adv.shape[0], -1), ord=1, axis=1)
        grads_flat = grads.numpy().reshape(x_adv.shape[0], -1)
        diff = (x_adv.numpy() - xi).reshape(x_adv.shape[0], -1)
        for i in range(x_adv.shape[0]):
            i_fol = -np.dot(grads_flat[i], diff[i]) + self.ep * grad_norm[i]
            fols.append(i_fol)
            
        return x_adv.numpy(), target.numpy(), np.array(fols), ginis



class PGD:
    """
    We use PGD to generate a batch of adversarial examples. PGD could be seen as iterative version of FGSM.
    """
    def __init__(self, model, ep=0.01, step=None, epochs=10, isRand=True):
        """
        isRand is set True to improve the attack success rate. 
        """
        self.isRand = isRand
        self.model = model
        self.ep = ep
        if step == None:
            self.step = ep/6
        self.epochs = epochs
        
    def generate(self, x, y, randRate=1):
        """
        x: clean inputs, shape of x: [batch_size, width, height, channel] 
        y: ground truth, one hot vectors, shape of y: [batch_size, N_classes] 
        """
        fols = []
        target = tf.constant(y)
    
        xi = x.copy()
        if self.isRand:
            x = x + np.random.uniform(-self.ep * randRate, self.ep * randRate, x.shape)
            x = np.clip(x, 0, 1)
        
        
        
        
        x_adv = tf.Variable(x)
        for i in range(self.epochs):
            count=0
            grads_list=[]
            print(f"for epoch {i} the range is {int(np.ceil(x_adv.shape[0]/500))}")
            for j in range(int(np.ceil(x_adv.shape[0]/500))):
                print(f"For epoch {i} , count {count} to count {count+500}")
                with tf.GradientTape() as tape:
                    x_=tf.Variable(x_adv[count:count+500])
                    target_=tf.constant(target[count:count+500])
                    loss = keras.losses.categorical_crossentropy(target_, self.model(x_))
                    grads_list.append(tape.gradient(loss, x_))
                    count+=500
                del tape
            grads = tf.concat(grads_list,axis=0)
            delta = tf.sign(grads)
            x_adv.assign_add(self.step * delta)
            x_adv = tf.clip_by_value(x_adv, clip_value_min=xi-self.ep, clip_value_max=xi+self.ep)
            x_adv = tf.clip_by_value(x_adv, clip_value_min=0, clip_value_max=1)
            x_adv = tf.Variable(x_adv)
        
        idxs = np.where(np.argmax(self.model(x_adv), axis=1) != np.argmax(y, axis=1))[0]
        print("SUCCESS:", len(idxs))
        print(f"idxs shape: {idxs.shape}")
        x_adv, xi, target = x_adv.numpy()[idxs], xi[idxs], target.numpy()[idxs]
        #print(f"line 134 x_adv shape: {x_adv.shape}")
        x_adv, target = tf.Variable(x_adv), tf.constant(target)
        print(f"line 162 x_adv shape: {x_adv.shape}")
        preds = self.model(x_adv).numpy()
        #print("line 138")
        ginis = np.sum(np.square(preds), axis=1)
        
        count=0
        grads=[]
        total_count=x_adv.shape[0]
        for k in range(int(np.ceil(x_adv.shape[0]/500))):
            print(f"In line 171 count {count} to count {count+500} , total count is {total_count}")
            with tf.GradientTape() as tape:
                x_=tf.Variable(x_adv[count:count+500])
                #print("line 139 in attack.py ran")
                target_=tf.constant(target[count:count+500])
                loss = keras.losses.categorical_crossentropy(target_, self.model(x_))
                grads.append(tape.gradient(loss, x_))
                count+=500
            del tape
            
        grads = tf.concat(grads,axis=0)
        grad_norm = np.linalg.norm(grads.numpy().reshape(x_adv.shape[0], -1), ord=1, axis=1)
        grads_flat = grads.numpy().reshape(x_adv.shape[0], -1)
        diff = (x_adv.numpy() - xi).reshape(x_adv.shape[0], -1)
        for i in range(x_adv.shape[0]):
            i_fol = -np.dot(grads_flat[i], diff[i]) + self.ep * grad_norm[i]
            fols.append(i_fol)
  
        return x_adv.numpy(), target.numpy(), np.array(fols), ginis
    

  
    