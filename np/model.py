import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def get_encoder():
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(32, activation="relu"),
                                        tf.keras.layers.Dense(32, activation=None)])
    return model    

class NPModel(tf.keras.Model):
    def __init__(self,x_context,y_context,x_target):
        super(NPModel,self).__init__()
        self._latent_encoder = get_encoder()
        self._deterministic_encoder = get_encoder()
        self._decoder = get_encoder()
        self.x_context = x_context
        self.y_context = y_context
    def conglomerate(self, tensor):
        return tf.reduce_mean(tensor, axis=0, keepdims=True)

    def call(self,x_target):
        params_context = self.conglomerate(self._latent_encoder(self.x_context,self.y_context)) # Context
        mu, log_sigma = tf.split(params_context,2,axis=1)
        #tf.print(mu.shape, log_sigma.shape)
        sigma = tf.exp(log_sigma)
        latent_rep = tf.random.normal(mu.shape)*sigma + mu # Reparametarisation trick - allows for gradient flow
        #tf.print(latent_rep.shape)
        deterministic_rep = self.conglomerate(self._deterministic_encoder(self.x_context,self.y_context))
        #tf.print("deterministic_rep", deterministic_rep.shape)
        representation = tf.concat([deterministic_rep, latent_rep],axis=1) # Need to tile along 0 axis and concat 1 axis
        tiled = representation * tf.ones_like(x_target)
        #tf.print(tiled.shape)
        params_decoder = self._decoder(tf.concat([tiled, x_target], axis=1))
        #tf.print(params_decoder.shape)
        return tf.concat([params_decoder, x_target], axis=1)
    
    def rms(self, y_target, output):
        params_decoder = output[:,:32]
        mu, log_sigma = tf.split(params_decoder,2,axis=1)
        mse = tf.reduce_mean(tf.math.square(y_target - mu))
        return tf.math.sqrt(mse)

    def loss(self,y_target,output):
        params_decoder = output[:,:32]
        x_target = output[:,32:]
        # - log_p(y_target|y_pred) + kl_div(latent_target || latent_context)
        mu, log_sigma = tf.split(params_decoder,2,axis=1)
        sigma = tf.exp(log_sigma)
        dist = tfp.distributions.MultivariateNormalDiag(mu,sigma)
        log_p = dist.log_prob(y_target)
        #print('log p', log_p)
        
        params_context = self.conglomerate(self._latent_encoder(self.x_context,self.y_context)) #Context
        mu_context, log_sigma_context = tf.split(params_context,2,axis=-1)
        sigma_context = tf.exp(log_sigma_context)
        prior = tfp.distributions.Normal(loc=mu_context,scale=sigma_context)
        #tf.print('mu, sigma context', mu_context, sigma_context)
        #print('y shape',y_target.shape)
        params_target = self.conglomerate(self._latent_encoder(x_target,y_target)) #Target
        mu_target, log_sigma_target = tf.split(params_target,2,axis=-1)
        sigma_target = tf.exp(log_sigma_target)
        #tf.print('mu, sigma target', mu_target, sigma_target)
        posterior = tfp.distributions.Normal(loc=mu_target,scale=sigma_target)
        #tf.print('posterior',posterior)
        kl = tf.reduce_sum(tfp.distributions.kl_divergence(posterior, prior),axis=-1,keepdims=True)
        #tf.print('kl',kl)
        #kl = tf.tile(kl,[1,500])
        loss = - tf.reduce_mean(log_p - kl /tf.cast(100,tf.float32))
        return loss

