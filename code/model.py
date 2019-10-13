from torch import nn,optim
import torch.nn.functional as F
from transformer import Transformer
from config import Config
import torch
import logging
logger = logging.getLogger(__name__)
class Encoder(nn.Module):
    def __init__(self,config):
        super(Encoder,self).__init__()
        self.config = config
        self.__dict__.update(config)
        self.encoder = Transformer(self.voc_size,n_layers=self.encoder_layers)
    def forward(self,x):
        return self.encoder(x)

class VariationalEncoder(nn.Module):
    def __init__(self,config,first_layer_hidden_dim):
        super(VariationalEncoder,self).__init__()
        self.__dict__.update(config)
        self.miu_vae_proj_layer= nn.Linear(first_layer_hidden_dim,self.vae_hidden_dim)
        self.miu_projection_module = nn.ModuleList([nn.Linear(self.vae_hidden_dim,self.vae_hidden_dim)
                                                for i in range(self.vae_layers)])
        self.sigma_vae_proj_layer = nn.Linear(first_layer_hidden_dim,self.vae_hidden_dim)
        self.sigma_projection_module = nn.ModuleList([nn.Linear(self.vae_hidden_dim,self.vae_hidden_dim)
                                                for i in range(self.vae_layers)])
    def forward(self,x):
        miu,logvar = F.relu(self.miu_vae_proj_layer(x)),F.relu(self.sigma_vae_proj_layer(x))
        for layer in zip(self.miu_projection_module,self.sigma_projection_module):
            miu,logvar = layer[0](miu),layer[1](logvar)
        return miu,logvar

class ReparameterLayer(nn.Module):
    def __init__(self,config):
        super(ReparameterLayer,self).__init__()
        self.__dict__.update(config)
    def forward(self,miu,logvar):
        eta = torch.randn_like(logvar,device=logvar.device)
        std = torch.exp(0.5*logvar)
        return miu + eta * std

class VAEDecoder(nn.Module):
    def __init__(self,config):
        super(VAEDecoder,self).__init__()
        self.__dict__.update(config)
        self.combine_size = self.encoder_hidden_dim * 2 + self.vae_hidden_dim * 2
        self.decoder_proj_layer = nn.Linear(self.combine_size,self.vae_decoder_hidden_dim)
        self.decoder_module = nn.ModuleList([nn.Linear(self.vae_decoder_hidden_dim,self.vae_decoder_hidden_dim)
                                             for i in range(self.vae_decoder_layers)])
        self.prediction_layer = nn.Linear(self.vae_decoder_hidden_dim,2)
        #self.softmax = nn.Softmax(dim=-1)

    def forward(self,entity,mention,entity_z,joint_z):
        x = torch.cat((entity,mention,entity_z,joint_z),dim=-1)
        x = self.decoder_proj_layer(x)
        for layer in self.decoder_module:
            x = layer(x)
        x = self.prediction_layer(x)
        return x
class VAEEntityLinkingModle(nn.Module):

    def __init__(self,config):
        super(VAEEntityLinkingModle,self).__init__()
        self.__dict__.update(config)
        self.entity_encoder = Encoder(config)
        self.mention_encoder = Encoder(config)
        self.mention_vae_encoder = VariationalEncoder(config,first_layer_hidden_dim=self.encoder_hidden_dim)
        self.joint_vae_encoder = VariationalEncoder(config,first_layer_hidden_dim=2*self.encoder_hidden_dim)
        self.mention_rp_layer = ReparameterLayer(config)
        self.joint_rp_layer = ReparameterLayer(config)
        self.decoder = VAEDecoder(config)
    def init_weight(self):
        modules = list(self.children())
        for child in modules:
            if not (isinstance(child,Encoder) or isinstance(child,ReparameterLayer)):
                for p in child.parameters():
                    if p.data.dim() > 1:
                        nn.init.kaiming_normal_(p.data)

        logger.info('init weight with kamming method except encoder layers')
    def parameters_number(self):
        n = sum([torch.numel(p.data) for p in self.parameters()])
        return n / 1000
    def forward(self,mention,entity):
        entity = self.entity_encoder(entity)
        mention = self.mention_encoder(mention)
        mention_mean_pooling = torch.sum(mention,dim=1) / mention.shape[1]
        entity_mean_pooling = torch.sum(entity,dim=1) / entity.shape[1]
        mention_miu,mention_logvar = self.mention_vae_encoder(mention_mean_pooling)
        cat_men_ent = torch.cat((mention_mean_pooling,entity_mean_pooling),dim=-1)
        joint_miu,joint_logvar = self.joint_vae_encoder(cat_men_ent)
        mention_z = self.mention_rp_layer(mention_miu,mention_logvar)
        joint_z = self.joint_rp_layer(joint_miu,joint_logvar)
        prediction = self.decoder(mention_mean_pooling,entity_mean_pooling,mention_z,joint_z)
        return (prediction,mention_miu,mention_logvar,joint_miu,joint_logvar)

def loss_func(out,target,mention_miu,mention_logvar,joint_miu,joint_logvar,criterion=nn.CrossEntropyLoss(reduction='mean'),gamma=0.02):
    CLSL = criterion(out,target)
    BCE = gamma * 0.5 * torch.sum( torch.exp(joint_logvar) / torch.exp(mention_logvar) +
                           (mention_miu-joint_miu) / torch.exp(mention_logvar) * (mention_miu-joint_miu) - 1 +
                           mention_logvar - joint_logvar )
    print(CLSL,BCE)
    return CLSL + BCE
def main():
    model = VAEEntityLinkingModle(Config)
    mention = torch.tensor([[1,1,2,3,4,0,0],[1,4,5,5,3,3,0],[1,0,0,0,0,0,0]],dtype=torch.long)
    entity = torch.tensor([[1,1,2,3,4,0,0],[1,4,5,5,3,3,0],[1,0,0,0,0,0,0]],dtype=torch.long)
    model.init_weight()
    prediction,mention_miu,mention_lagvar,joint_miu,joint_lagvar = model(mention,entity)
    target = torch.empty(3,dtype=torch.long).random_(2)
    loss = loss_func(prediction,target,mention_miu,mention_lagvar,joint_miu,joint_lagvar)
if __name__ == "__main__":
    main()