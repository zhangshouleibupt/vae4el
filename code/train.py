from model import VAEEntityLinkingModel,loss_func
from torch import nn,optim
import  torch
from tensorboardX import SummaryWriter
import  time
from config import Config
import logging
from tqdm import tqdm
from dataset import ELDataset
from torch.utils.data import Subset,DataLoader
logger = logging.getLogger(__name__)
device = torch.device('cuda:2') if torch.cuda.is_available() and Config['use_cuda'] else torch.device('cpu')

class Trainner():
    def __init__(self,config,model,dataset,criterion):
        self.__dict__.update(config)
        self.model = model
        self.dataset = dataset
        self.train_set = Subset(dataset,list(range(self.train_dataset_nums)))
        self.test_set = Subset(dataset,list(range(self.train_dataset_nums+1,len(dataset))))
        self.criterion = criterion
        self.optimizer = optim.Adam(model.parameters(),lr=self.learning_rate)
        self.writer = SummaryWriter('../runs')
        self.cur_loss_iter_num = 0
        self.precision = 0.0
        self.recall = 0.0
        self.f_score = 0.0
    def train(self):
        self.model.to(device)
        sampler = RandomSampler(self.train_set)
        train_dl = DataLoader(self.train_set,sampler=sampler,batch_size=self.batch_size)
        for epoch in range(self.epochs):
            self._update_parameters(train_dl,epoch)
        logger.info('training finished the blow are the detail on train set')
    def evaluate(self):
        pass
    def _update_parameters(self,dataloader,cur_epoch):
        self.model.to(device)
        l = len(dataloader)
        for i,batch in tqdm(enumerate(dataloader)):
            self.optimizer.zero_grad()
            mention,entity,label = list(map(lambda x:x.to(device),batch))
            out,mention_miu,mention_logvar,joint_miu,joint_logvar = model(mention,entity)
            input = (prediction,mention_miu,mention_logvar,joint_miu,joint_logvar)
            loss = self.criterion(*input)
            loss.backward()
            self.optimizer.step()
            _,prediction = out.topk(out,1)
            p,r,f_score = self._metrics(prediction,label)
            if (cur_epoch * l + i + 1) % self.print_every == 0:
                self.writer.add_scalar('loss',loss.item(),self.cur_loss_iter_num)
                self.writer.add_scalar('precision',p,self.cur_loss_iter_num)
                self.writer.add_scalar('recall',r,self.cur_loss_iter_num)
                self.writer.add_scalar('f_score',f_score,self.cur_loss_iter_num)
                self.cur_loss_iter_num += 1
            logger.info('finished %d batch in epoch %d,the below are reporter in this batch' % (i, cur_epoch))
            logger.info('loss is: %.2f, linking precion is: %.2f recall is: %.2f f_score is: %.2f' % (loss.item(), p, r, f_score))
        logger.info('finished in %d epoch now loss still remain %d epochs still not meet early stopping'%(cur_epoch,self.epochs-cur_epoch-1))
        #do some detail reporter on train set and eval set

def main():
    whole_dataset = EDLDataset('../data/aida.data')
    el_model = VAEEntityLinkingModel(config)
    trainner = Trainner(el_model,whole_dataset,loss_func)
    trainner.train()
if __name__ == "__main__":
    main()