from model import VAEEntityLinkingModel,loss_func
from torch import nn,optim
import  torch
from tensorboardX import SummaryWriter
import  time
from config import Config
import logging
from tqdm import tqdm
from dataset import ELDataset
from torch.utils.data import Subset,DataLoader,RandomSampler
from fairseq.data import Dictionary
from sklearn.metrics import f1_score
import numpy as np
import os
import  warnings
import math
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
device = torch.device('cuda:2') if torch.cuda.is_available() and Config['use_cuda'] else torch.device('cpu')

class Trainner():
    def __init__(self,config,model,dataset,criterion,mode='Train'):
        self.__dict__.update(config)
        self.model = model
        self.dataset = dataset
        self.train_set = Subset(dataset,
                                list(range(self.train_dataset_nums)))
        self.test_set = Subset(dataset,
                               list(range(self.train_dataset_nums,len(dataset))))
        self.criterion = criterion
        self.optimizer = optim.Adam(model.parameters(),lr=self.learning_rate)
        self.now_time = time.strftime("%m%d%-H%M-%S",time.localtime())
        self.writer = SummaryWriter('../runs/'+self.now_time)
        self.max_f1_on_test = 0.0
    def train(self):
        print("the model tensorboad dir is in %s" %self.now_time)
        self.model.to(device)
        train_dl = DataLoader(self.train_set,
                              sampler=RandomSampler(self.train_set),
                              batch_size=self.batch_size)
        print('finished load train dataloader')
        test_dl = DataLoader(self.test_set,
                             sampler=RandomSampler(self.test_set),
                             batch_size=self.batch_size,
                             shuffle=False)
        print('finished load test dataloader')
        l1,l2 = len(train_dl),len(test_dl)
        print('dataset len detail : %d batch in train, %d batch in test'%(l1,l2))
        for epoch in range(self.epochs):
            for i,batch in enumerate(train_dl):
                cur_iter_num = l1 * epoch + i + 1
                #update the model parameters
                loss_on_train , f1_on_train, acc_on_train = self._update_parameters(batch)
                loss_test, f1_test,acc_test = self.evaluate([batch])
                if cur_iter_num % self.check_interval == 0:
                    loss_on_test,f1_on_test,acc_on_test = self.evaluate(test_dl)
                    if f1_on_test > self.max_f1_on_test:
                        self.max_f1_on_test = f1_on_test
                        #use the f_score as the model name
                        model_name = ("%.4f"%f1_on_test)[2:]
                        self.save_checkpoint('%sf_score.model'%model_name)
                    print('on test: (batch: %d, epoch: %d, loss: %.4f, f1: %.4f, acc: %.4f)'
                            % (i + 1, epoch, loss_on_test, f1_on_test, acc_on_test))
                if cur_iter_num % self.print_every == 0:
                    print('on train : (batch: %d, epoch: %d, loss: %.4f, f1: %.4f, acc: %.4f)'
                          %(i+1,epoch,loss_on_train,f1_on_train,acc_on_train))
                    self.writer.add_scalar('loss_on_train',loss_on_train,cur_iter_num)
                    self.writer.add_scalar('f1_on_train',f1_on_train,cur_iter_num)
        print('training finished the blow are the detail on train set')

    def evaluate(self,batch_iter):
        self.model.eval()
        l = len(batch_iter)
        acc,f1,loss = 0,0,0
        prediction,label = None,None
        for batch in batch_iter:
            mention,entity,label = list(map(lambda x:x.to(device),batch))
            out_items  = None
            with torch.no_grad():
                out_items = self.model(mention,entity)
            out_items = list(map(lambda x:x.detach(),out_items))
            out,mention_miu,mention_logvar,joint_miu,joint_logvar = out_items
            tmp_loss = loss_func(out,label,mention_miu,mention_logvar,joint_miu,joint_logvar)
            loss += tmp_loss.item()
            prediction = out.argmax(dim=-1).squeeze()
            acc += torch.sum(prediction==label).item() / label.numel()
            prediction = prediction.cpu().numpy()
            label = label.cpu().numpy()
            f1 += f1_score(label,prediction)
        return loss / l, f1 / l, acc / l
    def save_checkpoint(self,model_dir):
        if not os.path.exists('../checkpoints'):
            os.mkdir('../checkpoints')
        this_train_save_to = os.path.join('../checkpoints',self.now_time)
        if not os.path.exists(this_train_save_to):
            os.mkdir(this_train_save_to)
        this_model_save_to = os.path.join(this_train_save_to,model_dir)
        torch.save(self.model,this_model_save_to)
    def _update_parameters(self,batch):
        self.model.to(device)
        self.model.train()
        self.optimizer.zero_grad()
        mention,entity,label = list(map(lambda x:x.to(device),batch))
        out,mention_miu,mention_logvar,joint_miu,joint_logvar = self.model(mention,entity)
        input = (out,label,mention_miu,mention_logvar,joint_miu,joint_logvar)
        loss = self.criterion(*input)
        loss.backward()
        self.optimizer.step()
        #cal acc just in gpu cause the speed is faster and method is simple
        prediction = out.detach().argmax(dim=-1).squeeze()
        acc = torch.sum(prediction==label).item() / torch.numel(label)
        prediction = prediction.cpu().numpy()
        label = label.cpu().numpy()
        f1 = f1_score(label,prediction)
        return loss.item(),f1,acc
def main():
    data_file = '../data/aida.data'
    print('start load dataset from %s path'%data_file)
    el_dict = Dictionary.load('../data/voc.dict')
    print('finished loaded dict')
    el_config = Config
    el_config['voc_size'] = len(el_dict)
    print('start prepare the raw data into tensor dataset')
    whole_dataset = ELDataset(el_dict,data_file)
    print('finished load dataset ')
    el_model = VAEEntityLinkingModel(el_config)
    el_model.init_weights()
    print('start training the model')
    trainner = Trainner(el_config,el_model,whole_dataset,loss_func)
    trainner.train()
if __name__ == "__main__":
    main()
