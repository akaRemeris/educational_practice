import numpy as np
import datetime
import copy
import torch
import warnings

from torch.utils.data import DataLoader


class Trainer(object):
    def __init__(self, device=None, save_model_flag=True, verbose=False):

                self.verbose = verbose
                self.save_model_flag = save_model_flag
                self.iteration_log = []
                self.model = None
                self.optimizer = None
                self.lr_scheduler = None
                self.loss_function = None

                if device is None:
                    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                else:
                    self.device = torch.device(device)
                
    def prerun_check(self):
        assert self.model is not None, 'Model is not initialized'
        assert self.optimizer is not None, 'Optimizer is not initialized.'
        assert self.loss_function is not None, 'Loss function is not initialized.'
        if self.lr_scheduler is None:
            warnings.warn('Lr scheduler is not initialized.')        

    def init_model(self, model):
        self.model = model
        self.model = self.model.to(self.device)                

    def init_loss_function(self, loss_function):
        self.loss_function = loss_function
        if self.verbose:
            print('Loss function initialized.')

    def init_optimizer(self, optim_class, lr, **kwargs):
        self.optimizer = optim_class(self.model.parameters(), lr, **kwargs)
        if self.verbose:
            print('Optimizer initialized.')
    
    def init_lr_scheduler(self, lr_scheduler_class, **kwargs):
        assert self.optimizer is not None, 'Optimizer is not initialized.'
        self.lr_scheduler = lr_scheduler_class(self.optimizer, **kwargs)
        if self.verbose:
            print('Lr scheduler initialized.')
    
    def document_run(self, comment='No comments'):
        """
        Append to iteration log basic info on current Trainer object state.
        """
        run_log = dict()
        run_log['optimizer'] = self.optimizer
        run_log['scheduler'] = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else 'No scheduler.'
        run_log['comment'] = comment
        self.iteration_log.append(run_log)

    def batch_train(self, X, y):
        preds = self.model.forward(X)
        loss = self.loss_function(preds, y)
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.clone().detach().cpu().numpy()
    
    def epoch_train(self, dataloader, max_train_epoch_size):
        
        train_start_time = datetime.datetime.now()
        train_loss = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            train_loss += self.batch_train(X_batch, y_batch)  

            if batch_idx > max_train_epoch_size:
                break            
        
        training_time = (datetime.datetime.now() - train_start_time).total_seconds()
        mean_train_loss = train_loss/len(dataloader)

        if self.verbose:
            print('Epoch training done.')
            print('Training time:', training_time)
            print('Number of training iterations:', batch_idx+1)
            print('Mean training loss:', mean_train_loss)

        return mean_train_loss
    
    def epoch_eval(self, dataloader, max_eval_epoch_size):
        with torch.no_grad():

            eval_loss = 0
            for batch_idx, (X_batch, y_batch) in enumerate(dataloader):                           
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                preds = self.model.forward(X_batch)
                eval_loss += self.loss_function(preds, y_batch).clone().detach().cpu().numpy()
                
                if batch_idx > max_eval_epoch_size:
                    break   

            mean_eval_loss = eval_loss/len(dataloader)

        if self.verbose:
            print('Epoch evaluation done.')
            print('Number of evaluation iterations:', batch_idx+1)
            print('Mean evaluation loss:', mean_eval_loss)

        return mean_eval_loss

    def train_eval_procedure(self, train_data, eval_data, batch_size, max_epochs, 
                            shuffle=True, early_stopping_patience=10, max_train_epoch_size=1e4, max_eval_epoch_size=1e3, run_comment='No comments.',
                            do_train=True, do_eval=True):
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle) if do_train else None
        eval_dataloader = DataLoader(eval_data, batch_size=batch_size, shuffle=shuffle) if do_eval else None
        best_eval_loss = np.inf
        best_epoch = 0
        self.prerun_check()
        self.document_run(run_comment)
        train_loss_log = []
        eval_loss_log = []

        for epoch in range(max_epochs):
            try:
                if self.verbose:
                    print('Epoch number:', epoch)

                if do_train:
                    mean_train_loss = self.epoch_train(train_dataloader, max_train_epoch_size)
                    train_loss_log.append(mean_train_loss)
                
                if do_eval:
                    mean_eval_loss = self.epoch_eval(eval_dataloader, max_eval_epoch_size)
                    eval_loss_log.append(mean_eval_loss)

                if do_eval and (mean_eval_loss < best_eval_loss):
                    best_eval_loss = mean_eval_loss
                    best_epoch = epoch
                    if self.save_model_flag:
                        self.best_model = copy.deepcopy(self.model) 
                    if self.verbose:
                        print('New best model!')
                elif do_eval and (epoch - best_epoch > early_stopping_patience):
                    print('Early stopping. Eval_loss has not improved since epoch number:', best_epoch)
                    break

                if do_eval and self.lr_scheduler is not None:
                    self.lr_scheduler.step(mean_eval_loss)
                
                if self.verbose:
                    print()                

            except KeyboardInterrupt:
                print("Interrupted by request.")
                break
            except Exception as e:
                print("Failed train/eval")
                print(e)
                break
        self.iteration_log[-1]['model_description'] = str(self.model)
        self.iteration_log[-1]['train_loss_log'] = train_loss_log
        self.iteration_log[-1]['eval_loss_log'] = eval_loss_log