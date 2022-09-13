# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.data.load_data_ground import DataSet_VQS
from core.model.net_attmaskturnsof_ground import Net_attmaskturnsof_ground
from core.model.optim import get_optim, adjust_lr
from core.data.data_utils import shuffle_list
from utils.vqa import VQA
from utils.vqaEval import VQAEval

import os, json, torch, datetime, pickle, copy, shutil, time
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import collections

class Execution:
    def __init__(self, __C):
        self.__C = __C

        print('Loading training set ........')
        self.dataset = DataSet_VQS(__C)

        # self.dataset_eval = None
        # if __C.EVAL_EVERY_EPOCH:
        # __C_eval = copy.deepcopy(__C)
        # setattr(__C_eval, 'RUN_MODE', 'val')
        #
        # print('Loading validation set for per-epoch evaluation ........')
        # self.dataset_eval = DataSet_VQS(__C_eval)


    def train(self, dataset, dataset_eval=None):

        # Obtain needed information
        data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb

        # Define the MCAN model
        net = Net_attmaskturnsof_ground(
            self.__C,
            pretrained_emb,
            token_size,
            ans_size
        )
        net.cuda()
        net.train()

        # Define the multi-gpu training if needed
        if self.__C.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.__C.DEVICES)

        # Define the binary cross entropy loss
        # loss_fn = torch.nn.BCELoss(size_average=False).cuda()
        loss_fn = torch.nn.BCELoss(reduction='sum').cuda()
        loss_kl = nn.KLDivLoss(reduction='batchmean')

        # Load checkpoint if resume training
        if self.__C.RESUME:
            print(' ========== Resume training')

            if self.__C.CKPT_PATH is not None:
                print('Warning: you are now using CKPT_PATH args, '
                      'CKPT_VERSION and CKPT_EPOCH will not work')

                path = self.__C.CKPT_PATH
            else:
                path = self.__C.CKPTS_PATH + \
                       'ckpt_' + self.__C.CKPT_VERSION + \
                       '/epoch' + str(self.__C.CKPT_EPOCH) + '.pkl'

            # Load the network parameters
            print('Loading ckpt {}'.format(path))
            ckpt = torch.load(path)
            print('Finish!')
            net.load_state_dict(ckpt['state_dict'])

            # Load the optimizer paramters
            optim = get_optim(self.__C, net, data_size, ckpt['lr_base'])
            optim._step = int(data_size / self.__C.BATCH_SIZE * self.__C.CKPT_EPOCH)
            optim.optimizer.load_state_dict(ckpt['optimizer'])


            start_epoch = self.__C.CKPT_EPOCH

        else:
            if ('ckpt_' + self.__C.VERSION) in os.listdir(self.__C.CKPTS_PATH):
                shutil.rmtree(self.__C.CKPTS_PATH + 'ckpt_' + self.__C.VERSION)

            os.mkdir(self.__C.CKPTS_PATH + 'ckpt_' + self.__C.VERSION)

            optim = get_optim(self.__C, net, data_size)
            start_epoch = 0

        loss_sum = 0
        named_params = list(net.named_parameters())
        grad_norm = np.zeros(len(named_params))

        if self.__C.SHUFFLE_MODE in ['external']:
            dataloader = Data.DataLoader(
                dataset,
                batch_size=self.__C.BATCH_SIZE,
                shuffle=False,
                num_workers=self.__C.NUM_WORKERS,
                pin_memory=self.__C.PIN_MEM,
                drop_last=True
            )
        else:
            dataloader = Data.DataLoader(
                dataset,
                batch_size=self.__C.BATCH_SIZE,
                shuffle=True,
                num_workers=self.__C.NUM_WORKERS,
                pin_memory=self.__C.PIN_MEM,
                drop_last=True
            )

        count = 0
        for name, para in net.named_parameters():
            item = 1
            for i in para.size():
                item = item * i
            count += item

        # Training script
        for epoch in range(start_epoch, self.__C.MAX_EPOCH):

            # Save log information
            logfile = open(
                self.__C.LOG_PATH +
                'log_run_' + self.__C.VERSION + '.txt',
                'a+'
            )

            logfile.write(
                'nowTime: ' +
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
                '\n'
            )
            logfile.close()

            if epoch in self.__C.LR_DECAY_LIST:
                adjust_lr(optim, self.__C.LR_DECAY_R)

            # Externally shuffle
            if self.__C.SHUFFLE_MODE == 'external':
                shuffle_list(dataset.ans_list)

            time_start = time.time()
            # Iteration
            for step, (
                    img_feat_iter,
                    ques_ix_iter,
                    ans_iter
            ) in enumerate(dataloader):
                optim.zero_grad()

                img_feat_iter = img_feat_iter.cuda()    # torch.Size([64, 100, 2048])
                ques_ix_iter = ques_ix_iter.cuda()  # torch.Size([64, 14])
                ans_iter = ans_iter.cuda()  # torch.Size([64, 3129])

                for accu_step in range(self.__C.GRAD_ACCU_STEPS):

                    sub_img_feat_iter = \
                        img_feat_iter[accu_step * self.__C.SUB_BATCH_SIZE:
                                      (accu_step + 1) * self.__C.SUB_BATCH_SIZE]
                    sub_ques_ix_iter = \
                        ques_ix_iter[accu_step * self.__C.SUB_BATCH_SIZE:
                                     (accu_step + 1) * self.__C.SUB_BATCH_SIZE]
                    sub_ans_iter = \
                        ans_iter[accu_step * self.__C.SUB_BATCH_SIZE:
                                 (accu_step + 1) * self.__C.SUB_BATCH_SIZE]


                    # pred = net(
                    #     sub_img_feat_iter,
                    #     sub_ques_ix_iter
                    # )
                    pred_0, pred_1 = net(
                        sub_img_feat_iter,
                        sub_ques_ix_iter
                    )

                    # loss = loss_fn(pred, sub_ans_iter)

                    loss_bce_0 = loss_fn(torch.sigmoid(pred_0), sub_ans_iter)
                    loss_bce_1 = loss_fn(torch.sigmoid(pred_1), sub_ans_iter)
                    kl_loss_01 = loss_kl(F.log_softmax(pred_0, dim=1), F.softmax(Variable(pred_1), dim=1))
                    kl_loss_10 = loss_kl(F.log_softmax(pred_1, dim=1), F.softmax(Variable(pred_0), dim=1))
                    loss = loss_bce_0 + loss_bce_1 + kl_loss_01 + kl_loss_10

                    loss.backward()
                    loss_sum += loss.cpu().data.numpy() * self.__C.GRAD_ACCU_STEPS

                    if self.__C.VERBOSE:
                        if dataset_eval is not None:
                            mode_str = self.__C.SPLIT['train'] + '->' + self.__C.SPLIT['val']
                        else:
                            mode_str = self.__C.SPLIT['train'] + '->' + self.__C.SPLIT['test']

                        print("\r[version %s][epoch %2d][step %4d/%4d][%s] loss: %.4f, lr: %.2e" % (
                            self.__C.VERSION,
                            epoch + 1,
                            step,
                            int(data_size / self.__C.BATCH_SIZE),
                            mode_str,
                            loss.cpu().data.numpy() / self.__C.SUB_BATCH_SIZE,
                            optim._rate
                        ), end='          ')

                # Gradient norm clipping
                if self.__C.GRAD_NORM_CLIP > 0:
                    nn.utils.clip_grad_norm_(
                        net.parameters(),
                        self.__C.GRAD_NORM_CLIP
                    )

                # Save the gradient information
                for name in range(len(named_params)):
                    norm_v = torch.norm(named_params[name][1].grad).cpu().data.numpy() \
                        if named_params[name][1].grad is not None else 0
                    grad_norm[name] += norm_v * self.__C.GRAD_ACCU_STEPS
                    # print('Param %-3s Name %-80s Grad_Norm %-20s'%
                    #       (str(grad_wt),
                    #        params[grad_wt][0],
                    #        str(norm_v)))
                optim.step()

            time_end = time.time()
            print('Finished in {}s'.format(int(time_end-time_start)))

            # print('')
            epoch_finish = epoch + 1

            # Save checkpoint
            state = {
                'state_dict': net.state_dict(),
                'optimizer': optim.optimizer.state_dict(),
                'lr_base': optim.lr_base
            }
            torch.save(
                state,
                self.__C.CKPTS_PATH +
                'ckpt_' + self.__C.VERSION +
                '/epoch' + str(epoch_finish) +
                '.pkl'
            )

            # Logging
            logfile = open(
                self.__C.LOG_PATH +
                'log_run_' + self.__C.VERSION + '.txt',
                'a+'
            )
            logfile.write(
                'epoch = ' + str(epoch_finish) +
                '  loss = ' + str(loss_sum / data_size) +
                '\n' +
                'lr = ' + str(optim._rate) +
                '\n\n'
            )
            logfile.close()

            # Eval after every epoch
            if dataset_eval is not None:
                self.eval(
                    dataset_eval,
                    state_dict=net.state_dict(),
                    valid=True
                )

            # if self.__C.VERBOSE:
            #     logfile = open(
            #         self.__C.LOG_PATH +
            #         'log_run_' + self.__C.VERSION + '.txt',
            #         'a+'
            #     )
            #     for name in range(len(named_params)):
            #         logfile.write(
            #             'Param %-3s Name %-80s Grad_Norm %-25s\n' % (
            #                 str(name),
            #                 named_params[name][0],
            #                 str(grad_norm[name] / data_size * self.__C.BATCH_SIZE)
            #             )
            #         )
            #     logfile.write('\n')
            #     logfile.close()

            loss_sum = 0
            grad_norm = np.zeros(len(named_params))


    # Evaluation
    def eval(self, dataset, state_dict=None, valid=False):

        # Load parameters
        path = self.__C.CKPTS_PATH + \
       'ckpt_' + self.__C.CKPT_VERSION + \
       '/epoch' + str(self.__C.CKPT_EPOCH) + '.pkl'

        val_ckpt_flag = False
        if state_dict is None:
            val_ckpt_flag = True
            print('Loading ckpt {}'.format(path))
            state_dict = torch.load(path)['state_dict']
            print('Finish!')

        # Store the prediction list
        qid_list = [ques['question_id'] for ques in dataset.ques_list]
        ans_ix_list = []
        pred_list = []

        data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb

        net = Net_attmaskturnsof_ground(
            self.__C,
            pretrained_emb,
            token_size,
            ans_size
        )
        print("***************************Net_attmaskturnsof_ground eval***************************")

        net.cuda()
        net.eval()

        if self.__C.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.__C.DEVICES)

        net.load_state_dict(state_dict)

        dataloader = Data.DataLoader(
            dataset,
            batch_size=self.__C.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.__C.NUM_WORKERS,
            pin_memory=True
        )
        intersection_all = 0
        union_all = 0
        for step, (
                img_feat_iter,
                ques_ix_iter,
                ans_iter,
                ground_mask,  # b,640,640 torch.uint8
                img_bbox  # b,100,4
        ) in enumerate(dataloader):
            print("\rEvaluation: [step %4d/%4d]" % (
                step,
                int(data_size / self.__C.EVAL_BATCH_SIZE),
            ), end='          ')

            img_feat_iter = img_feat_iter.cuda()
            ques_ix_iter = ques_ix_iter.cuda()

            # pred, att = net(
            #     img_feat_iter,
            #     ques_ix_iter
            # )
            pred, pred2, att = net(
                img_feat_iter,
                ques_ix_iter
            )
            # att b,100
            # att_np = att.cpu().data.numpy()
            # att_argmax = np.argmax(att_np, axis=1)  # b
            att_np = att.cpu().data
            _,att_argmax = att_np.topk(3,dim=1)  # b,2
            pre_mask_list = []
            for i in range(pred.size(0)):
                pre_mask = torch.zeros((640,640),dtype=torch.uint8)
                x1,y1,x2,y2 = img_bbox[i][att_argmax[i][0]]
                x1, y1, x2, y2 = int(x1.item()),int(y1.item()),int(x2.item()),int(y2.item())
                pre_mask[y1:y2,x1:x2] = 1

                x1_t, y1_t, x2_t, y2_t = img_bbox[i][att_argmax[i][1]]
                x1_t, y1_t, x2_t, y2_t = int(x1_t.item()), int(y1_t.item()), int(x2_t.item()), int(y2_t.item())
                pre_mask_2 = pre_mask.clone()
                pre_mask_2[y1_t:y2_t, x1_t:x2_t] = 1
                # pre_mask[y1_t:y2_t, x1_t:x2_t] = 1

                x1_r, y1_r, x2_r, y2_r = img_bbox[i][att_argmax[i][2]]
                x1_r, y1_r, x2_r, y2_r = int(x1_r.item()), int(y1_r.item()), int(x2_r.item()), int(y2_r.item())
                pre_mask_3 = pre_mask_2.clone()
                pre_mask_3[y1_r:y2_r, x1_r:x2_r] = 1
                # pre_mask[y1_r:y2_r, x1_r:x2_r] = 1

                in1, un1 = compute_mask_IU(pre_mask,ground_mask[i])
                in2, un2 = compute_mask_IU(pre_mask_2,ground_mask[i])
                in3, un3 = compute_mask_IU(pre_mask_3,ground_mask[i])
                index = np.argmax([float(in1)/float(un1), float(in2)/float(un2), float(in3)/float(un3)])
                if index == 0:
                    pre_mask_list.append(pre_mask)
                elif index == 1:
                    pre_mask_list.append(pre_mask_2)
                elif index == 2:
                    pre_mask_list.append(pre_mask_3)
                # pre_mask_list.append(pre_mask)
            pre_mask = torch.stack(pre_mask_list,dim=0)  # b,640,640
            intersection, union = compute_mask_IU(pre_mask,ground_mask)
            intersection_all += intersection
            union_all += union
        print('')
        ans_ix_list = np.array(ans_ix_list).reshape(-1)
        print("the intersection_all is: ", intersection_all, "the union_all is: ", union_all)
        print("the mIOU is: ", float(intersection_all) / float(union_all))
        exit()
        result = [{
            'answer': dataset.ix_to_ans[str(ans_ix_list[qix])],  # ix_to_ans(load with json) keys are type of string
            'question_id': int(qid_list[qix])
        }for qix in range(qid_list.__len__())]

        # Write the results to result file
        if valid:
            if val_ckpt_flag:
                result_eval_file = \
                    self.__C.CACHE_PATH + \
                    'result_run_' + self.__C.CKPT_VERSION + \
                    '.json'
            else:
                result_eval_file = \
                    self.__C.CACHE_PATH + \
                    'result_run_' + self.__C.VERSION + \
                    '.json'

        else:
            if self.__C.CKPT_PATH is not None:
                result_eval_file = \
                    self.__C.RESULT_PATH + \
                    'result_run_' + self.__C.CKPT_VERSION + \
                    '.json'
            else:
                result_eval_file = \
                    self.__C.RESULT_PATH + \
                    'result_run_' + self.__C.CKPT_VERSION + \
                    '_epoch' + str(self.__C.CKPT_EPOCH) + \
                    '.json'

            print('Save the result to file: {}'.format(result_eval_file))

        json.dump(result, open(result_eval_file, 'w'))

        # Save the whole prediction vector
        if self.__C.TEST_SAVE_PRED:

            if self.__C.CKPT_PATH is not None:
                ensemble_file = \
                    self.__C.PRED_PATH + \
                    'result_run_' + self.__C.CKPT_VERSION + \
                    '.json'
            else:
                ensemble_file = \
                    self.__C.PRED_PATH + \
                    'result_run_' + self.__C.CKPT_VERSION + \
                    '_epoch' + str(self.__C.CKPT_EPOCH) + \
                    '.json'

            print('Save the prediction vector to file: {}'.format(ensemble_file))

            pred_list = np.array(pred_list).reshape(-1, ans_size)
            result_pred = [{
                'pred': pred_list[qix],
                'question_id': int(qid_list[qix])
            }for qix in range(qid_list.__len__())]

            pickle.dump(result_pred, open(ensemble_file, 'wb+'), protocol=-1)


    def run(self, run_mode):
        if run_mode == 'train':
            self.empty_log(self.__C.VERSION)
            self.train(self.dataset, self.dataset_eval)

        elif run_mode == 'val':
            self.eval(self.dataset, valid=True)

        elif run_mode == 'test':
            self.eval(self.dataset)

        else:
            exit(-1)

    def empty_log(self, version):
        print('Initializing log file ........')
        if (os.path.exists(self.__C.LOG_PATH + 'log_run_' + version + '.txt')):
            os.remove(self.__C.LOG_PATH + 'log_run_' + version + '.txt')
        print('Finished!')
        print('')


def compute_mask_IU(masks, target):
    temp = (masks * target)
    intersection = temp.sum()
    union = ((masks + target) - temp).sum()
    return intersection.item(), union.item()

