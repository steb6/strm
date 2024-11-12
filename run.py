import torch
import numpy as np
import argparse
import os
import pickle
from utils import print_and_log, get_log_files, TestAccuracies, loss, aggregate_accuracy, verify_checkpoint_dir, task_confusion, binary_classification_accuracy
from model import CNN_STRM
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quiet TensorFlow warnings
import tensorflow as tf

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import torchvision
import video_reader
import random 

import logging
from tqdm import tqdm
import wandb

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def setup_logger(name, log_file, level = logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger
    
# logger for training accuracies
train_logger = setup_logger('Training_accuracy', './runs_strm/train_output.log')

# logger for evaluation accuracies
eval_logger = setup_logger('Evaluation_accuracy', './runs_strm/eval_output.log')    

#############################################
#setting up seeds
manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
########################################################

def main():
    learner = Learner()
    learner.run()

class OpenSetLoss(torch.nn.Module):
    def __init__(self):
        super(OpenSetLoss, self).__init__()

    def forward(self, logits, targets):
        if len(logits.shape) > 2:
            logits = logits.squeeze(0)

        known_indices = targets != -1
        known_logits = torch.gather(logits[known_indices], 1, targets[known_indices].unsqueeze(1)).squeeze(1)  # 20
        known_loss = torch.exp(torch.tensor(1)) - torch.exp(known_logits)
        known_loss = known_loss.mean()  # TODO mean or sum?

        unknown_indices = targets == -1
        unknown_logits = logits[unknown_indices].reshape(-1)
        pos_unknown_logits = unknown_logits[unknown_logits > 0]
        pos_unknown_logits = pos_unknown_logits.mean()
        unknown_loss = -1 + torch.exp(pos_unknown_logits)
        
        open_set_loss = known_loss + unknown_loss
        return open_set_loss

class Learner:
    def __init__(self):
        self.args = self.parse_command_line()

        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final \
            = get_log_files(self.args.checkpoint_dir, self.args.resume_from_checkpoint, False)

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)

        # WANDB
        wandb.init(project="strm")
        wandb.login()
        
        gpu_device = 'cuda'
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        self.train_set, self.validation_set, self.test_set = self.init_data()

        self.vd = video_reader.VideoDataset(self.args)
        self.video_loader = torch.utils.data.DataLoader(self.vd, batch_size=1, num_workers=self.args.num_workers)
        
        self.loss = loss
        self.accuracy_fn = aggregate_accuracy
        self.open_set_accuracy_fn = binary_classification_accuracy
        
        if self.args.opt == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.opt == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        self.test_accuracies = TestAccuracies(self.test_set)
        
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.args.sch, gamma=0.1)
        
        self.start_iteration = 0
        if self.args.resume_from_checkpoint:
            self.load_checkpoint()
        self.optimizer.zero_grad()

        self.open_set_loss = OpenSetLoss()


    def init_model(self):
        model = CNN_STRM(self.args)
        model = model.to(self.device) 
        if self.args.num_gpus > 1:
            model.distribute_model()
        wandb.watch(model)
        return model

    def init_data(self):
        train_set = [self.args.dataset]
        validation_set = [self.args.dataset]
        test_set = [self.args.dataset]
        return train_set, validation_set, test_set


    """
    Command line parser
    """
    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--dataset", choices=["ssv2", "kinetics", "hmdb", "ucf", "nturgbd"], default="ssv2", help="Dataset to use.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate.")
        parser.add_argument("--tasks_per_batch", type=int, default=16, help="Number of tasks between parameter optimizations.")
        parser.add_argument("--checkpoint_dir", "-c", default=None, help="Directory to save checkpoint to.")
        parser.add_argument("--test_model_path", "-m", default=None, help="Path to model to load and test.")
        parser.add_argument("--training_iterations", "-i", type=int, default=100020, help="Number of meta-training iterations.")
        parser.add_argument("--resume_from_checkpoint", "-r", dest="resume_from_checkpoint", default=False, action="store_true", help="Restart from latest checkpoint.")
        parser.add_argument("--way", type=int, default=5, help="Way of each task.")
        parser.add_argument("--shot", type=int, default=5, help="Shots per class.")
        parser.add_argument("--query_per_class", type=int, default=5, help="Target samples (i.e. queries) per class used for training.")
        parser.add_argument("--query_per_class_test", type=int, default=1, help="Target samples (i.e. queries) per class used for testing.")
        parser.add_argument('--test_iters', nargs='+', type=int, help='iterations to test at. Default is for ssv2 otam split.', default=[75000])
        parser.add_argument("--num_test_tasks", type=int, default=10000, help="number of random tasks to test on.")
        parser.add_argument("--print_freq", type=int, default=1000, help="print and log every n iterations.")
        parser.add_argument("--seq_len", type=int, default=8, help="Frames per video.")
        parser.add_argument("--num_workers", type=int, default=10, help="Num dataloader workers.")
        parser.add_argument("--method", choices=["resnet18", "resnet34", "resnet50"], default="resnet50", help="method")
        parser.add_argument("--trans_linear_out_dim", type=int, default=1152, help="Transformer linear_out_dim")
        parser.add_argument("--opt", choices=["adam", "sgd"], default="sgd", help="Optimizer")
        parser.add_argument("--trans_dropout", type=int, default=0.1, help="Transformer dropout")
        parser.add_argument("--save_freq", type=int, default=5000, help="Number of iterations between checkpoint saves.")
        parser.add_argument("--img_size", type=int, default=224, help="Input image size to the CNN after cropping.")
        parser.add_argument('--temp_set', nargs='+', type=int, help='cardinalities e.g. 2,3 is pairs and triples', default=[2,3])
        parser.add_argument("--scratch", choices=["bc", "bp", "new", "/home/sberti_datasets/"], default="bp", help="directory containing dataset, splits, and checkpoint saves.")
        parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to split the ResNet over")
        parser.add_argument("--debug_loader", default=False, action="store_true", help="Load 1 vid per class for debugging")
        parser.add_argument("--split", type=int, default=7, help="Dataset split.")
        parser.add_argument('--sch', nargs='+', type=int, help='iters to drop learning rate', default=[1000000])
        parser.add_argument("--test_model_only", type=bool, default=False, help="Only testing the model from the given checkpoint")
        parser.add_argument("--use_fine_grain_tasks", type=float, default=0., help="The percentage of fine-grain tasks to use")
        parser.add_argument("--open_set", action="store_true", help="If True, open-set task is added")   

        args = parser.parse_args()
        
        if args.scratch == "bc":
            args.scratch = "/mnt/storage/home2/tp8961/scratch"
        elif args.scratch == "bp":
            args.num_gpus = 4
            # this is low becuase of RAM constraints for the data loader
            args.num_workers = 3
            args.scratch = "/work/tp8961"
        elif args.scratch == "new":
            args.scratch = "."
        
        if args.checkpoint_dir == None:
            print("need to specify a checkpoint dir")
            exit(1)

        if (args.method == "resnet50") or (args.method == "resnet34"):
            args.img_size = 224
        if args.method == "resnet50":
            args.trans_linear_in_dim = 2048
        else:
            args.trans_linear_in_dim = 512
        
        if args.dataset == "ssv2":
            args.traintestlist = os.path.join(args.scratch, "splits/ssv2_OTAM")
            args.path = "/home/iit.local/sberti/datasets/SSv2/all"
        elif args.dataset == "kinetics":
            args.traintestlist = os.path.join(args.scratch, "video_datasets/splits/kineticsTrainTestlist")
            args.path = os.path.join(args.scratch, "video_datasets/data/kinetics_256q5_1.zip")
        elif args.dataset == "ucf":
            args.traintestlist = os.path.join(args.scratch, "video_datasets/splits/ucfTrainTestlist")
            args.path = os.path.join(args.scratch, "video_datasets/data/UCF-101_320.zip")
        elif args.dataset == "hmdb":
            args.traintestlist = os.path.join(args.scratch, "video_datasets/splits/hmdb51TrainTestlist")
            args.path = os.path.join(args.scratch, "video_datasets/data/hmdb51_256q5.zip")
            # args.path = os.path.join(args.scratch, "video_datasets/data/hmdb51_jpegs_256.zip")
        elif args.dataset == "nturgbd":
            args.traintestlist = os.path.join("splits", "nturgbd")
            args.path = os.path.join(args.scratch, "NTU_dataset")

        with open("args.pkl", "wb") as f:
            pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)

        return args

    def run(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as session:
                train_accuracies = []
                losses = []
                total_iterations = self.args.training_iterations

                iteration = self.start_iteration

                if self.args.test_model_only:
                    print("Model being tested at path: " + self.args.test_model_path)
                    self.load_checkpoint()
                    accuracy_dict = self.test(session, 1)
                    print(accuracy_dict)
                    exit()  # NOTE added by me


                for task_dict in self.video_loader:
                    if iteration >= total_iterations:
                        break
                    iteration += 1
                    torch.set_grad_enabled(True)

                    task_loss, task_accuracy = self.train_task(task_dict)
                    train_accuracies.append(task_accuracy)
                    losses.append(task_loss)

                    # optimize
                    if ((iteration + 1) % self.args.tasks_per_batch == 0) or (iteration == (total_iterations - 1)):
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    self.scheduler.step()
                    if (iteration + 1) % self.args.print_freq == 0:
                        # print training stats
                        all_acc = np.mean([x["all"] for x in train_accuracies])
                        open_acc = np.mean([x["open"] for x in train_accuracies])
                        closed_acc = np.mean([x["closed"] for x in train_accuracies])
                        all_loss = np.mean([x["all"] for x in losses])
                        open_loss = np.mean([x["open"] for x in losses])
                        closed_loss = np.mean([x["closed"] for x in losses])
                        print_and_log(self.logfile,"Task [{}/{}], Train All Loss: {:.7f}, Train Open Loss: {:.7f}, Train Closed Loss: {:.7f}, Train All Accuracy: {:.7f}, Train Open Accuracy: {:.7f}, Train Closed Accuracy: {:.7f}".format(iteration + 1, total_iterations, all_loss, open_loss, closed_loss, all_acc, open_acc, closed_acc))
                        # train_logger.info("For Task: {0}, the training loss is {1} and Training Accuracy is {2} and Open Set Loss is {3} and Open Set Accuracy is {4}".format(iteration + 1, torch.Tensor(losses).mean().item(),
                        #     torch.Tensor(train_accuracies).mean().item(), torch.Tensor(open_set_losses).mean().item(), torch.Tensor(open_set_accuracies).mean().item()))

                        wandb.log({"Train All Accuracy": all_acc,
                                   "Train Open Accuracy": open_acc,
                                   "Train Closed Accuracy": closed_acc,
                                   "Train All Loss": all_loss,
                                   "Train Open Loss": open_loss,
                                   "Train Closed Loss": closed_loss})     
                        
                        train_accuracies = []
                        losses = []

                    if ((iteration + 1) % self.args.save_freq == 0) and (iteration + 1) != total_iterations:
                        self.save_checkpoint(iteration + 1)


                    if ((iteration + 1) in self.args.test_iters) and (iteration + 1) != total_iterations:
                        accuracy_dict = self.test(session, iteration + 1)
                        print(accuracy_dict)
                        self.test_accuracies.print(self.logfile, accuracy_dict)
                        wandb.log(accuracy_dict)

                # save the final model
                torch.save(self.model.state_dict(), self.checkpoint_path_final)

        self.logfile.close()

    def train_task(self, task_dict):
        context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list, unknown_images, unknown_labels = self.prepare_task(task_dict)

        context_images = context_images.to(self.device)
        context_labels = context_labels.to(self.device)
        target_images = target_images.to(self.device)
        unknown_images = unknown_images.to(self.device)

        all_images = torch.cat((target_images, unknown_images), 0)
        model_dict, precomputed_context_features = self.model(context_images, context_labels, all_images)
        target_logits = model_dict['logits'].to(self.device)

        # Open set logits
        open_set_targets = torch.cat([target_labels, torch.full_like(unknown_labels, fill_value=-1)])
        open_set_loss_value = self.open_set_loss(target_logits, open_set_targets)
        target_logits_post_pat = model_dict['logits_post_pat'].to(self.device)
        open_set_loss_value_post_pat = self.open_set_loss(target_logits_post_pat, open_set_targets)

        # Get known and unknown logits
        known_target_logits, unknown_target_logits = target_logits[:, :target_labels.size(0)], target_logits[:, target_labels.size(0):]
        known_target_logits_post_pat, unknown_target_logits_post_pat = target_logits_post_pat[:, :target_labels.size(0)], target_logits_post_pat[:, target_labels.size(0):]

        # Target logits after applying query-distance-based similarity metric on patch-level enriched features
        target_labels = target_labels.to(self.device)
        task_loss = self.loss(known_target_logits, target_labels, self.device) / self.args.tasks_per_batch
        task_loss_post_pat = self.loss(known_target_logits_post_pat, target_labels, self.device) / self.args.tasks_per_batch

        # Joint loss
        all_task_loss = task_loss + 0.1*task_loss_post_pat + open_set_loss_value + 0.1*open_set_loss_value_post_pat  # added open_set_loss

        # Add the logits before computing the accuracy
        target_logits = target_logits + 0.1*target_logits_post_pat
        # Open set accuracy (FSOS acc) (50% closed set known, 50% open set unknown)
        target_labels = torch.cat([target_labels, torch.full_like(target_labels.to(self.device), fill_value=-1)])
        task_accuracy = self.accuracy_fn(target_logits, target_labels)

        all_task_loss.backward(retain_graph=False)

        losses = {"all": all_task_loss.item(),
                  "open": (open_set_loss_value+0.1*open_set_loss_value_post_pat).item(), 
                  "closed": (task_loss + 0.1*task_loss_post_pat).item()}
        return losses, task_accuracy

    def test(self, session, num_episode):
        self.model.eval()
        with torch.no_grad():
            self.video_loader.dataset.train = False
            accuracy_dict = {}
            accuracies = []
            losses = []
            iteration = 0
            item = self.args.dataset
            progress_bar = tqdm(total=self.args.num_test_tasks, desc='Processing')
            
            for task_dict in self.video_loader:
                if iteration >= self.args.num_test_tasks:
                    break
                iteration += 1
                progress_bar.update()

                context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list, unknown_images, unknown_labels = self.prepare_task(task_dict)
                context_images = context_images.to(self.device)
                context_labels = context_labels.to(self.device)
                target_images = target_images.to(self.device)
                unknown_images = unknown_images.to(self.device)

                all_images = torch.cat((target_images, unknown_images), 0)
                model_dict, precomputed_context_features = self.model(context_images, context_labels, all_images)
                target_logits = model_dict['logits'].to(self.device)

                # Open set logits
                open_set_targets = torch.cat([target_labels, torch.full_like(unknown_labels, fill_value=-1)])
                open_set_loss_value = self.open_set_loss(target_logits, open_set_targets)
                target_logits_post_pat = model_dict['logits_post_pat'].to(self.device)
                open_set_loss_value_post_pat = self.open_set_loss(target_logits_post_pat, open_set_targets)

                # Get known and unknown logits
                known_target_logits, unknown_target_logits = target_logits[:, :target_labels.size(0)], target_logits[:, target_labels.size(0):]
                known_target_logits_post_pat, unknown_target_logits_post_pat = target_logits_post_pat[:, :target_labels.size(0)], target_logits_post_pat[:, target_labels.size(0):]

                # Target logits after applying query-distance-based similarity metric on patch-level enriched features
                target_labels = target_labels.to(self.device)
                task_loss = self.loss(known_target_logits, target_labels, self.device) / self.args.num_test_tasks
                task_loss_post_pat = self.loss(known_target_logits_post_pat, target_labels, self.device) / self.args.num_test_tasks

                # Joint loss
                all_task_loss = task_loss + 0.1 * task_loss_post_pat + open_set_loss_value + 0.1 * open_set_loss_value_post_pat

                # Add the logits before computing the accuracy
                target_logits = target_logits + 0.1 * target_logits_post_pat
                # Open set accuracy (FSOS acc) (50% closed set known, 50% open set unknown)
                target_labels = torch.cat([target_labels, torch.full_like(target_labels.to(self.device), fill_value=-1)])
                task_accuracy = self.accuracy_fn(target_logits, target_labels)

                losses.append(all_task_loss.item())
                accuracies.append(task_accuracy["all"])

                eval_logger.info("For Task: {0}, the testing loss is {1} and Testing Accuracy is {2}".format(iteration, all_task_loss.item(), task_accuracy["all"]))

                del target_logits
                del target_logits_post_pat

            accuracy = np.array(accuracies).mean() * 100.0
            confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))
            loss = np.array(losses).mean()
            accuracy_dict[item] = {"accuracy": accuracy, "confidence": confidence, "loss": loss}
            eval_logger.info("For Task: {0}, the testing loss is {1} and Testing Accuracy is {2}".format(num_episode, loss, accuracy))

            self.video_loader.dataset.train = True
        self.model.train()

        return accuracy_dict


    def prepare_task(self, task_dict, images_to_device = True):
        context_images, context_labels = task_dict['support_set'][0], task_dict['support_labels'][0]
        target_images, target_labels = task_dict['target_set'][0], task_dict['target_labels'][0]
        real_target_labels = task_dict['real_target_labels'][0]
        batch_class_list = task_dict['batch_class_list'][0]

        unknown_images = task_dict['unknown_set'][0]
        unknown_labels = task_dict['unknown_labels'][0]

        if images_to_device:
            context_images = context_images.to(self.device)
            target_images = target_images.to(self.device)
            unknown_images = unknown_images.to(self.device)
        context_labels = context_labels.to(self.device)
        target_labels = target_labels.type(torch.LongTensor).to(self.device)
        unknown_labels = unknown_labels.type(torch.LongTensor).to(self.device)

        return context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list, unknown_images, unknown_labels

    def shuffle(self, images, labels):
        """
        Return shuffled data.
        """
        permutation = np.random.permutation(images.shape[0])
        return images[permutation], labels[permutation]


    def save_checkpoint(self, iteration):
        d = {'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}

        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint{}.pt'.format(iteration)))
        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint.pt'))

    def load_checkpoint(self):
        if self.args.test_model_only:
            checkpoint = torch.load(self.args.test_model_path)
        else:
           checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'checkpoint.pt'))
        self.start_iteration = checkpoint['iteration']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])


if __name__ == "__main__":
    main()
