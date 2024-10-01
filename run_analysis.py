import torch
import numpy as np
import argparse
import os
import pickle
from utils import print_and_log, get_log_files, TestAccuracies, loss, aggregate_accuracy, verify_checkpoint_dir, task_confusion
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
import cv2

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


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()

        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final \
            = get_log_files(self.args.checkpoint_dir, self.args.resume_from_checkpoint, False)

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)

        
        gpu_device = 'cuda'
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        self.train_set, self.validation_set, self.test_set = self.init_data()

        self.vd = video_reader.VideoDataset(self.args)
        self.video_loader = torch.utils.data.DataLoader(self.vd, batch_size=1, num_workers=self.args.num_workers)
        
        self.loss = loss
        self.accuracy_fn = aggregate_accuracy
        
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

    def init_model(self):
        model = CNN_STRM(self.args)
        model = model.to(self.device) 
        if self.args.num_gpus > 1:
            model.distribute_model()
        model.register_activation_hook()
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
        parser.add_argument("--scratch", choices=["bc", "bp", "new", "/home/sberti_datasets/", "/home/steb6/datasets/"], default="bp", help="directory containing dataset, splits, and checkpoint saves.")
        parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to split the ResNet over")
        parser.add_argument("--debug_loader", default=False, action="store_true", help="Load 1 vid per class for debugging")
        parser.add_argument("--split", type=int, default=7, help="Dataset split.")
        parser.add_argument('--sch', nargs='+', type=int, help='iters to drop learning rate', default=[1000000])
        parser.add_argument("--test_model_only", type=bool, default=False, help="Only testing the model from the given checkpoint")
        parser.add_argument("--use_fine_grain_tasks", type=float, default=0., help="If True, dataset loads fine-grained tasks")

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
                        print_and_log(self.logfile,'Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}'
                                      .format(iteration + 1, total_iterations, torch.Tensor(losses).mean().item(),
                                              torch.Tensor(train_accuracies).mean().item()))
                        train_logger.info("For Task: {0}, the training loss is {1} and Training Accuracy is {2}".format(iteration + 1, torch.Tensor(losses).mean().item(),
                            torch.Tensor(train_accuracies).mean().item()))

                        avg_train_acc = torch.Tensor(train_accuracies).mean().item()
                        avg_train_loss = torch.Tensor(losses).mean().item()
                        
                        train_accuracies = []
                        losses = []

                    if ((iteration + 1) % self.args.save_freq == 0) and (iteration + 1) != total_iterations:
                        self.save_checkpoint(iteration + 1)


                    if ((iteration + 1) in self.args.test_iters) and (iteration + 1) != total_iterations:
                        accuracy_dict = self.test(session, iteration + 1)
                        print(accuracy_dict)
                        self.test_accuracies.print(self.logfile, accuracy_dict)

                # save the final model
                torch.save(self.model.state_dict(), self.checkpoint_path_final)

        self.logfile.close()

    def train_task(self, task_dict):
        context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list = self.prepare_task(task_dict)

        context_images = context_images.to(self.device)
        context_labels = context_labels.to(self.device)
        target_images = target_images.to(self.device)

        model_dict = self.model(context_images, context_labels, target_images)
        target_logits = model_dict['logits'].to(self.device)

        # Target logits after applying query-distance-based similarity metric on patch-level enriched features
        target_logits_post_pat = model_dict['logits_post_pat'].to(self.device)

        target_labels = target_labels.to(self.device)

        task_loss = self.loss(target_logits, target_labels, self.device) / self.args.tasks_per_batch
        task_loss_post_pat = self.loss(target_logits_post_pat, target_labels, self.device) / self.args.tasks_per_batch

        # Joint loss
        task_loss = task_loss + 0.1*task_loss_post_pat

        # Add the logits before computing the accuracy
        target_logits = target_logits + 0.1*target_logits_post_pat

        task_accuracy = self.accuracy_fn(target_logits, target_labels)

        task_loss.backward(retain_graph=False)

        return task_loss, task_accuracy

    def test(self, session, num_episode):
        self.model.eval()
        with torch.no_grad():

                self.video_loader.dataset.train = False
                accuracy_dict ={}
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

                    context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list = self.prepare_task(task_dict)
                    model_dict = self.model(context_images, context_labels, target_images)
                    target_logits = model_dict['logits'].to(self.device)

                    # Target logits after applying query-distance-based similarity metric on patch-level enriched features   
                    target_logits_post_pat = model_dict['logits_post_pat'].to(self.device)

                    target_labels = target_labels.to(self.device)

                    # Add the logits before computing the accuracy
                    target_logits = target_logits + 0.1*target_logits_post_pat

                    ############################
                    # NOTE start task analysis #
                    ############################
                    predictions = target_logits.argmax(dim=-1).squeeze()
                    wrong_predictions = predictions != target_labels
                    if wrong_predictions.sum() > 0:
                        # get the wrong predictions
                        batch_class_list = batch_class_list.int()
                        print("SUPPORT CLASSES:", [self.video_loader.dataset.class_folders[i] for i in batch_class_list])
                        wrong_target_labels = target_labels.int()[wrong_predictions]
                        wrong_predictions_labels = predictions.int()[wrong_predictions]
                        wrong_target_images = target_images.reshape(-1, 8, 3, 224, 224)[wrong_predictions]

                        # get also wrong activations
                        support_activations, target_activations = self.model.activations
                        target_activations = target_activations.reshape(-1, 8, 2048, 7, 7)[wrong_predictions]
                        target_activations = target_activations.norm(dim=2)

                        for pred, true, video, activations in zip(wrong_predictions_labels, wrong_target_labels, wrong_target_images, target_activations):
                            print("TRUE", self.video_loader.dataset.class_folders[batch_class_list[true]])
                            print("PRED", self.video_loader.dataset.class_folders[batch_class_list[pred]])
                            images_with_activations = []
                            for img, act in zip(video, activations):

                                img = img.cpu().numpy().swapaxes(0, 1).swapaxes(1, 2)
                                act = act.cpu().numpy()
                                act = (act - act.min()) / (act.max() - act.min())
                                act = cv2.resize(act, (img.shape[1], img.shape[0]))
                                heatmap = cv2.applyColorMap(np.uint8(255 * act), cv2.COLORMAP_JET)

                                img = np.uint8(255 * img)
                                overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

                                images_with_activations.append(overlay)

                                cv2.imshow("Overlay", overlay)
                                cv2.waitKey(0)

                            # Concatenate images horizontally
                            concatenated_image = cv2.hconcat(images_with_activations)
                            t = self.video_loader.dataset.class_folders[batch_class_list[true]]
                            p = self.video_loader.dataset.class_folders[batch_class_list[pred]]
                            save_path = f"wrong_predictions_activations/true_{t}_pred_{p}.png"
                            cv2.imwrite(save_path, concatenated_image)

                        
                    self.model.activations = []
                    ############################
                    # NOTE end   task analysis #
                    ############################

                    accuracy = self.accuracy_fn(target_logits, target_labels)
                    
                    loss = self.loss(target_logits, target_labels, self.device)/self.args.num_test_tasks
                   
                    # Loss using the new distance metric after  patch-level enrichment
                    loss_post_pat = self.loss(target_logits_post_pat, target_labels, self.device)/self.args.num_test_tasks

                    # Joint loss
                    loss = loss + 0.1*loss_post_pat

                    eval_logger.info("For Task: {0}, the testing loss is {1} and Testing Accuracy is {2}".format(iteration + 1, loss.item(),
                            accuracy.item()))
                    losses.append(loss.item())    
                    accuracies.append(accuracy.item())
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

        if images_to_device:
            context_images = context_images.to(self.device)
            target_images = target_images.to(self.device)
        context_labels = context_labels.to(self.device)
        target_labels = target_labels.type(torch.LongTensor).to(self.device)

        return context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list  

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
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            filtered_dict = {k.replace('.module', ''):v for k,v in checkpoint['model_state_dict'].items()}
            self.model.load_state_dict(filtered_dict)
            self.model.new_dist_loss_post_pat = [n.cuda(0) for n in self.model.new_dist_loss_post_pat]
            
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])


if __name__ == "__main__":
    main()
