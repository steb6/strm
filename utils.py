import torch
import torch.nn.functional as F
import os
import math
from enum import Enum
import sys
from sklearn.metrics import roc_auc_score


class TestAccuracies:
    """
    Determines if an evaluation on the validation set is better than the best so far.
    In particular, this handles the case for meta-dataset where we validate on multiple datasets and we deem
    the evaluation to be better if more than half of the validation accuracies on the individual validation datsets
    are better than the previous best.
    """

    def __init__(self, validation_datasets):
        self.datasets = validation_datasets
        self.dataset_count = len(self.datasets)
#        self.current_best_accuracy_dict = {}
#        for dataset in self.datasets:
#            self.current_best_accuracy_dict[dataset] = {"accuracy": 0.0, "confidence": 0.0}

#    def is_better(self, accuracies_dict):
#        is_better = False
#        is_better_count = 0
#        for i, dataset in enumerate(self.datasets):
#            if accuracies_dict[dataset]["accuracy"] > self.current_best_accuracy_dict[dataset]["accuracy"]:
#                is_better_count += 1
#
#        if is_better_count >= int(math.ceil(self.dataset_count / 2.0)):
#            is_better = True
#
#        return is_better

#    def replace(self, accuracies_dict):
#        self.current_best_accuracy_dict = accuracies_dict

    def print(self, logfile, accuracy_dict):
        print_and_log(logfile, "")  # add a blank line
        print_and_log(logfile, "Test Accuracies:")
        for dataset in self.datasets:
            print_and_log(logfile, "{0:}: {1:.1f}+/-{2:.1f}".format(dataset, accuracy_dict[dataset]["accuracy"],
                                                                    accuracy_dict[dataset]["confidence"]))
        print_and_log(logfile, "")  # add a blank line

#    def get_current_best_accuracy_dict(self):
#        return self.current_best_accuracy_dict


def verify_checkpoint_dir(checkpoint_dir, resume, test_mode):
    if resume:  # verify that the checkpoint directory and file exists
        if not os.path.exists(checkpoint_dir):
            print("Can't resume for checkpoint. Checkpoint directory ({}) does not exist.".format(checkpoint_dir), flush=True)
            sys.exit()

        checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pt')
        if not os.path.isfile(checkpoint_file):
            print("Can't resume for checkpoint. Checkpoint file ({}) does not exist.".format(checkpoint_file), flush=True)
            sys.exit()
    #elif test_mode:
    #    if not os.path.exists(checkpoint_dir):
    #        print("Can't test. Checkpoint directory ({}) does not exist.".format(checkpoint_dir), flush=True)
    #        sys.exit()
    else:
        if os.path.exists(checkpoint_dir):
            # import shutil  # TODO ADDED BY ME
            # input("Are you sure to erase directory {checkpoint_dir}?")  # TODO ADDED BY ME
            # shutil.rmtree(checkpoint_dir)  # TODO ADDED BY ME
            print("Checkpoint directory ({}) already exits.".format(checkpoint_dir), flush=True)
            print("If starting a new training run, specify a directory that does not already exist.", flush=True)
            print("If you want to resume a training run, specify the -r option on the command line.", flush=True)
            # sys.exit()


def print_and_log(log_file, message):
    """
    Helper function to print to the screen and the cnaps_layer_log.txt file.
    """
    print(message, flush=True)
    log_file.write(message + '\n')


def get_log_files(checkpoint_dir, resume, test_mode):
    """
    Function that takes a path to a checkpoint directory and returns a reference to a logfile and paths to the
    fully trained model and the model with the best validation score.
    """
    verify_checkpoint_dir(checkpoint_dir, resume, test_mode)
    #if not test_mode and not resume:
    if not resume:
        os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path_validation = os.path.join(checkpoint_dir, 'best_validation.pt')
    checkpoint_path_final = os.path.join(checkpoint_dir, 'fully_trained.pt')
    logfile_path = os.path.join(checkpoint_dir, 'log.txt')
    if os.path.isfile(logfile_path):
        logfile = open(logfile_path, "a", buffering=1)
    else:
        logfile = open(logfile_path, "w", buffering=1)

    return checkpoint_dir, logfile, checkpoint_path_validation, checkpoint_path_final


def stack_first_dim(x):
    """
    Method to combine the first two dimension of an array
    """
    x_shape = x.size()
    new_shape = [x_shape[0] * x_shape[1]]
    if len(x_shape) > 2:
        new_shape += x_shape[2:]
    return x.view(new_shape)


def split_first_dim_linear(x, first_two_dims):
    """
    Undo the stacking operation
    """
    x_shape = x.size()
    new_shape = first_two_dims
    if len(x_shape) > 1:
        new_shape += [x_shape[-1]]
    return x.view(new_shape)


def sample_normal(mean, var, num_samples):
    """
    Generate samples from a reparameterized normal distribution
    :param mean: tensor - mean parameter of the distribution
    :param var: tensor - variance of the distribution
    :param num_samples: np scalar - number of samples to generate
    :return: tensor - samples from distribution of size numSamples x dim(mean)
    """
    sample_shape = [num_samples] + len(mean.size())*[1]
    normal_distribution = torch.distributions.Normal(mean.repeat(sample_shape), var.repeat(sample_shape))
    return normal_distribution.rsample()


def loss(test_logits_sample, test_labels, device):
    """
    Compute the classification loss.
    """
    size = test_logits_sample.size()
    sample_count = size[0]  # scalar for the loop counter
    num_samples = torch.tensor([sample_count], dtype=torch.float, device=device, requires_grad=False)

    log_py = torch.empty(size=(size[0], size[1]), dtype=torch.float, device=device)
    for sample in range(sample_count):
        log_py[sample] = -F.cross_entropy(test_logits_sample[sample], test_labels, reduction='none')
    score = torch.logsumexp(log_py, dim=0) - torch.log(num_samples)
    return -torch.sum(score, dim=0)


def aggregate_accuracy(test_logits_sample, test_labels):
    if len(test_logits_sample.shape) > 2:
        test_logits_sample = test_logits_sample.squeeze(0)

    # Closed set accuracy
    known_indices = test_labels != -1
    known_test_logits_sample, known_test_labels = test_logits_sample[known_indices], test_labels[known_indices]
    known_predictions = torch.argmax(known_test_logits_sample, dim=-1)
    closed_set_acc = torch.mean(torch.eq(known_test_labels, known_predictions).float()).item()

    # AUROC for open set
    should_be_one_indices = test_labels[known_indices] + (torch.arange(0, len(test_labels[known_indices])).to(known_indices.device) * test_logits_sample.shape[1])
    should_be_one = test_logits_sample[known_indices].reshape(-1)[should_be_one_indices]

    flat_logits = test_logits_sample.reshape(-1)
    indices = torch.ones_like(flat_logits, dtype=bool)
    indices[should_be_one_indices] = False
    should_be_zero = flat_logits[indices]
    aurroc = roc_auc_score(torch.cat([torch.ones_like(should_be_one), torch.zeros_like(should_be_zero)]).detach().cpu().numpy(),
                           torch.cat([should_be_one, should_be_zero]).detach().cpu().numpy())

    return {"all": 0, "closed": closed_set_acc, "open": aurroc}
    


def task_confusion(test_logits, test_labels, real_test_labels, batch_class_list):
    preds = torch.argmax(torch.logsumexp(test_logits, dim=0), dim=-1)
    real_preds = batch_class_list[preds]
    return real_preds

def linear_classifier(x, param_dict):
    """
    Classifier.
    """
    return F.linear(x, param_dict['weight_mean'], param_dict['bias_mean'])

def binary_classification_accuracy(predictions, labels, threshold=0.5):
    """
    Compute binary classification accuracy.
    
    Args:
    - predictions (torch.Tensor): The predicted probabilities (output of the sigmoid function).
    - labels (torch.Tensor): The true labels (0 or 1).
    - threshold (float): The threshold to convert probabilities to binary predictions.
    
    Returns:
    - accuracy (float): The accuracy of the binary classification.
    """
    # Convert probabilities to binary predictions
    binary_predictions = (predictions >= threshold).float()
    
    # Compare predictions with true labels
    correct_predictions = (binary_predictions == labels).float()
    
    # Compute accuracy
    accuracy = correct_predictions.mean().item()
    
    return accuracy
