import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
import torch.distributed as dist
from distutils.util import strtobool
from datetime import datetime
import os
plt.switch_backend('agg')

def save_result(args, obs, estimation, state, filedir="/home/xingyining/liushiqi/LLM-Filter/results/"):
    """
    Save the results of LLMFilter_tracking as a single .npz file.

    Parameters:
    - args (object): Should contain `model_id` as an attribute for system identification.
    - obs (array-like): Observations from the system.
    - estimation (array-like): Estimated values from the LLM-Filter.
    - state (array-like): True states of the system.
    - filedir (str): Directory where the results should be saved. Default is provided.

    Returns:
    - None
    """
    # Ensure the inputs are numpy arrays
    obs = np.array(obs)
    estimation = np.array(estimation)
    state = np.array(state)
    
    # Create the full file path
    if not os.path.exists(filedir):
        os.makedirs(filedir)  # Create directory if it doesn't exist

    filename = os.path.join(filedir, f"LLMFilter_{args.model}_{args.model_id}_results.npz")
    
    # Save arrays to a single .npz file
    np.savez(filename, obs=obs, estimation=estimation, state=state)
    
    print(f"Results successfully saved to {filename}")

def get_model_config(model_id):
    """Retrieve observation and state dimensions based on model_id."""
    model_configs = {
        'tracking': (2, 4), 'pendulum': (2, 4),
        'hopf': (2, 2), 'oscillator': (2, 2),
        'selkov': (2, 2), 'hopfC': (2, 2), 
        'oscillatorC': (2, 2), 'selkovC': (2, 2),
        'oscillator100': (2, 2), 'oscillator75': (2, 2), 'oscillator50': (2, 2), 'oscillator25': (2, 2), 'selkov100': (2, 2), 'selkov75': (2, 2),
        'selkov50': (2, 2), 'selkov26': (2, 2),
        'hopf100': (2, 2), 'hopf75': (2, 2), 
        'hopf50': (2, 2), 'hopf25': (2, 2),
        'lorenz96': (72, 72), 'VL20': (72, 72), 
        'lorenz96C': (72, 72), 'VL20C': (72, 72),
    }
    return model_configs.get(model_id, (None, None))

def load_content(args):
    file_name = os.path.splitext(args.data_path)[0]
    if args.prompt_domain == 1:
        with open('./dataset/System-as-Prompt/{0}.txt'.format(file_name), 'r') as f:
            content = f.read()
    elif args.prompt_domain == 2:
        with open('./dataset/prompt_bank/{0}.txt'.format(file_name), 'r') as f:
            content = f.read()
    else:
        raise ValueError('Invalid prompt domain')
    return content

def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** (epoch - 1))}    
    elif args.lradj == 'type2':
        lr_adjust = {epoch: args.learning_rate * (0.6 ** epoch)}  
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
            print('next learning rate is {}'.format(lr))

class EarlyStopping:
    def __init__(self, args, verbose=False, delta=0):
        self.patience = args.patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.use_multi_gpu = args.use_multi_gpu
        if self.use_multi_gpu:
            self.local_rank = args.local_rank
        else:
            self.local_rank = None

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                if (self.use_multi_gpu and self.local_rank == 0) or not self.use_multi_gpu:
                    print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
            if self.use_multi_gpu:
                if self.local_rank == 0:
                    self.save_checkpoint(val_loss, model, path)
                dist.barrier()
            else:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if (self.use_multi_gpu and self.local_rank == 0) or not self.use_multi_gpu:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.use_multi_gpu:
                if self.local_rank == 0:
                    self.save_checkpoint(val_loss, model, path)
                dist.barrier()
            else:
                self.save_checkpoint(val_loss, model, path)
            if self.verbose:
                if (self.use_multi_gpu and self.local_rank == 0) or not self.use_multi_gpu:
                    print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
            self.counter = 0


    def save_checkpoint(self, val_loss, model, path):
        param_grad_dic = {
        k: v.requires_grad for (k, v) in model.named_parameters()
        }
        state_dict = model.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        torch.save(state_dict, path + '/' + f'checkpoint.pth')
        


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )