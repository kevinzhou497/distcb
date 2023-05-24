import numpy as np
import torch
import torchvision
from torchvision.models import resnet18
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional
import torch.nn as nn
from torch.utils.data import RandomSampler
import pandas as pd
from king_housing_preprocess import *


class CBDataset(Dataset):
    def __init__(self, context, label):
        self.context = context
        self.label = label

    def __len__(self):
        return len(self.context)

    def __getitem__(self, idx):
        context_val = torch.tensor(self.context.iloc[idx].values, dtype=torch.float)
        label_val = torch.tensor(self.label.iloc[idx])
        return context_val, label_val


# dictionary to map fine labels to coarse labels for the CIFAR-100 experiment
# source: https://gist.github.com/adam-dziedzic/4322df7fc26a1e75bee3b355b10e30bc
fine_to_coarse = {
    0: 4,
    1: 1,
    2: 14,
    3: 8,
    4: 0,
    5: 6,
    6: 7,
    7: 7,
    8: 18,
    9: 3,
    10: 3,
    11: 14,
    12: 9,
    13: 18,
    14: 7,
    15: 11,
    16: 3,
    17: 9,
    18: 7,
    19: 11,
    20: 6,
    21: 11,
    22: 5,
    23: 10,
    24: 7,
    25: 6,
    26: 13,
    27: 15,
    28: 3,
    29: 15,
    30: 0,
    31: 11,
    32: 1,
    33: 10,
    34: 12,
    35: 14,
    36: 16,
    37: 9,
    38: 11,
    39: 5,
    40: 5,
    41: 19,
    42: 8,
    43: 8,
    44: 15,
    45: 13,
    46: 14,
    47: 17,
    48: 18,
    49: 10,
    50: 16,
    51: 4,
    52: 17,
    53: 4,
    54: 2,
    55: 0,
    56: 17,
    57: 4,
    58: 18,
    59: 17,
    60: 10,
    61: 3,
    62: 2,
    63: 12,
    64: 12,
    65: 16,
    66: 12,
    67: 1,
    68: 9,
    69: 19,
    70: 2,
    71: 10,
    72: 0,
    73: 1,
    74: 16,
    75: 12,
    76: 9,
    77: 13,
    78: 15,
    79: 13,
    80: 16,
    81: 19,
    82: 2,
    83: 4,
    84: 6,
    85: 19,
    86: 5,
    87: 5,
    88: 8,
    89: 19,
    90: 18,
    91: 1,
    92: 2,
    93: 15,
    94: 6,
    95: 0,
    96: 17,
    97: 8,
    98: 14,
    99: 13,
}


def algorithm(
    dataloader,
    num_actions,
    time_steps,
    batch_size,
    pred_classes,
    experiment,
    model_type,
):
    """Runs the contextual bandit experiment.
    
    Args:
        dataloader: A PyTorch Dataloader with data sampled from the dataset
        num_actions: The number of possible actions
        time_steps: The number of episodes the experiment should run for
        batch_size: The number of samples in each batch
        pred_classes: The number of possible costs 
        experiment: The name of the dataset being used, either "Housing" "Prudential" or "CIFAR"
        model_type: The name of the algorithm being used, either "SquareCB" "FastCB" or "DistributionalCB"
    """
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"

    # Note: several hyperparameters and logging were set in WandB sweeps, which have been removed 
    # The following are dummy values to represent a potential configuration of these hyperparameters
    lr = 0.001
    p = 0.25
    gamma_knot = 100
    optimizer = "Adam"
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    if experiment == "Prudential":
        input_size = 1887
        hidden_sizes = [512, 256]
        if model_type == "DistributionalCB":
            output_size = num_actions * pred_classes
        else:
            output_size = num_actions

        model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size),
        )
        ev_mapping = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]).to(
            device
        )
    elif experiment == "Housing":
        input_size = 88
        hidden_sizes = [32768, 16384]
        if model_type == "DistributionalCB":
            output_size = num_actions * pred_classes
        else:
            output_size = num_actions

        model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size),
        )

        ev_mapping = (torch.arange(0, 1.01, 0.01)).to(device)
    else:
        model = resnet18()
        in_features = model.fc.in_features
        out_features = num_actions
        model.fc = torch.nn.Linear(in_features, out_features)

        ev_mapping = torch.tensor([0.0, 0.5, 1.0]).to(device)

    if experiment == "CIFAR":
        if model_type == "DistributionalCB":
            model.fc = torch.nn.Linear(model.fc.in_features, num_actions * pred_classes)

    model = model.to(device)
    if optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr)
    elif optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr)

    cost = 0
    costs = []
    total_cost = 0
    cost_last_block = 0

    if model_type == "FastCB":
        criterion = torch.nn.BCELoss()
    elif model_type == "DistributionalCB":
        criterion = torch.nn.CrossEntropyLoss()
    # SquareCB
    else:
        criterion = torch.nn.MSELoss()

    for t, (context, label) in enumerate(dataloader):
        context, label = context.to(device), label.to(device)
        # Get the coarse labels for CIFAR
        if experiment == "CIFAR":
            coarse_label = torch.empty((batch_size))
            fine_to_coarse_func = np.vectorize(fine_to_coarse.get)
            coarse_label = torch.tensor(
                fine_to_coarse_func(label.detach().cpu().numpy())
            )
            coarse_label = coarse_label.to(device)

        # get the predictions from the oracle
        output = model(context)
        if model_type == "DistributionalCB":
            output = output.view(-1, batch_size, num_actions, pred_classes)
            logits_predictions = output
            predictions = torch.nn.functional.softmax(output, dim=-1)
        else:
            predictions = torch.nn.functional.sigmoid(output)
        expected_values = np.zeros((batch_size, num_actions), dtype=float)
        best_action = np.zeros((batch_size), dtype=int)
        best_action_cost = np.zeros((batch_size), dtype=float)

        # getting the best actions and associated costs
        if model_type == "DistributionalCB":
            # expected values calculation from the distribution
            expected_values = torch.einsum("ijkl,l->jk", predictions, ev_mapping)

            best_action = torch.argmin(expected_values, dim=1)
            best_action_cost = expected_values[torch.arange(batch_size), best_action]
        else:
            best_action = torch.argmin(predictions, dim=-1)
            best_action_cost = predictions[torch.arange(batch_size), best_action]

        probabilities = torch.zeros((batch_size, num_actions), dtype=float)

        # setting the gamma per timestep
        gamma = gamma_knot * ((t + 1) ** p)

        # IGW / ReIGW procedure
        for i in range(batch_size):
            if model_type == "SquareCB":
                values = predictions[i] - best_action_cost[i]
                zero_values = torch.isclose(values, torch.zeros_like(values), atol=1e-9)
                probability = torch.where(
                    zero_values,
                    0.0,
                    (
                        1
                        / (num_actions + gamma * (predictions[i] - best_action_cost[i]))
                    ),
                )
                prob_sum = torch.sum(probability)
                best_actions_idx = torch.eq(probability, 0.0)
                num_best_actions = torch.sum(best_actions_idx)
                probability = torch.where(
                    best_actions_idx,
                    (1.0 - prob_sum.item()) / num_best_actions,
                    probability,
                )

            elif model_type == "FastCB":
                values = predictions[i] - best_action_cost[i]
                zero_values = torch.isclose(values, torch.zeros_like(values), atol=1e-9)
                probability = torch.where(
                    zero_values,
                    0.0,
                    (
                        best_action_cost[i]
                        / (
                            num_actions * best_action_cost[i]
                            + gamma * (predictions[i] - best_action_cost[i])
                        )
                    ),
                )
                prob_sum = torch.sum(probability)
                best_actions_idx = torch.eq(probability, 0.0)
                num_best_actions = torch.sum(best_actions_idx)
                probability = torch.where(
                    best_actions_idx,
                    (1.0 - prob_sum.item()) / num_best_actions,
                    probability,
                )

            elif model_type == "DistributionalCB":
                values = expected_values[i] - best_action_cost[i]
                zero_values = torch.isclose(values, torch.zeros_like(values), atol=1e-9)
                probability = torch.where(
                    zero_values,
                    0.0,
                    (
                        best_action_cost[i]
                        / (
                            num_actions * best_action_cost[i]
                            + gamma * (expected_values[i] - best_action_cost[i])
                        )
                    ),
                )
                prob_sum = torch.sum(probability)
                best_actions_idx = torch.eq(probability, 0.0)
                num_best_actions = torch.sum(best_actions_idx)
                probability = torch.where(
                    best_actions_idx,
                    (1.0 - prob_sum.item()) / num_best_actions,
                    probability,
                )

            probabilities[i] = probability

        if experiment == "CIFAR":
            actions = list(range(num_actions))
            actions = np.array(actions)
        else:
            actions = list(range(1, num_actions + 1))
            actions = np.array(actions)

        probabilities = probabilities.cpu().detach().numpy()
        predicted_action = np.zeros((batch_size), dtype=int)

        # ensuring probabiliites add up to 1, accounts for potential np float precision issues
        probability_sums = np.sum(probabilities, axis=1).reshape(batch_size, 1)
        probabilities = np.divide(probabilities, probability_sums)

        # getting the model's chosen action
        for i in range(batch_size):
            predicted_action[i] = int(np.random.choice(actions, 1, p=probabilities[i]))
        predicted_action = torch.tensor(predicted_action).to(device)

        optimizer.zero_grad()
        cost = torch.empty((batch_size), dtype=torch.float64)
        cost_index = torch.empty((batch_size), dtype=torch.long)
        if model_type == "DistributionalCB":
            chosen_predictions = torch.empty((batch_size, pred_classes), dtype=float)
        else:
            chosen_predictions = torch.empty((batch_size), dtype=float)

        label = label.flatten()

        # gathering the model predictions for each chosen action
        if model_type == "DistributionalCB":
            if experiment == "CIFAR":
                chosen_predictions = logits_predictions[
                    0, np.arange(batch_size), predicted_action
                ]
            else:
                chosen_predictions = logits_predictions[
                    0, np.arange(batch_size), predicted_action - 1
                ]
        else:
            if experiment == "CIFAR":
                chosen_predictions = predictions[
                    np.arange(batch_size), predicted_action
                ]
            else:
                chosen_predictions = predictions[
                    np.arange(batch_size), predicted_action - 1
                ]

        # getting cost
        if experiment == "Prudential":
            over_predict = predicted_action > label
            cost = torch.where(over_predict, 1.0, 0.1 * (label - predicted_action))

            full_cost = cost == 1.0
            cost_index = torch.where(full_cost, 8, (cost * 10).type(torch.int))
            cost_index = cost_index.type(torch.long)

        elif experiment == "Housing":
            over_predict = predicted_action / 100 > label
            cost = torch.where(over_predict, 1.0, 1.0 - predicted_action / 100)
            cost_index = (cost * 100).type(torch.long)

        elif experiment == "CIFAR":
            fine_to_coarse_func = np.vectorize(fine_to_coarse.get)
            coarse_predicted = torch.tensor(
                fine_to_coarse_func(predicted_action.detach().cpu().numpy())
            ).to(device)
            full_correct = (predicted_action == label).cpu().numpy()
            coarse_correct = (
                ((coarse_predicted == coarse_label) & (predicted_action != label))
                .cpu()
                .numpy()
            )
            incorrect = (coarse_predicted != coarse_label).cpu().numpy()
            correct_cost = np.full((batch_size,), 0.0)
            half_cost = np.full((batch_size,), 0.5)
            incorrect_cost = np.full((batch_size,), 1.0)
            cost = np.select(
                [full_correct, coarse_correct, incorrect],
                [correct_cost, half_cost, incorrect_cost],
            )
            cost = torch.tensor(cost)
            cost_index = (cost * 2).type(torch.long)

        model_loss = 0
        cost = cost.to(device)
        cost_index = cost_index.to(device)

        chosen_predictions = chosen_predictions.type(torch.float)
        cost = cost.type(torch.float)

        # getting the model's loss
        if model_type == "SquareCB" or model_type == "FastCB":
            chosen_predictions = chosen_predictions.to(device)
            model_loss = criterion(chosen_predictions, cost)
        else:
            chosen_predictions = chosen_predictions.to(device)
            model_loss = criterion(chosen_predictions, cost_index)

        episode_cost = torch.sum(cost).item()
        model_item = model_loss.item()
        total_cost += episode_cost

        # recording cost over the last 100 episodes
        if t >= time_steps - 100:
            cost_last_block += episode_cost

        costs.append(episode_cost)
        # update
        model_loss.backward()
        optimizer.step()
    return model, costs


# Note: in the experiments functions, the paths to files and datasets are removed

def cifar_experiment():
    # CIFAR transforms
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4866, 0.4409], [0.2673, 0.2564, 0.2762]),
        ]
    )

    # fill in with path to desired location of CIFAR data
    cifar100_train_data = torchvision.datasets.CIFAR100(
        root="", train=True, transform=train_transform, download=True
    )
    
    batch = 32
    num_actions = 100
    time_steps = 15000
    model_type = "DistributionalCB"
    sampler = RandomSampler(
        cifar100_train_data, replacement=True, num_samples=time_steps * batch
    )
    train_dataloader = DataLoader(
        cifar100_train_data,
        batch_size=batch,
        pin_memory=True,
        drop_last=True,
        sampler=sampler
    )
    algorithm(train_dataloader, num_actions, time_steps, batch, 3, "CIFAR", model_type)


def prudential_experiment():
    # fill in with path to downloaded Prudential Kaggle competition data
    original_data = pd.read_csv("")

    # fill in with path to preprocessed Prudential context data
    data = pd.read_csv("")

    dataset = CBDataset(data, original_data["Response"])
    time_steps = 5000
    batch = 32
    num_actions = 8
    model_type = "DistributionalCB"
    sampler = RandomSampler(dataset, replacement=True, num_samples=time_steps * batch)
    train_dataloader = DataLoader(
        dataset, batch_size=batch, pin_memory=True, drop_last=True, sampler=sampler
    )
    algorithm(
        train_dataloader,
        num_actions,
        time_steps,
        batch,
        num_actions + 1,
        "Prudential",
        model_type
    )


def housing_experiment():
    # fill in with path to arff file
    dataset = ArffToPytorch(
        "", target="price", skipcol=["id"], skiprow=lambda z: z["price"] > 1e6
    )
    time_steps = 5000
    batch = 32
    num_actions = 100
    model_type = "DistributionalCB"
    sampler = RandomSampler(dataset, replacement=True, num_samples=time_steps * batch)
    train_dataloader = DataLoader(
        dataset, batch_size=batch, pin_memory=True, drop_last=True, sampler=sampler
    )
    algorithm(
        train_dataloader,
        num_actions,
        time_steps,
        batch,
        num_actions + 1,
        "Housing",
        model_type
    )
