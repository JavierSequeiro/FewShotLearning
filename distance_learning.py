
import torch
import argparse
from src.model import *
from scripts.evaluate import Evaluator
from src.dataset import *
from configs.config_distance import myConfig
from src.myLogger import myLogger
from src.results_plotter import ResultsPlotter
from scripts.train import myTrainerDistance

 

if __name__ == "__main__":
    """
    Load Parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_name", "--experiment_name", help="Experiment name for logging", type=str, required=True)
    args = parser.parse_args()
    print(args)

    torch.cuda.empty_cache()
    """
    Load Config file and Logger
    """
    config = myConfig()
    logger = myLogger(logger_name=__name__, config=config).get_logger()
    torch.manual_seed(config.rand_seed)

    metrics_lists  = {"train_acc": [],
                      "val_acc": [],
                      "CI_err": []}
    
    
    """
    Name Specific Experiment based on argument parsed
    """
    config.specific_experiment_val = args.experiment_name
    logger.info(f"Working with Positive Margin: {config.positive_margin} || Negative Margin: {config.negative_margin}")

    """
    Load Data
    """
    train_dataset, val_dataset, test_dataset = myFinalDataset(config=config,
                                                                logger=logger)._get_dataset()

    """
    Load Model
    """
    if "resnet" in config.model_name:
        model, arch_name = ResNet(config=config,
                                    logger=logger).load_pretrained_resnet_distance()
        config.model_name = arch_name
    else:
        # Will implement functionality to work with different architectures
        pass
    
    """
    Train Model (or skip in case personal model is loaded)
    """
    if config.train_new_model:
        trained_model, train_acc, val_acc, CI_err = myTrainerDistance(config=config,
                                                                        logger=logger).train_model(model=model,
                                                                                                    train_set=train_dataset,
                                                                                                    val_set= val_dataset,
                                                                                                    show_training_curves=True,
                                                                                                    return_acc=True)
        # Store Train and Validation Accuracy
        # metrics_lists["train_acc"].append(train_acc)
        metrics_lists["val_acc"].append(val_acc)
        metrics_lists["CI_err"].append(CI_err)
    else:
        trained_model = model
    
    """
    Evaluate Model
    """
    Evaluator(model=trained_model,
                test_set=train_dataset,
                config=config,
                logger=logger).evaluate_distance_model()

    """
    Plot results with Confidence Interval
    """
    ResultsPlotter(config=config,
                    logger=logger).plot_accuracies_w_CI(metrics_dict=metrics_lists,
                                                        variable_values=[args.experiment_name])




