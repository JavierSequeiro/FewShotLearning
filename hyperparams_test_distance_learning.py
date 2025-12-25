
import torch
from src.model import *
from scripts.evaluate import Evaluator
from src.dataset import *
from configs.config_distance import myConfig
from src.myLogger import myLogger
from src.results_plotter import ResultsPlotter
from scripts.train import myTrainerDistance

 

if __name__ == "__main__":
    torch.cuda.empty_cache()
    config = myConfig()
    logger = myLogger(logger_name=__name__, config=config).get_logger()
    torch.manual_seed(config.rand_seed)

    # Load Data
    # variable = "K"
    # Support Points
    sp_variable_values = [20]
    # K values KNN
    # k_variable_values = [5, 7, 11]
    k_val = 11
    # neg_margin_values = [1.0, 1.3, 1.5]
    neg_margin_values = [0.2, 0.3, 0.4]
    pos_margin_values = [0.7, 0.8, 0.9]
    
    merged_variable_values = []

    metrics_lists  = {"train_acc": [],
                      "val_acc": [],
                      "CI_err": []}
    """
    FOR LOOP to Perform several experiments varying hyperparameters
    """
    # for sp_val in sp_variable_values:
    for pos_marg in pos_margin_values:
        # config.K_support_points = sp_val
        config.positive_margin = pos_marg
        # for k_val in k_variable_values:
        for neg_marg in neg_margin_values:
            # config.K_knn = k_val
            config.negative_margin = neg_marg
            # config.specific_experiment_val = f"SP_{sp_val}_K_{k_val}"
            # config.specific_experiment_val = f"SP_{sp_val}_Neg_Marg_{str(neg_marg)[-1]}"
            config.specific_experiment_val = f"Pos_Marg_0_{str(pos_marg)[-1]}_Neg_Marg_0_{str(neg_marg)[-1]}"

            # if k_val <= sp_val:
            merged_variable_values.append(config.specific_experiment_val)
            # logger.info(f"Working with SUPPORT POINTS: {config.K_support_points} || K KNN: {config.K_knn}")
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
                                                                variable_values=merged_variable_values)




