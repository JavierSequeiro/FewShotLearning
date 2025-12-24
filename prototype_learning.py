
import torch
from src.model import *
from scripts.evaluate import Evaluator
from src.dataset import myProtoDataset
from configs.config_proto import myConfig
from scripts.train import  myTrainerProto
from src.myLogger import myLogger
from src.results_plotter import ResultsPlotter



if __name__ == "__main__":
    torch.cuda.empty_cache()
    config = myConfig()
    logger = myLogger(logger_name=__name__, config=config).get_logger()
    torch.manual_seed(config.rand_seed)
    # Load Data
    
    # variable_values = [128, 256, 512]
    variable_values = ["resnet34", "resnet50"]

    metrics_lists  = {"train_acc": [],
                      "val_acc": [],
                      "CI_err": []}
    """
    FOR LOOP to Perform several experiments varying hyperparameters
    """
    for val in variable_values:
        
        # config.final_layer_dim = val
        config.model_name = val
        config.specific_experiment_val = f"Model_{val}"

        logger.info(f"Working with Model = {val}")
        
        """
        Load Data
        """
        train_dataset, val_dataset, test_dataset = myProtoDataset(config=config, logger=logger)._get_dataset()
      
        """
        Load Model
        """

        if "resnet" in config.model_name:
            model, arch_name = ResNet(config=config, logger=logger).load_pretrained_resnet_prototypes()
            config.model_name = arch_name
        else:
            pass
        
        """
        Train Model (or skip in case personal model is loaded)
        """
        if config.train_new_model: 
            trained_model, train_acc, val_acc, CI_err = myTrainerProto(config=config, logger=logger).train_model(model=model,
                                                                train_set=train_dataset,
                                                                val_set=val_dataset,
                                                                show_training_curves=True,
                                                                return_acc=True)
            # Store Train and Validation Accuracy
            metrics_lists["val_acc"].append(val_acc)
            metrics_lists["CI_err"].append(CI_err)
        else:
            trained_model = model
        
        """
        Evaluate Model
        """
        Evaluator(model=trained_model,test_set=test_dataset, config=config, logger=logger).evaluate_proto_model()

        """
        Plot results with Confidence Interval
        """
        ResultsPlotter(config=config, logger=logger).plot_accuracies_w_CI(metrics_dict=metrics_lists,
                                                                    variable_values=variable_values)




