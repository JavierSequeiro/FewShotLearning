import matplotlib.pyplot as plt
import os
import os.path as osp
import numpy as np

class ResultsPlotter:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        os.makedirs(self.config.plot_results_path, exist_ok=True)

    def plot_accuracies(self, metrics_dict, variable_values):

        self.logger.info("SHOW TRAINING CURVES UP AND RUNNING CONSTRUCTION!")

        plt.figure(figsize=(14,10))
        for i, t_acc in enumerate(metrics_dict["train_acc"]):

            plt.subplot(1,2,1)
            plt.plot(t_acc,label=f"{self.config.specific_experiment} = {variable_values[i]}",)
        # plt.plot(metrics_dict["val_acc"],label="Valid. Acc.", color="orange")
        plt.xlabel("Epochs")
        plt.title("Train Accuracy")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(0, 1)
        plt.legend()


        for i, v_acc in enumerate(metrics_dict["val_acc"]):

            plt.subplot(1,2,2)
            plt.plot(v_acc,label=f"{self.config.specific_experiment} = {variable_values[i]}",)
        plt.xlabel("Epochs")
        plt.title("Validation Accuracy")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(0, 1)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(osp.join(self.config.plot_results_path, f"multiple_exps_metrics_plot_{self.config.specific_experiment}.png"))

        del self.logger

    
    def plot_accuracies_w_CI(self, metrics_dict, variable_values):

        self.logger.info("SHOW TRAINING CURVES UP AND RUNNING CONSTRUCTION!")

        plt.figure(figsize=(14,10))


        for i, v_acc in enumerate(metrics_dict["val_acc"]):

            CI_err = metrics_dict["CI_err"][i]
            print(f"V acc: {v_acc} || i {i}")
            print(f"CI Err: {CI_err}")
            x = np.linspace(1,len(v_acc),len(v_acc))

            plt.errorbar(x=x,y=v_acc,yerr=CI_err, label=f"{self.config.specific_experiment} = {variable_values[i]}",marker='o', linestyle='-')

        plt.xlabel("Epochs")
        plt.title("Validation Accuracy")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(0, 1)
        plt.legend()
        

        plt.tight_layout()
        plt.savefig(osp.join(self.config.plot_results_path, f"multiple_exps_metrics_plot_{self.config.specific_experiment}.png"))
