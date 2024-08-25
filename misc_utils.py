import math
from prettytable import PrettyTable


def model_summary(model):
    print()
    print('model_summary')
    
    table = PrettyTable(["Modules", "Parameters", "Trainable parameters"])
    total_params = 0
    total_trainable_params = 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        if not parameter.requires_grad:
            trainable_params = 0
        else:
            trainable_params = params
        
        table.add_row([name, params, trainable_params])
        total_params += params
        total_trainable_params += trainable_params
        
    print(table)
    print(f"Total params: {total_params}")
    print(f"Total trainable params: {total_trainable_params}")
    print()


def check_classification_model_metric_is_best(best_metric, cur_metric):
    metric_names = ['acc', 'train_acc', 'f1', 'train_f1']
    for metric in metric_names:
        best = best_metric[metric]
        cur = cur_metric[metric]
        if best < cur:
            print(f"{metric} improved: {best:.4f} ===> {cur:.4f}")
            return True
        if not math.isclose(cur, best, abs_tol=1e-6):
            return False
    return False
