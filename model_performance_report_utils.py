import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics, preprocessing


def compute_stats(y_true, y_pred, literal_labels=None, mode='binary', cm_plot_name=None):
    assert mode in ['binary', 'multiclass'], f"Invalid mode: Mode must be either binary or multiclass."

    correct = 0
    cnt = [[0] * 2 for _ in range(2)]
    for label, target_label in zip(y_true, y_pred):
        correct += int(label == target_label)
        if mode == 'binary':
            if label == target_label:
                cnt[0][target_label] += 1
            else:
                cnt[1][target_label] += 1

    if mode == 'binary':
        fp, tp, fn, tn = cnt[1][1], cnt[0][1], cnt[1][0], cnt[0][0]
        print(f'fp={fp}, tp={tp}, fn={fn}, tn={tn}')

    acc = correct * 100 / len(y_true)

    average = mode if mode == 'binary' else 'macro'
    prec = metrics.precision_score(y_true, y_pred, average=average) * 100
    rec = metrics.recall_score(y_true, y_pred, average=average) * 100
    f1 = metrics.f1_score(y_true, y_pred, average=average)
    if mode == 'binary':
        far = fn / (fn + tp) * 100
        frr = fp / (fp + tn) * 100

    n_labels = len(literal_labels)

    print(f'Accuracy: {acc:.3f}%')
    print(f'Precision: {prec:.3f}%')
    print(f'Recall: {rec:.3f}%')
    print(f'F1 Score: {f1:.3f}')
    if mode == 'binary':
        print(f'FAR: {far:.3f}%')
        print(f'FRR: {frr:.3f}%')

    print('---------------------------- CLASSIFICATION REPORT ----------------------------')
    if literal_labels is not None:
        y_true_literals = [literal_labels[x] for x in y_true]
        y_pred_literals = [literal_labels[x] for x in y_pred]
        print(metrics.classification_report(y_true_literals, y_pred_literals, digits=3))
    else:
        print(metrics.classification_report(y_true, y_pred, digits=3))
        
    cm = metrics.confusion_matrix(y_true, y_pred, labels=[i for i in range(n_labels)])
    if literal_labels is not None:
        df_cm = pd.DataFrame(cm, index=literal_labels, columns=literal_labels)
    else:
        df_cm = pd.DataFrame(cm, index=[i for i in range(n_labels)], columns=[i for i in range(n_labels)])
    plot = sns.heatmap(df_cm, annot=True, fmt="d", cmap='Blues').get_figure()
    plot.savefig('cm.jpg' if cm_plot_name is None else cm_plot_name)


def plot_multiclass_pr_curve(y_true, logits, literal_labels, pr_plot_name=None):
    onehot_encoder = preprocessing.OneHotEncoder(sparse_output=False)
    integer_encoded = np.reshape(y_true, (len(y_true), 1))
    y_true_cat = onehot_encoder.fit_transform(integer_encoded)

    n_classes = len(y_true_cat[0])
    logits = np.asarray(logits)
    prec = {}
    rec = {}
    for i in range(n_classes):
        prec[i], rec[i], th = metrics.precision_recall_curve(y_true_cat[:, i], logits[:, i])
        plt.plot(rec[i], prec[i], lw=2, label='class {}'.format(literal_labels[i]))
        fscore = (2 * prec[i] * rec[i]) / (prec[i] + rec[i])
        ix = np.nanargmax(fscore)
        if literal_labels is not None:
            print('Class %s: Best Threshold=%f, F-Score=%.3f' % (literal_labels[i], th[ix], fscore[ix]))
        else:
            print('Class %i: Best Threshold=%f, F-Score=%.3f' % (i, th[ix], fscore[ix]))

    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.savefig('pr_curve.png' if pr_plot_name is None else pr_plot_name)
    
    
def plot_binary_pr_curve(y_true, positive_probs, pr_plot_name=None):
    p, r, th = metrics.precision_recall_curve(y_true, positive_probs)
    fscore = (2 * p * r) / (p + r)
    ix = np.nanargmax(fscore)
    fig, ax = plt.subplots()
    ax.plot(r, p, color='purple')
    ax.scatter(r[ix], p[ix], marker='o', color='orange', label='Best')
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    plt.savefig('pr_curve.png' if pr_plot_name is None else pr_plot_name)
    print('Best Threshold=%f, F-Score=%.3f' % (th[ix], fscore[ix]))
