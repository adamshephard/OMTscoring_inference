from sklearn.metrics import auc, roc_curve, f1_score, precision_recall_curve, average_precision_score, RocCurveDisplay, plot_roc_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt

def plot_metrics(target, prediction):
    fpr, tpr, thresholds = roc_curve(target, prediction)
    roc_auc = auc(fpr, tpr)
    print('roc_auc is:', roc_auc)

    precision, recall, _ = precision_recall_curve(target, prediction)
    average_precision = average_precision_score(target, prediction)
    pr_auc = auc(recall, precision)

    print('Average precision-recall score: {0:0.2f}'.format(
         average_precision))

    plt.figure(figsize=(12, 4))
    lw = 2
    plt.subplot(121)
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve, AUC={0:0.2f}'.format(roc_auc))
    plt.legend(loc="lower right")

    plt.subplot(122)
    plt.step(recall, precision, alpha=0.4, color='darkorange', where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='navy', step='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0, 1.05])
    plt.xlim([0, 1])
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    return
    # plt.savefig(os.path.join(output, 'roc_pr' + set + '.png'))
    # plt.close(plt.gcf())