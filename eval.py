import numpy as np
import functools
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder
from model import LogReg
import torch.nn as nn
import torch as th

def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            print_statistics(statistics, f.__name__)
            return statistics
        return wrapper
    return decorator


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool_)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()


@repeat(10)
def label_classification(embeddings, y, ratio, name, data):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool_)
    X = normalize(X, norm='l2')
    X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=1 - ratio)

    logreg = LogisticRegression(solver='liblinear')  # logistic 回归模型
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)  # 将概率值转换为独热编码的格式

    micro = f1_score(y_test, y_pred, average="micro")  # 微平均
    macro = f1_score(y_test, y_pred, average="macro")  # 宏平均

    return {
        'F1Mi': micro,
        'F1Ma': macro
    }


def evaluation(embeddings, y, name, device, data, learning_rate2, weight_decay2):
    print("=== Evaluation ===")
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()

    citegraph = ['Cora', 'CiteSeer', 'PubMed',]
    cograph = ['DBLP', 'computers', 'photo','CS', 'Physics']
    num_class = {
        'Cora': 7,
        'CiteSeer': 6,
        'PubMed': 3,
        'DBLP': 10,
        'computers': 10,
        'photo': 8,
        'CS': 15,
        'Physics': 5
    }

    train_embs, test_embs, train_labels, test_labels, val_embs, val_labels =None, None, None, None, None, None

    if name in citegraph:
        graph = data
        train_mask = graph.train_mask.cpu()
        val_mask = graph.val_mask.cpu()
        test_mask = graph.test_mask.cpu()

        train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
        val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
        test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

        train_embs = X[train_idx]
        val_embs = X[val_idx]
        test_embs = X[test_idx]

        train_labels = Y[train_idx]
        val_labels = Y[val_idx]
        test_labels = Y[test_idx]

    if name in cograph:
        train_embs, test_embs, train_labels, test_labels = train_test_split(X, Y,
                                                            test_size=0.8)

        train_embs,val_embs,train_labels,val_labels  = train_test_split(train_embs,train_labels,
                                                            test_size=0.5)
    train_embs= th.tensor(train_embs).to(device)
    test_embs = th.tensor(test_embs).to(device)
    val_embs = th.tensor(val_embs).to(device)

    train_labels = th.tensor(train_labels).to(device)
    test_labels = th.tensor(test_labels).to(device)
    val_labels = th.tensor(val_labels).to(device)

    ''' Linear Evaluation '''
    logreg = LogReg(train_embs.shape[1], num_class[name])
    opt = th.optim.Adam(logreg.parameters(), lr=learning_rate2, weight_decay=weight_decay2)

    logreg = logreg.to(device)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0
    eval_acc = 0

    for epoch in range(2000):
        logreg.train()
        opt.zero_grad()
        logits = logreg(train_embs)
        preds = th.argmax(logits, dim=1)
        train_acc = th.sum(preds == train_labels).float() / train_labels.shape[0]
        loss = loss_fn(logits, train_labels)
        loss.backward()
        opt.step()

        logreg.eval()
        with th.no_grad():
            val_logits = logreg(val_embs)
            test_logits = logreg(test_embs)

            val_preds = th.argmax(val_logits, dim=1)
            test_preds = th.argmax(test_logits, dim=1)

            val_acc = th.sum(val_preds == val_labels).float() / val_labels.shape[0]
            test_acc = th.sum(test_preds == test_labels).float() / test_labels.shape[0]
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                if test_acc > eval_acc:
                    eval_acc = test_acc

    print('Linear evaluation accuracy:{:.4f}'.format(eval_acc))
    return eval_acc


