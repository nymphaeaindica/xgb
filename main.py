import pandas as pd
import datetime
import os
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import pars
import roc_curve_target as rct

xml_path = 'path.xml'

# settings parsing
global_params, external_params, param_list = pars.parsconf(xml_path)

train, test = pars.parsdata(global_params)

# find best model by GridSearchCV

nfolds = external_params['kfold']
group_kfold = GroupKFold(n_splits=nfolds)
ind = list(group_kfold.split(train['features'], train['labels'], train['target_id']))

########################################################################################################################
import check_kfolds as ck
status = ck.test_kfold(ind, train['target_id'])
########################################################################################################################



dtrain = xgb.DMatrix(train['features'], label=train['labels'])
dtest = xgb.DMatrix(test['features'], label=test['labels'])

results = pd.DataFrame(columns=['n', 'metric_train', 'metric_val', 'num_trees'])
for n, param in enumerate(param_list):
    tmp_bst = xgb.cv(param, dtrain, external_params['num_boost_round'], nfold=nfolds, folds=ind, early_stopping_rounds=external_params['early_stopping_rounds'])
    stop_index = tmp_bst.tail(1).index[0]
    print(stop_index)
    metric_train = tmp_bst.values[stop_index, 0]
    metric_val = tmp_bst.values[stop_index, 2]
    results = results.append({'n': n, 'metric_train': metric_train, 'metric_val': metric_val, 'num_trees': stop_index}, ignore_index=True)

best_val_results = results.sort_values(by='metric_val', ascending=True)[:external_params['best_model_count']]

scores = []
# list of models
models = []
for _, row in best_val_results.iterrows():
    model = xgb.train(params=param_list[int(row[0])], dtrain=dtrain, num_boost_round=int(row['num_trees']))
    models.append(model)
    # predict
    tmp = model.predict(dtest)
    scores.append(tmp)
best_val_results['model'] = models

y_target = rct.y_true_target(test['labels'], test['target_id'])
# plot roc_curves
fig = plt.figure(1)
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')

roc_auc = []

import plotly
import plotly.graph_objs as go
f = go.FigureWidget()
f.layout.hovermode = 'closest'
f.layout.hoverdistance = -1 #ensures no "gaps" for selecting sparse data
default_linewidth = 2
highlighted_linewidth_delta = 10


for score in scores:
    tpr, fpr, thr = rct.roc_target(score, y_target, test['target_id'], external_params['threshold'])
    tmp_roc_auc = auc(fpr, tpr)
    # plt.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.2f)' % tmp_roc_auc)
    trace = go.Scatter(
        x=fpr,
        y=tpr,
        text=list(map(str, thr)),
        hoverinfo='text',
        mode='lines+markers',
        name=str(tmp_roc_auc),
        line={'width': default_linewidth}
    )
    f.add_trace(trace)
    roc_auc.append(tmp_roc_auc)

# our custom event handler
def update_trace(trace, points, selector):
    # this list stores the points which were clicked on
    # in all but one event they it be empty
    if len(points.point_inds) > 0:
        for i in range( len(f.data) ):
            f.data[i]['line']['width'] = default_linewidth + highlighted_linewidth_delta * (i == points.trace_index)


# we need to add the on_click event to each trace separately
for i in range(len(f.data)):
    f.data[i].on_click(update_trace)

unique_url = plotly.offline.plot(f, filename=global_params['working_dir'] + 'roc_curves.html')
# plt.legend(loc="lower right")
# plt.show()
# fig.savefig(global_params['working_dir'] + 'roc_curves.png')



best_val_results['roc_auc'] = roc_auc
best_val_results_sort_by_auc = best_val_results.sort_values(by='roc_auc', ascending=False)

# function for writing log and model
model_ind = 0
for _, current_model in best_val_results_sort_by_auc.iterrows():
    tmp_name = 'xgbmodel_'+str(model_ind)+'_'+str(round(current_model['roc_auc'], 4))+'_'+str(round(current_model['metric_val'], 4))
    current_model['model'].save_model(global_params['working_dir']+tmp_name+'.bin')
    model_ind = model_ind+1

    log_file_name = global_params['working_dir'] + tmp_name + '.txt'
    if os.path.isfile(log_file_name):
        os.remove(log_file_name)

    file = open(log_file_name, "a")
    now = datetime.datetime.now()
    file.write('%s \n' % now)
    param = param_list[int(current_model['n'])]
    for key, val in param.items():
        tmp_str = str(key) + ':    ' + str(val)
        file.write('%s \n' % tmp_str)
    file.write('%s \n' % ('num_trees' + ':    ' + str(current_model['num_trees'])))
    file.write('%s \n' % ('kfold' + ':    ' + str(external_params['kfold'])))
    file.write('%s \n' % ('metric_train' + ':    ' + str(current_model['metric_train'])))
    file.write('%s \n' % ('metric_val' + ':    ' + str(current_model['metric_val'])))
    file.write('%s \n' % ('roc_auc' + ':    ' + str(current_model['roc_auc'])))
    file.close()

# # roc curves by anomaly
# from sklearn import metrics
# # plot roc_curves
# plt.figure(2)
# plt.xlim([-0.02, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC curve')
#
# for score in scores:
#     fpr, tpr, _ = metrics.roc_curve(test['labels'], score)
#     roc_auc = auc(fpr, tpr)
#     plt.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
#
# plt.legend(loc="lower right")
# plt.show()
print('OK')
