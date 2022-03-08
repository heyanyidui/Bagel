from model import DonutX
import pandas as pd
import numpy as np
from kpi_series import KPISeries
from sklearn.metrics import precision_recall_curve
from evaluation_metric import range_lift_with_delay

KPIs = ['0efb375b-b902-3661-ab23-9a0bb799f4e3', 'e0747cad-8dc8-38a9-a9ab-855b61f5551d',
        '1c6d7a26-1f1a-3321-bb4d-7a9d969ec8f0', '9c639a46-34c8-39bc-aaf0-9144b37adfc8']

# 1. 确定 KPI
df1 = pd.read_csv('data/phase2_train.csv', index_col=None)
df2 = pd.read_hdf('data/phase2_ground_truth.hdf', index_col=None)

df2['KPI ID'] = df2['KPI ID'].astype('str')

df1 = df1[df1['KPI ID'] == KPIs[0]]
df2 = df2[df2['KPI ID'] == KPIs[0]]

# 合并
df = df1.append(df2, ignore_index=True)

kpi = KPISeries(
    value=df.value,
    timestamp=df.timestamp,
    label=df.label,
    name='sample_data',
)

train_kpi, valid_kpi, test_kpi = kpi.split((0.49, 0.21, 0.3))

train_kpi, train_kpi_mean, train_kpi_std = train_kpi.normalize(return_statistic=True)
valid_kpi = valid_kpi.normalize(mean=train_kpi_mean, std=train_kpi_std)
test_kpi = test_kpi.normalize(mean=train_kpi_mean, std=train_kpi_std)

model = DonutX(cuda=False, max_epoch=100, latent_dims=8, network_size=[100, 100])
model.fit(train_kpi.label_sampling(0.), valid_kpi)
y_prob = model.predict(test_kpi.label_sampling(0.))
y_prob = range_lift_with_delay(y_prob, test_kpi.label)
precisions, recalls, thresholds = precision_recall_curve(test_kpi.label, y_prob)
f1_scores = (2 * precisions * recalls) / (precisions + recalls)
print(f'best F1-score: {np.max(f1_scores[np.isfinite(f1_scores)])}')