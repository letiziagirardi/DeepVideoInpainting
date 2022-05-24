import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

##GRAFICO DI DATI MANIPOLATI SOLO CON INPAINTING (modello1) TESTATI SU TECNICA E TECNICA+TCN 
# create a dataset
### GMCNN 00
esperimentoA = [0.92321208, 0.92454357, 0.95364425, 0.9434326,  0.93264997, 0.94269167,
  0.95887847, 0.97545716, 0.97523082, 0.97790345, 0.81791009, 0.89918007,
  0.92790235, 0.95510908, 0.98717011, 0.97550778, 0.91691395, 0.96734918,
  0.95433843, 0.95753518, 0.95441361, 0.95417856, 0.94579162, 0.96565899,
  0.96023068, 0.93383565, 0.95787663, 0.90859438, 0.97045694, 0.920763,
  0.81469091]
espA = sum(esperimentoA)/len(esperimentoA)

esperimentoAtcn = [0.04176615, 0.19625756, 0.28919628, 0.18317649, 0.03747841, 0.05905425,
  0.08149694, 0.12622987, 0.14604983, 0.11665964, 0.02515825, 0.11309957,
  0.2463898,  0.28427674, 0.3093976,  0.12587512, 0.13265999, 0.08601833,
  0.19675839, 0.06051344, 0.09984397, 0.08783302, 0.04548802, 0.1997444,
  0.08562662, 0.20685225, 0.18262401, 0.01829981, 0.1268338,  0.1964319,
  0.20729598]
espAtcn = sum(esperimentoAtcn)/len(esperimentoAtcn)

##OPN
esperimentoC = [0.88217473, 0.87005745, 0.89173556, 0.90260237, 0.79647069, 0.84851175,
  0.91963222, 0.92559717, 0.93295988, 0.96217925, 0.7776795 , 0.82280301,
  0.9689477,  0.97327375, 0.9791703 , 0.96703677, 0.91763712, 0.96474873,
  0.97841826, 0.95875314, 0.94037008 ,0.93257923, 0.93679502, 0.9768857,
  0.94114055, 0.9778622 , 0.95284601 ,0.91238537, 0.94577934, 0.97874652,
  0.97848147]
espC = sum(esperimentoC)/len(esperimentoC)

esperimentoCtcn = [0.04357227, 0.22826638, 0.26170703, 0.17238805 ,0.04577713, 0.05293926,
  0.09214478, 0.16649537, 0.18898027, 0.13303513, 0.0334103 , 0.13504509,
  0.2780924 , 0.2822127  ,0.2848767 , 0.15406954, 0.19673193, 0.1413991,
  0.30876978, 0.09023071, 0.10408766 ,0.08968833, 0.0505246 , 0.25855252,
  0.08101036, 0.26717859, 0.20638073 ,0.01617915, 0.135793  , 0.23805475,
  0.29488263]
espCtcn = sum(esperimentoCtcn)/len(esperimentoCtcn)

##STTN
esperimentoE = [0.91700665, 0.8560739,  0.91847009, 0.93204075, 0.94174494, 0.94151185,
  0.95563683, 0.98000061, 0.98204775, 0.98055702, 0.89922028, 0.90026203,
  0.94016805, 0.94934637, 0.98605056, 0.97757181, 0.92278887, 0.95972525,
  0.98356982, 0.96529435, 0.97470126, 0.97311651, 0.95330822, 0.98421024,
  0.97033253, 0.98386325, 0.96851072, 0.93445436, 0.97809463, 0.98932926,
  0.99039926]
espE = sum(esperimentoE)/len(esperimentoE)

esperimentoEtcn = [0.04071853, 0.19187135, 0.23073659, 0.14994783, 0.03726887, 0.05136917,
  0.07708374, 0.12195042, 0.13996308, 0.10877111, 0.02337106, 0.11044104,
  0.23725588, 0.26136411, 0.28246128, 0.12967801, 0.13962808, 0.09098048,
  0.22762796, 0.06512146, 0.09599662, 0.08568204, 0.04602695, 0.20674584,
  0.07661664, 0.23979748, 0.18318546, 0.01459142, 0.13232721, 0.23525234,
  0.29595701]
espEtcn = sum(esperimentoEtcn)/len(esperimentoEtcn)

#grafico base
names = ['GMCNN_GMCNN', 'GMCNN_GMCNN_tcn', 'OPN_OPN', 'OPN_OPN_tcn','STTN_STTN', 'STTN_STTN_tcn']
values = [ espA, espAtcn, espC, espCtcn, espE, espEtcn]
#subplot(nrows, ncols, index, **kwargs)
#plt.subplot(2,2,2)
plt.bar(names, values, color=['black', 'black', 'green', 'green', 'cyan', 'cyan'])


plt.xticks(rotation='vertical')
plt.suptitle('MODELS 1 tested on inpaining data with techniques and with technique+tcn')


## grafico divisione
data = pd.DataFrame({"name": ["GMCNN_GMCNN", "GMCNN_GMCNN_tcn", "OPN_OPN", "OPN_OPN_tcn", "STTN_STTN", "STTN_STTN_tcn"],
                     "value": [espA, espAtcn, espC, espCtcn, espE, espEtcn],
                     "group": ["NOTCN","TCN", "NOTCN", "TCN","NOTCN", "TCN"]})

data = data.sort_values("name")

map_group_color = {"NOTCN": "green", "TCN": "red"}
#barh per bar orizzontale
ax = data.plot.bar(x="name", y="value", 
                    color=data.group.replace(map_group_color))

legend_handles = [mpatches.Patch(color=color, label=group)
                  for group, color in map_group_color.items()]

ax.legend(handles=legend_handles)

#plt.figure(figsize=(9, 3))
plt.tight_layout()
#show graph
plt.show()
