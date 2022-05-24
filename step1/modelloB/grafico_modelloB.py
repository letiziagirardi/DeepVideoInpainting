import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

##GRAFICO DI DATI MANIPOLATI  CON INPAINTING+TCN (modello2) TESTATI SU TECNICA E TECNICA+TCN 
# create a dataset
### GMCNN 00
esperimentoA = [0.04176615, 0.19625756, 0.28919628, 0.18317649, 0.03747841, 0.05905425,
  0.08149694, 0.12622987, 0.14604983, 0.11665964, 0.02515825, 0.11309957,
  0.2463898,  0.28427674, 0.3093976,  0.12587512, 0.13265999, 0.08601833,
  0.19675839, 0.06051344, 0.09984397, 0.08783302, 0.04548802, 0.1997444,
  0.08562662, 0.20685225, 0.18262401, 0.01829981, 0.1268338,  0.1964319,
  0.20729598]
espA = sum(esperimentoA)/len(esperimentoA)

esperimentoAtcn = [0.07278758, 0.49766008, 0.67409819, 0.67427875, 0.44501433, 0.5341362,
  0.61896535, 0.54384338, 0.74973832, 0.81790898, 0.39762865, 0.61094516,
  0.15067016, 0.36397815, 0.85114598, 0.73330386, 0.59224771, 0.44309831,
  0.64146644, 0.80284276, 0.80283224, 0.84692704, 0.7757159 , 0.75962533,
  0.25502501, 0.76240706, 0.81157395, 0.53934556, 0.67955862, 0.69003555,
  0.76932294]
espAtcn = sum(esperimentoAtcn)/len(esperimentoAtcn)

##OPN
esperimentoC = [0.04357227, 0.22826638, 0.26170703, 0.17238805 ,0.04577713, 0.05293926,
  0.09214478, 0.16649537, 0.18898027, 0.13303513, 0.0334103 , 0.13504509,
  0.2780924 , 0.2822127  ,0.2848767 , 0.15406954, 0.19673193, 0.1413991,
  0.30876978, 0.09023071, 0.10408766 ,0.08968833, 0.0505246 , 0.25855252,
  0.08101036, 0.26717859, 0.20638073 ,0.01617915, 0.135793  , 0.23805475,
  0.29488263]
espC = sum(esperimentoC)/len(esperimentoC)

esperimentoCtcn = [0.30228007 ,0.73613641, 0.78479458, 0.80884745, 0.76450738 ,0.69444648,
  0.73893521, 0.80896811, 0.84567667, 0.85084561, 0.69826523, 0.72228106,
  0.93968561, 0.82518328, 0.78190858, 0.91630439, 0.86512349, 0.94865281,
  0.93409282, 0.90178729, 0.76433671, 0.88503327, 0.85645473, 0.92401267,
  0.78199679, 0.94277824, 0.91136131, 0.80803451, 0.84780921, 0.90273039,
  0.93016398]
espCtcn = sum(esperimentoCtcn)/len(esperimentoCtcn)

##STTN
esperimentoE = [0.04071853, 0.19187135, 0.23073659, 0.14994783, 0.03726887, 0.05136917,
  0.07708374, 0.12195042, 0.13996308, 0.10877111, 0.02337106, 0.11044104,
  0.23725588, 0.26136411, 0.28246128, 0.12967801, 0.13962808, 0.09098048,
  0.22762796, 0.06512146, 0.09599662, 0.08568204, 0.04602695, 0.20674584,
  0.07661664, 0.23979748, 0.18318546, 0.01459142, 0.13232721, 0.23525234,
  0.29595701]
espE = sum(esperimentoE)/len(esperimentoE)

esperimentoEtcn = [0.55256947 ,0.77932753, 0.85450656, 0.7990696,  0.60832902, 0.58109986,
  0.78086798, 0.89557903, 0.90871642 ,0.85068473, 0.74788748, 0.80330862,
  0.87931799, 0.75854388, 0.82345903, 0.9064334,  0.79100124, 0.81191992,
  0.90427234, 0.91067547, 0.89834398, 0.90445444, 0.86168332, 0.96078927,
  0.84843453, 0.94940323, 0.9229956,  0.88537148, 0.88441921, 0.95192524,
  0.93920994]
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

#show graph
plt.tight_layout()
plt.show()
