import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

#FIGURA 2

#data = np.loadmat('matrix_data.mat')
#print(round(n.mean(data['f1']),8))

#ESPERIMENTO MODELLO b -- TEST 2
f1scoreModel1TCN = [ 0.13917376, 0.13967799, 0.14475879, #PESI GMCNN -- TEST CON GMCNN (0) - OPN (1) - STTN (2)
                    0.15963059, 0.16233794, 0.16177372, #PESI OPN -- TEST CON GMCNN (0) - OPN (1) - STTN (2) 
                    0.1376799, 0.13912854, 0.13967061 #PESI STTN -- TEST CON GMCNN (0) - OPN (1) - STTN (2)
]

## grafico divisione
f1scoreModel1TCN = pd.DataFrame({"name": ["GMCNN_GMCNNtcn", "GMCNN_OPNtcn",  "GMCNN_STTNtcn", "OPN_GMCNNtcn", "OPN_OPNtcn",  "OPN_STTNtcn", "STTN_GMCNNtcn", "STTN_OPNtcn",  "STTN_STTNtcn"],
                     "value": f1scoreModel1TCN,
                     "group": ["GMCNN","GMCNN", "GMCNN","OPN","OPN", "OPN","STTN","STTN", "STTN"]})
f1scoreModel1TCN = f1scoreModel1TCN.sort_values("group")

map_group_color = {"GMCNN": "green", "OPN": "orange","STTN": "lightblue"}
#barh per bar orizzontale
#plt.figure(figsize=(6,4))
#figsize=(10,7), bottom=0.9,
ax = f1scoreModel1TCN.plot.bar(x="name", y="value",  color=f1scoreModel1TCN.group.replace(map_group_color)) 
ax.set_yticks(np.arange(0, 1.0, 0.1))

legend_handles = [mpatches.Patch(color=color, label=group)
                  for group, color in map_group_color.items()]

ax.legend(handles=legend_handles)
plt.tight_layout()
plt.show()
