import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

#FIGURA 1 -- PERFETTA --> MODELL 1 SU TECNICHE GMCNN OPN STTN

#data = np.loadmat('matrix_data.mat')
#print(round(n.mean(data['f1']),8))


#ESPERIMENTO MODELLO a -- TEST 1
f1scoreModel1 = [ 0.94042101, 0.77011578, 0.86985282, #PESI GMCNN -- TEST CON GMCNN (0) - OPN (1) - STTN (2)
            0.73975224, 0.92626648, 0.76626819, #PESI OPN -- TEST CON GMCNN (0) - OPN (1) - STTN (2) 
            0.9241019, 0.85149753, 0.95449703 #PESI STTN -- TEST CON GMCNN (0) - OPN (1) - STTN (2)
]

## grafico divisione
dataModel1 = pd.DataFrame({"name": ["GMCNN_GMCNN", "GMCNN_OPN",  "GMCNN_STTN", "OPN_GMCNN", "OPN_OPN",  "OPN_STTN", "STTN_GMCNN", "STTN_OPN",  "STTN_STTN"],
                     "value": f1scoreModel1,
                     "group": ["GMCNN","GMCNN", "GMCNN", "OPN", "OPN", "OPN", "STTN", "STTN", "STTN"]})
dataModel1 = dataModel1.sort_values("group")

map_group_color = {"GMCNN": "green", "OPN": "orange","STTN": "lightblue"}
#barh per bar orizzontale
#plt.figure(figsize=(6,4))
#figsize=(10,7), bottom=0.9,
ax = dataModel1.plot.bar(x="name", y="value",  color=dataModel1.group.replace(map_group_color)) 
ax.set_yticks(np.arange(0, 1.0, 0.1))

legend_handles = [mpatches.Patch(color=color, label=group)
                  for group, color in map_group_color.items()]

ax.legend(handles=legend_handles)
plt.tight_layout()
plt.show()
