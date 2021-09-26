"""
箱线图
各个方法在某个数据上的表现


notch=True,  # 切口形状，不然就是平常的长方形
patch_artist=True # 自动填色，否则就是白底黄线
vert=True # 垂直摆放

yeast-1
[[0.754581, 0.754059, 0.764524, 0.750118, 0.735006, 0.746090, 0.760388, 0.751824, 0.765660, 0.757013],
[0.737861, 0.729883, 0.706232, 0.814341, 0.751772, 0.665369, 0.765302, 0.764169, 0.713275, 0.751221],
[0.695252, 0.633359, 0.606791, 0.673811, 0.680628, 0.600528, 0.597591, 0.652940, 0.667502, 0.581483],
[0.606620, 0.684704, 0.704641, 0.671809, 0.684759, 0.637738, 0.667936, 0.620148, 0.658407, 0.620602],
[0.827849, 0.800768, 0.782727, 0.779688, 0.812920, 0.767737, 0.815080, 0.832177, 0.799688, 0.787944],
[0.811441, 0.781345, 0.780728, 0.800322, 0.812341, 0.790815, 0.804047, 0.813404, 0.784039, 0.802460],
[0.770877, 0.787257, 0.775696, 0.783117, 0.795276, 0.805815, 0.776954, 0.784127, 0.771353, 0.781326],
[0.801940, 0.759117, 0.788486, 0.795074, 0.766364, 0.751903, 0.762752, 0.746145, 0.799138, 0.769102],
[0.863968, 0.869827, 0.879317, 0.846860, 0.835303, 0.852616, 0.857079, 0.841306, 0.862173, 0.884203]]


yeast-06
[[0.795761, 0.804445, 0.834832, 0.832230, 0.842621, 0.817057, 0.813576, 0.822665, 0.858863, 0.901707],
[0.845482, 0.819911, 0.812059, 0.838333, 0.835235, 0.831748, 0.824679, 0.848598, 0.847727, 0.808102],
[0.649049, 0.766672, 0.751469, 0.693408, 0.719516, 0.731837, 0.698062, 0.701049, 0.693512, 0.745015],
[0.717296, 0.711465, 0.706026, 0.704844, 0.703369, 0.705924, 0.724131, 0.696280, 0.736113, 0.709443],
[0.809125, 0.775982, 0.790029, 0.755985, 0.804012, 0.782902, 0.732109, 0.808610, 0.792772, 0.773870],
[0.815939, 0.792933, 0.737613, 0.777904, 0.813189, 0.812198, 0.713515, 0.811007, 0.839051, 0.781461],
[0.821754, 0.830462, 0.813129, 0.854304, 0.836176, 0.834133, 0.839293, 0.815185, 0.828408, 0.848720],
[0.822508, 0.819018, 0.809486, 0.801291, 0.842612, 0.796451, 0.812062, 0.800042, 0.846854, 0.841142],
[0.883011, 0.848335, 0.829616, 0.840228, 0.852695, 0.837349, 0.844016, 0.880767, 0.837839, 0.854308]]

cleve-1
[[0.527303, 0.425178, 0.590984, 0.636230, 0.471593, 0.378788, 0.462318, 0.392800, 0.588832, 0.669905],
[0.199780, 0.399509, 0.489429, 0.308029, 0.340013, 0.250914, 0.393953, 0.255047, 0.396889, 0.357595],
[0.492827, 0.479820, 0.349415, 0.399446, 0.524706, 0.462464, 0.505199, 0.429970, 0.486643, 0.507025],
[0.468851, 0.440845, 0.433138, 0.563114, 0.539989, 0.401300, 0.489938, 0.503205, 0.496208, 0.546479],
[0.553399, 0.488064, 0.405259, 0.472662, 0.467655, 0.533418, 0.604376, 0.535889, 0.496232, 0.458780],
[0.483570, 0.532268, 0.515108, 0.560882, 0.541685, 0.529815, 0.617482, 0.538729, 0.551445, 0.570986],
[0.516532, 0.474066, 0.484658, 0.552186, 0.478004, 0.455424, 0.554836, 0.409775, 0.569374, 0.625989],
[0.485438, 0.485638, 0.505665, 0.501524, 0.512068, 0.550936, 0.491723, 0.511577, 0.515509, 0.502623],
[0.741850, 0.573611, 0.780077, 0.773667, 0.744854, 0.655432, 0.660551, 0.838244, 0.566275, 0.559588]]

newth-0
[[0.986282, 0.984751, 0.988530, 0.987165, 0.976247, 1.0, 0.983784, 1.0, 0.972388, 0.996830],
[0.959554, 0.979389, 0.950783, 0.976664, 1.0, 0.984435, 0.982821, 0.949108, 1.0, 0.918851],
[0.942970, 0.859637, 0.931667, 0.962535, 1.0, 0.879491, 0.859723, 0.890469, 0.942127, 0.968137],
[0.901082, 1.0, 0.826303, 0.912045, 0.939026, 1.0, 0.866175, 0.938097, 0.939603, 0.950871],
[1.0, 0.953597, 0.961114, 0.932655, 1.0, 0.986622, 0.974047, 0.875964, 0.903184, 0.965790],
[0.983748, 0.974195, 0.972959, 0.993743, 0.988377, 0.969326, 0.983872, 0.990741, 0.996397, 0.981388],
[0.991916, 1.0, 0.996859, 0.980691, 0.998380, 0.978026, 0.971263, 0.992456, 1.0, 1.0],
[0.988406, 0.996557, 1.0, 0.996957, 0.992158, 1.0, 1.0, 0.993346, 0.988126, 0.979077],
[0.999205, 0.992861, 0.991716, 0.999157, 0.999979, 0.999703, 0.993182, 0.998605, 0.997682, 1.0]]


ecoli-4
[[1.000000, 0.863510, 0.818118, 0.920975, 0.971087, 0.859775, 0.853981, 0.911003, 0.929890, 0.914324],
[0.846142, 0.749167, 0.757848, 0.912190, 0.832414, 0.848771, 0.979830, 0.730810, 0.885630, 0.959482],
[0.785365, 0.756522, 0.749598, 0.761302, 0.807315, 0.727137, 0.669945, 0.712186, 0.729271, 0.760018],
[0.681370, 0.749829, 0.653918, 0.673150, 0.664682, 0.732491, 0.772507, 0.632667, 0.641763, 0.753315],
[0.830075, 0.838093, 0.891074, 0.863651, 0.959350, 0.789645, 0.753219, 0.885612, 0.823914, 0.897443],
[0.937793, 0.814224, 0.872883, 0.910974, 0.961434, 0.854294, 0.875181, 0.862607, 0.882092, 0.855290],
[0.817148, 0.903632, 0.853909, 0.884076, 0.677296, 0.838776, 0.958547, 0.885162, 1.000000, 0.788469],
[0.919861, 0.932542, 0.877411, 0.917326, 0.917364, 0.954975, 0.926008, 0.817520, 0.883165, 0.883364],
[0.936622, 0.968051, 0.949041, 0.986700, 0.944925, 0.939909, 0.945201, 0.960462, 0.925620, 0.935075]]


ecoli-2356
[[1.000000, 0.948301, 0.876489, 0.911206, 0.761360, 0.959576, 0.838155, 0.858622, 0.825358, 0.953644],
[0.846604, 0.807386, 0.898627, 0.932848, 0.850000, 1.000000, 1.000000, 0.879725, 1.000000, 0.894467],
[0.870687, 0.819885, 0.815351, 0.814610, 0.808766, 0.799387, 0.817207, 0.777604, 0.810517, 0.831944],
[0.693535, 0.815673, 0.871593, 0.802478, 0.839442, 0.855599, 0.844266, 0.728241, 0.823281, 0.895555],
[0.715081, 0.921865, 0.940918, 0.803585, 0.744492, 0.925725, 0.899742, 0.797246, 0.732803, 0.811503],
[0.885636, 0.883590, 0.981676, 0.878742, 0.862133, 0.949685, 0.886247, 0.905356, 0.941116, 0.955821],
[0.826255, 0.934240, 0.945541, 0.926162, 0.934516, 1.000000, 0.932143, 0.910530, 0.870885, 0.923504],
[0.838534, 1.000000, 0.875945, 0.808991, 0.888844, 0.954539, 0.782660, 0.718030, 0.985479, 0.885128],
[1.000000, 0.904507, 0.863322, 0.936808, 0.945051, 0.848773, 0.909795, 0.924238, 0.826508, 0.970620]]

balance-1
[[0.626252, 0.759916, 0.623871, 0.750296, 0.586120, 0.691968, 0.590134, 0.666811, 0.608012, 0.641749],
[0.507184, 0.569341, 0.640629, 0.433226, 0.607615, 0.644666, 0.748801, 0.522620, 0.707502, 0.687984],
[0.457676, 0.495921, 0.490820, 0.492678, 0.469535, 0.473429, 0.438139, 0.458775, 0.456203, 0.541406],
[0.485762, 0.438008, 0.398555, 0.432416, 0.424252, 0.454339, 0.386644, 0.447320, 0.402608, 0.474994],
[0.481872, 0.435130, 0.500462, 0.456948, 0.448190, 0.455208, 0.442547, 0.487000, 0.489935, 0.481117],
[0.364432, 0.304165, 0.329162, 0.271733, 0.331220, 0.320692, 0.244737, 0.314872, 0.332516, 0.212266],
[0.307301, 0.392412, 0.430547, 0.411685, 0.473235, 0.225230, 0.467976, 0.303412, 0.304939, 0.344085],
[0.567243, 0.515037, 0.655320, 0.701549, 0.705737, 0.591186, 0.580621, 0.633545, 0.531271, 0.657264],
[0.691267, 0.672616, 0.747737, 0.689979, 0.651752, 0.647799, 0.659963, 0.704606, 0.707513, 0.772984]]

yeast-4
[[0.905328, 0.835504, 0.912389, 0.846214, 0.842467, 0.907797, 0.907142, 0.892831, 0.992551, 0.868685],
[0.937856, 0.818664, 0.863018, 0.824141, 0.878250, 0.892911, 0.743830, 0.781869, 0.971067, 0.856399],
[0.865438, 0.865084, 0.856833, 0.749449, 0.864905, 0.903986, 0.782054, 0.890600, 0.931109, 0.845325],
[0.715027, 0.707593, 0.696379, 0.817644, 0.711049, 0.581061, 0.565062, 0.821731, 0.753005, 0.630143],
[0.900838, 0.758744, 0.682152, 0.563360, 0.721371, 0.560962, 0.695202, 0.739605, 0.725764, 0.851847],
[0.927185, 0.862880, 0.928221, 0.851896, 0.918402, 0.917801, 0.865513, 0.956654, 0.854134, 0.852128],
[0.893205, 0.880779, 0.929515, 0.900388, 0.869296, 0.928928, 0.908532, 0.873142, 0.922343, 0.843545],
[0.865190, 0.874105, 0.844127, 0.847445, 0.844391, 0.897847, 0.909123, 0.871417, 0.857314, 0.877112],
[0.944508, 0.938540, 0.929895, 0.959401, 0.892781, 0.894406, 0.902108, 0.845154, 0.930820, 0.858839]]

yeast-6
[[0.918225, 0.910974, 0.929929, 0.874145, 0.882908, 0.936669, 0.935153, 0.872758, 0.887890, 0.864735],
[0.850102, 0.865835, 0.894311, 1.000000, 0.802726, 0.979799, 1.000000, 0.997764, 0.866760, 0.833360],
[0.878261, 0.810890, 0.793107, 0.799175, 0.699307, 0.869139, 0.775100, 0.888447, 0.767971, 0.885977],
[0.712674, 0.821858, 0.881065, 0.711194, 0.838036, 0.648230, 0.850285, 0.899427, 0.777791, 0.788447],
[0.715737, 0.717944, 0.618396, 0.700943, 0.803304, 0.892735, 0.798199, 0.762116, 0.688508, 0.720671],
[0.804110, 0.811657, 0.687770, 0.638134, 0.688482, 0.917938, 0.766742, 0.921701, 1.000000, 0.829662],
[0.961121, 0.932370, 0.992143, 0.926585, 0.983402, 0.933527, 0.871230, 0.890659, 0.894901, 0.956150],
[0.937578, 0.984577, 0.935606, 0.930943, 0.997730, 0.843793, 0.877702, 0.893908, 0.933321, 0.859739],
[0.931959, 1.000000, 0.889288, 1.000000, 0.960182, 0.991841, 1.000000, 0.941257, 0.938330, 0.863678]]

"""

from matplotlib import pyplot as plt
import numpy as np
from matplotlib.axes import Axes

label_font = {}

x_labels = ["R-K", "S-K", "R-D", "S-T", "RF", "AB", "EE", "BB", "HABC"]
title = "AUC box diagram of different methods on different data"
title_font = {'family': 'Times New Roman',
              'weight': 'bold',
              'size': 24,
              }
sub_title = ["yeast-1 IR=2", "yeast-06 IR=4", "cleve-1 IR=5",
             "newth-0 IR=6", "ecoli-4 IR=9", "ecoli-2356 IR=11",
             "balance-1 IR=12", "yeast-4 IR=28", "yeast IR=41"]
sub_title_font = {'family': 'Times New Roman',
                  'weight': 'bold',
                  'size': 16,
                  }
nums = 9
all_data = [
    [[0.754581, 0.754059, 0.764524, 0.750118, 0.735006, 0.746090, 0.760388, 0.751824, 0.765660, 0.757013],
     [0.737861, 0.729883, 0.706232, 0.814341, 0.751772, 0.665369, 0.765302, 0.764169, 0.713275, 0.751221],
     [0.695252, 0.633359, 0.606791, 0.673811, 0.680628, 0.600528, 0.597591, 0.652940, 0.667502, 0.581483],
     [0.606620, 0.684704, 0.704641, 0.671809, 0.684759, 0.637738, 0.667936, 0.620148, 0.658407, 0.620602],
     [0.827849, 0.800768, 0.782727, 0.779688, 0.812920, 0.767737, 0.815080, 0.832177, 0.799688, 0.787944],
     [0.811441, 0.781345, 0.780728, 0.800322, 0.812341, 0.790815, 0.804047, 0.813404, 0.784039, 0.802460],
     [0.770877, 0.787257, 0.775696, 0.783117, 0.795276, 0.805815, 0.776954, 0.784127, 0.771353, 0.781326],
     [0.801940, 0.759117, 0.788486, 0.795074, 0.766364, 0.751903, 0.762752, 0.746145, 0.799138, 0.769102],
     [0.863968, 0.869827, 0.879317, 0.846860, 0.835303, 0.852616, 0.857079, 0.841306, 0.862173, 0.884203]],

    [[0.795761, 0.804445, 0.834832, 0.832230, 0.842621, 0.817057, 0.813576, 0.822665, 0.858863, 0.901707],
     [0.845482, 0.819911, 0.812059, 0.838333, 0.835235, 0.831748, 0.824679, 0.848598, 0.847727, 0.808102],
     [0.649049, 0.766672, 0.751469, 0.693408, 0.719516, 0.731837, 0.698062, 0.701049, 0.693512, 0.745015],
     [0.717296, 0.711465, 0.706026, 0.704844, 0.703369, 0.705924, 0.724131, 0.696280, 0.736113, 0.709443],
     [0.809125, 0.775982, 0.790029, 0.755985, 0.804012, 0.782902, 0.732109, 0.808610, 0.792772, 0.773870],
     [0.815939, 0.792933, 0.737613, 0.777904, 0.813189, 0.812198, 0.713515, 0.811007, 0.839051, 0.781461],
     [0.821754, 0.830462, 0.813129, 0.854304, 0.836176, 0.834133, 0.839293, 0.815185, 0.828408, 0.848720],
     [0.822508, 0.819018, 0.809486, 0.801291, 0.842612, 0.796451, 0.812062, 0.800042, 0.846854, 0.841142],
     [0.883011, 0.848335, 0.829616, 0.840228, 0.852695, 0.837349, 0.844016, 0.880767, 0.837839, 0.854308]],

    [[0.527303, 0.425178, 0.590984, 0.636230, 0.471593, 0.378788, 0.462318, 0.392800, 0.588832, 0.669905],
     [0.199780, 0.399509, 0.489429, 0.308029, 0.340013, 0.250914, 0.393953, 0.255047, 0.396889, 0.357595],
     [0.492827, 0.479820, 0.349415, 0.399446, 0.524706, 0.462464, 0.505199, 0.429970, 0.486643, 0.507025],
     [0.468851, 0.440845, 0.433138, 0.563114, 0.539989, 0.401300, 0.489938, 0.503205, 0.496208, 0.546479],
     [0.553399, 0.488064, 0.405259, 0.472662, 0.467655, 0.533418, 0.604376, 0.535889, 0.496232, 0.458780],
     [0.483570, 0.532268, 0.515108, 0.560882, 0.541685, 0.529815, 0.617482, 0.538729, 0.551445, 0.570986],
     [0.516532, 0.474066, 0.484658, 0.552186, 0.478004, 0.455424, 0.554836, 0.409775, 0.569374, 0.625989],
     [0.485438, 0.485638, 0.505665, 0.501524, 0.512068, 0.550936, 0.491723, 0.511577, 0.515509, 0.502623],
     [0.741850, 0.573611, 0.780077, 0.773667, 0.744854, 0.655432, 0.660551, 0.838244, 0.566275, 0.559588]],

    [[0.986282, 0.984751, 0.988530, 0.987165, 0.976247, 1.0, 0.983784, 1.0, 0.972388, 0.996830],
     [0.959554, 0.979389, 0.950783, 0.976664, 1.0, 0.984435, 0.982821, 0.949108, 1.0, 0.918851],
     [0.942970, 0.859637, 0.931667, 0.962535, 1.0, 0.879491, 0.859723, 0.890469, 0.942127, 0.968137],
     [0.901082, 1.0, 0.826303, 0.912045, 0.939026, 1.0, 0.866175, 0.938097, 0.939603, 0.950871],
     [1.0, 0.953597, 0.961114, 0.932655, 1.0, 0.986622, 0.974047, 0.875964, 0.903184, 0.965790],
     [0.983748, 0.974195, 0.972959, 0.993743, 0.988377, 0.969326, 0.983872, 0.990741, 0.996397, 0.981388],
     [0.991916, 1.0, 0.996859, 0.980691, 0.998380, 0.978026, 0.971263, 0.992456, 1.0, 1.0],
     [0.988406, 0.996557, 1.0, 0.996957, 0.992158, 1.0, 1.0, 0.993346, 0.988126, 0.979077],
     [0.999205, 0.992861, 0.991716, 0.999157, 0.999979, 0.999703, 0.993182, 0.998605, 0.997682, 1.0]],

    [[1.000000, 0.863510, 0.818118, 0.920975, 0.971087, 0.859775, 0.853981, 0.911003, 0.929890, 0.914324],
     [0.846142, 0.749167, 0.757848, 0.912190, 0.832414, 0.848771, 0.979830, 0.730810, 0.885630, 0.959482],
     [0.785365, 0.756522, 0.749598, 0.761302, 0.807315, 0.727137, 0.669945, 0.712186, 0.729271, 0.760018],
     [0.681370, 0.749829, 0.653918, 0.673150, 0.664682, 0.732491, 0.772507, 0.632667, 0.641763, 0.753315],
     [0.830075, 0.838093, 0.891074, 0.863651, 0.959350, 0.789645, 0.753219, 0.885612, 0.823914, 0.897443],
     [0.937793, 0.814224, 0.872883, 0.910974, 0.961434, 0.854294, 0.875181, 0.862607, 0.882092, 0.855290],
     [0.817148, 0.903632, 0.853909, 0.884076, 0.677296, 0.838776, 0.958547, 0.885162, 1.000000, 0.788469],
     [0.919861, 0.932542, 0.877411, 0.917326, 0.917364, 0.954975, 0.926008, 0.817520, 0.883165, 0.883364],
     [0.936622, 0.968051, 0.949041, 0.986700, 0.944925, 0.939909, 0.945201, 0.960462, 0.925620, 0.935075]],

    [[1.000000, 0.948301, 0.876489, 0.911206, 0.761360, 0.959576, 0.838155, 0.858622, 0.825358, 0.953644],
     [0.846604, 0.807386, 0.898627, 0.932848, 0.850000, 1.000000, 1.000000, 0.879725, 1.000000, 0.894467],
     [0.870687, 0.819885, 0.815351, 0.814610, 0.808766, 0.799387, 0.817207, 0.777604, 0.810517, 0.831944],
     [0.693535, 0.815673, 0.871593, 0.802478, 0.839442, 0.855599, 0.844266, 0.728241, 0.823281, 0.895555],
     [0.715081, 0.921865, 0.940918, 0.803585, 0.744492, 0.925725, 0.899742, 0.797246, 0.732803, 0.811503],
     [0.885636, 0.883590, 0.981676, 0.878742, 0.862133, 0.949685, 0.886247, 0.905356, 0.941116, 0.955821],
     [0.826255, 0.934240, 0.945541, 0.926162, 0.934516, 1.000000, 0.932143, 0.910530, 0.870885, 0.923504],
     [0.838534, 1.000000, 0.875945, 0.808991, 0.888844, 0.954539, 0.782660, 0.718030, 0.985479, 0.885128],
     [1.000000, 0.904507, 0.863322, 0.936808, 0.945051, 0.848773, 0.909795, 0.924238, 0.826508, 0.970620]],

    [[0.626252, 0.759916, 0.623871, 0.750296, 0.586120, 0.691968, 0.590134, 0.666811, 0.608012, 0.641749],
     [0.507184, 0.569341, 0.640629, 0.433226, 0.607615, 0.644666, 0.748801, 0.522620, 0.707502, 0.687984],
     [0.457676, 0.495921, 0.490820, 0.492678, 0.469535, 0.473429, 0.438139, 0.458775, 0.456203, 0.541406],
     [0.485762, 0.438008, 0.398555, 0.432416, 0.424252, 0.454339, 0.386644, 0.447320, 0.402608, 0.474994],
     [0.481872, 0.435130, 0.500462, 0.456948, 0.448190, 0.455208, 0.442547, 0.487000, 0.489935, 0.481117],
     [0.364432, 0.304165, 0.329162, 0.271733, 0.331220, 0.320692, 0.244737, 0.314872, 0.332516, 0.212266],
     [0.307301, 0.392412, 0.430547, 0.411685, 0.473235, 0.225230, 0.467976, 0.303412, 0.304939, 0.344085],
     [0.567243, 0.515037, 0.655320, 0.701549, 0.705737, 0.591186, 0.580621, 0.633545, 0.531271, 0.657264],
     [0.691267, 0.672616, 0.747737, 0.689979, 0.651752, 0.647799, 0.659963, 0.704606, 0.707513, 0.772984]],

    [[0.905328, 0.835504, 0.912389, 0.846214, 0.842467, 0.907797, 0.907142, 0.892831, 0.992551, 0.868685],
     [0.937856, 0.818664, 0.863018, 0.824141, 0.878250, 0.892911, 0.743830, 0.781869, 0.971067, 0.856399],
     [0.865438, 0.865084, 0.856833, 0.749449, 0.864905, 0.903986, 0.782054, 0.890600, 0.931109, 0.845325],
     [0.715027, 0.707593, 0.696379, 0.817644, 0.711049, 0.581061, 0.565062, 0.821731, 0.753005, 0.630143],
     [0.900838, 0.758744, 0.682152, 0.563360, 0.721371, 0.560962, 0.695202, 0.739605, 0.725764, 0.851847],
     [0.927185, 0.862880, 0.928221, 0.851896, 0.918402, 0.917801, 0.865513, 0.956654, 0.854134, 0.852128],
     [0.893205, 0.880779, 0.929515, 0.900388, 0.869296, 0.928928, 0.908532, 0.873142, 0.922343, 0.843545],
     [0.865190, 0.874105, 0.844127, 0.847445, 0.844391, 0.897847, 0.909123, 0.871417, 0.857314, 0.877112],
     [0.944508, 0.938540, 0.929895, 0.959401, 0.892781, 0.894406, 0.902108, 0.845154, 0.930820, 0.858839]],

    [[0.918225, 0.910974, 0.929929, 0.874145, 0.882908, 0.936669, 0.935153, 0.872758, 0.887890, 0.864735],
     [0.850102, 0.865835, 0.894311, 1.000000, 0.802726, 0.979799, 1.000000, 0.997764, 0.866760, 0.833360],
     [0.878261, 0.810890, 0.793107, 0.799175, 0.699307, 0.869139, 0.775100, 0.888447, 0.767971, 0.885977],
     [0.712674, 0.821858, 0.881065, 0.711194, 0.838036, 0.648230, 0.850285, 0.899427, 0.777791, 0.788447],
     [0.715737, 0.717944, 0.618396, 0.700943, 0.803304, 0.892735, 0.798199, 0.762116, 0.688508, 0.720671],
     [0.804110, 0.811657, 0.687770, 0.638134, 0.688482, 0.917938, 0.766742, 0.921701, 1.000000, 0.829662],
     [0.961121, 0.932370, 0.992143, 0.926585, 0.983402, 0.933527, 0.871230, 0.890659, 0.894901, 0.956150],
     [0.937578, 0.984577, 0.935606, 0.930943, 0.997730, 0.843793, 0.877702, 0.893908, 0.933321, 0.859739],
     [0.931959, 1.000000, 0.889288, 1.000000, 0.960182, 0.991841, 1.000000, 0.941257, 0.938330, 0.863678]]
]
# 首先有图（fig），然后有轴（ax）
nrows = 3
ncols = 3

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 9))
# fig.suptitle(title)
# 子图间距 wspace为水平间隔，hspace为垂直间隔
fig.subplots_adjust(wspace=0.2, hspace=0.4)
for i in range(nrows):
    for j in range(ncols):
        bplot = axes[i][j].boxplot(all_data[i * nrows + j], )
        axes[i][j].set_title(sub_title[i * nrows + j], fontdict=sub_title_font)

plt.setp(axes, xticks=[1, 2, 3, 4, 5, 6, 7, 8, 9],
         xticklabels=x_labels)


save_name = "box_line_AUC_compare"
plt.savefig("./png_img/" + save_name + ".png", dpi=300, bbox_inches='tight')
plt.savefig("./eps_img/" + save_name + ".eps", dpi=300, bbox_inches='tight')

plt.show()
