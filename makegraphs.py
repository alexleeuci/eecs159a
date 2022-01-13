
# importing the required module
import matplotlib.pyplot as plt
import numpy
# x axis values

#1) hdfs, for pca
y_init = [9.98250783e-001, 1.74914202e-003, 7.52489597e-008, 1.14009914e-032,
 2.40948150e-036, 1.52627642e-042, 1.25206482e-057, 8.67920971e-062,
 7.30113888e-063, 1.38254096e-076, 1.69201113e-079, 1.80565146e-112,
 0.00000000e+000, 0.00000000e+000]

#2) hdfs, for sbd
y_init = [9.98250783e-001, 1.74914202e-003, 7.52489597e-008, 1.14009914e-032,
 2.40948150e-036, 1.52627642e-042, 1.25206482e-057, 8.67920971e-062,
 7.30113888e-063, 1.38254096e-076, 1.69201113e-079, 1.80565146e-112,
 0.00000000e+000, 0.00000000e+000]

#3) hadoop
y_init = [0.41621761, 0.10122298, 0.00970324, 0.03563154, 0.01818739, 0.01625405,
 0.01579565, 0.01571707, 0.01467116, 0.01440812, 0.01391866, 0.01357157,
 0.01267373, 0.01192779, 0.01175774, 0.01157754, 0.01144411, 0.01117119,
 0.01079558, 0.00980421, 0.00916889, 0.0084517,  0.00787562, 0.00784697,
 0.00753625, 0.00742259, 0.00737574, 0.00725329, 0.00713241, 0.0071186,
 0.00694398, 0.00690876, 0.00678273, 0.00625038, 0.00602544, 0.00608898,
 0.00605384, 0.00561494, 0.00521074, 0.00482832, 0.00481649, 0.00471215,
 0.00465291, 0.00459522, 0.00447966, 0.00426091, 0.00406385, 0.00390016,
 0.00377464, 0.00372259, 0.00372259, 0.00372259, 0.00372259, 0.00372259,
 0.00372259, 0.00372259, 0.00372259, 0.00372259, 0.00372259, 0.00362338]

#4)...and now for hadoop, on svd
y_init = [0.41713607, 0.10439713, 0.03877234, 0.0185828,  0.01625762, 0.01579619,
 0.01575152, 0.01467128, 0.01442589, 0.01392249, 0.01368409, 0.012784,
 0.0119288,  0.01180483, 0.01158331, 0.01149308, 0.01118475, 0.01079717,
 0.00985691, 0.00925514, 0.00845913, 0.0079292,  0.00784741, 0.00753629,
 0.00744382, 0.00737619, 0.00725563, 0.00713371, 0.00711915, 0.00701253,
 0.0069437,  0.00688024, 0.00678244, 0.00625031, 0.00608902, 0.00605407,
 0.00561533, 0.00522033, 0.00484598, 0.00481659, 0.0047182,  0.00465415,
 0.00459739, 0.00448325, 0.00434826, 0.00414931, 0.00391412, 0.00378567,
 0.00375689, 0.00372259, 0.00372259, 0.00372259, 0.00372259, 0.00372259,
 0.00372259, 0.00372259, 0.00372259, 0.00372259, 0.00372259, 0.00284413]

y = [sum(y_init[:i]) for i in range(len(y_init))]
print(y)
# corresponding y axis values
x = [i for i in range(len(y))]
 
# plotting the points
plt.plot(x, y)
 
# naming the x axis
plt.xlabel('count of dimensions')
# naming the y axis
plt.ylabel('explained variance')
 
# giving a title to my graph
plt.title('fitness for SVD')
 
# function to show the plot
plt.savefig('books_read.png')