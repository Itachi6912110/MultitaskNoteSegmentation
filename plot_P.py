import matplotlib.pyplot as plt
import math
import numpy as np
import sys

def pitch2freq(pitch_np):
    freq_l = [ (2**((pitch_np[i]-69)/12))*440 for i in range(pitch_np.shape[0]) ]
    return np.ndarray(shape=(len(freq_l),), dtype=float, buffer=np.array(freq_l))

def linearize(Z, CF, start_f, end_f):
    Z_linear = np.zeros((1, Z.shape[1])) # dummy
    # within start_f to CF[0]
    sfreq = start_f
    efreq = math.ceil(CF[0])
    for k in range(sfreq, efreq, 1):
        d1 = CF[0] - k
        d2 = CF[1] - k
        Zk = (d2*Z[0]-d1*Z[1]) / (d2-d1)
        Zk = Zk.reshape((1, -1))
        Z_linear = np.vstack((Z_linear, Zk))

    # within CF[0] to CF[-1]
    for i in range(CF.shape[0]-1):
        sfreq = math.ceil(CF[i])
        efreq = math.ceil(CF[i+1])
        for k in range(sfreq, efreq, 1):
            d1 = k - CF[i]
            d2 = CF[i+1] - k
            Zk = (Z[i]*d2+Z[i+1]*d1)/(d1+d2)
            Zk = Zk.reshape((1, -1))
            Z_linear = np.vstack((Z_linear, Zk))

    # within CF[-1] to end_f
    sfreq = math.ceil(CF[CF.shape[0]-1])
    efreq = end_f
    for k in range(sfreq, efreq, 1):
        d1 = k - CF[CF.shape[0]-1]
        d2 = k - CF[CF.shape[0]-2]
        Zk = (d2*Z[CF.shape[0]-1]-d1*Z[CF.shape[0]-2]) / (d2-d1)
        Zk = Zk.reshape((1, -1))
        Z_linear = np.vstack((Z_linear, Zk))

    Z_linear = np.delete(Z_linear, 0, 0)
    return Z_linear


pitch_file_name = "pitch/ISMIR2014/"+str(sys.argv[1])+"_P"
est_file_name = "output/est/p_exp2/"+str(sys.argv[1])+"_test"
ref_file_name = "ans/ISMIR2014_ans/"+str(sys.argv[1])+".GroundTruth"
exp_file_name = "output/est/p_exp1/"+str(sys.argv[1])+"_test"
Z_file_name = "data/ISMIR2014_note/"+str(sys.argv[1])+"_Z"
CF_file_name = "data/ISMIR2014_note/"+str(sys.argv[1])+"_CF"
out_file = "img/P_img_"+str(sys.argv[1])+".png"

title_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}
axis_font = {'fontname':'Arial', 'size':'16'}

mksize  = 8
Start_F = 250
End_F   = 500
Start_T = 15
End_T   = 20

#----------------------------
# Data Collection
#----------------------------
print("Ploting File ", pitch_file_name)
try:
    myfile = open(pitch_file_name, 'r')
except IOError:
    print("Could not open file ", pitch_file_name)
    exit()
try:
    myfile = open(est_file_name, 'r')
except IOError:
    print("Could not open file ", est_file_name)
    exit()
try:
    myfile = open(ref_file_name, 'r')
except IOError:
    print("Could not open file ", ref_file_name)
    exit()
try:
    myfile = open(exp_file_name, 'r')
except IOError:
    print("Could not open file ", exp_file_name)
    exit()
try:
    myfile = open(Z_file_name, 'r')
except IOError:
    print("Could not open file ", Z_file_name)
    exit()
try:
    myfile = open(CF_file_name, 'r')
except IOError:
    print("Could not open file ", CF_file_name)
    exit()
#----------------------------
# Plot
#----------------------------
data = np.loadtxt(pitch_file_name)
est = np.loadtxt(est_file_name)
ref = np.loadtxt(ref_file_name)
exp = np.loadtxt(exp_file_name)
Z   = np.loadtxt(Z_file_name)
CF  = np.loadtxt(CF_file_name)[1:] # 175 -> 174
Z   = Z[(CF > Start_F) & ( CF < End_F), :]
CF  = CF[(CF > Start_F) & ( CF < End_F)]

# pitch contour
x = data[:,0].reshape((data.shape[0],))
y = data[:,1].reshape((data.shape[0],))
y[y == 0] = 'nan'
x_filt = x[(x > Start_T) & (x < End_T)]
y_filt = y[(x > Start_T) & (x < End_T)]

# mark est interval
t1 = est[:,0].reshape((est.shape[0],))
t2 = est[:,1].reshape((est.shape[0],))
f  = est[:,2].reshape((est.shape[0],))

# mark ref interval
rt1 = ref[:,0].reshape((ref.shape[0],))
rt2 = ref[:,1].reshape((ref.shape[0],))
rp  = ref[:,2].reshape((ref.shape[0],))
rf  = pitch2freq(rp)

# mark exp interval
et1 = exp[:,0].reshape((exp.shape[0],))
et2 = exp[:,1].reshape((exp.shape[0],))
ef  = exp[:,2].reshape((exp.shape[0],))

# background Z
Z_filt = Z[:, int(Start_T//0.02):int(End_T//0.02)]
Z_linear = linearize(Z_filt, CF, Start_F, End_F)

fig, ax = plt.subplots()
ax.imshow(Z_linear, aspect='auto', cmap='Purples', \
               origin='lower', extent=[Start_T, End_T, Start_F, End_F])

pitch_contour, = ax.plot(x_filt, y_filt, 'b--', label = 'Pitch contour')

for i in range(est.shape[0]):
    x_est = x[(x > t1[i]) & (x < t2[i]) & (t1[i] > Start_T) & (t2[i] < End_T)]
    y_est = np.repeat(f[i], x_est.shape[0])
    if x_est.shape[0] != 0:
        interval1, = ax.plot(x_est, y_est, 'r-', label = 'Multi-task', linewidth=5.0, alpha=0.5)
        on1, = ax.plot(x_est[0], y_est[0], 'ro', label = 'Multi-task onset', markersize=mksize)
        off1, = ax.plot(x_est[-1], y_est[-1], 'rx', label = 'Multi-task offset', markersize=mksize)
for i in range(exp.shape[0]):
    x_est = x[(x > et1[i]) & (x < et2[i]) & (et1[i] > Start_T) & (et2[i] < End_T)]
    y_est = np.repeat(ef[i], x_est.shape[0])
    if x_est.shape[0] != 0:
        interval2, = ax.plot(x_est, y_est, 'y-', label = 'Single-task', linewidth=5.0, alpha=0.5)
        on2, = ax.plot(x_est[0], y_est[0], 'yo', label = 'Single-task onset', markersize=mksize)
        off2, = ax.plot(x_est[-1], y_est[-1], 'yx', label = 'Single-task offset', markersize=mksize)
for i in range(ref.shape[0]):
    x_est = x[(x > rt1[i]) & (x < rt2[i]) & (rt1[i] > Start_T) & (rt2[i] < End_T)]
    y_est = np.repeat(rf[i], x_est.shape[0])
    if x_est.shape[0] != 0:
        interval3, = ax.plot(x_est, y_est, 'g-', label = 'Ground truth', linewidth=5.0, alpha=0.5)
        on3, = ax.plot(x_est[0], y_est[0], 'go', label = 'Ground truth onset', markersize=mksize)
        off3, = ax.plot(x_est[-1], y_est[-1], 'gx', label = 'Ground truth offset', markersize=mksize)
ax.axis([Start_T, End_T, Start_F, End_F])
ax.tick_params(labelsize=14)
plt.xlabel("t (s)", fontsize=16)
plt.ylabel("f (Hz)", fontsize=16)
plt.grid(True, axis='y', alpha=0.7, linestyle='-.')
plt.legend(handles=[pitch_contour, interval1, interval2, interval3], fontsize=14, loc='lower left')
#plt.show(fig)
#plt.xlabel('Time(s)')
#plt.ylabel('Frequency(Hz)')
#plt.grid(True, axis='y')
#plt.show()
#plt.axis([0, plot_y.shape[0], 0, 174])

fig.savefig(out_file, dpi=300)