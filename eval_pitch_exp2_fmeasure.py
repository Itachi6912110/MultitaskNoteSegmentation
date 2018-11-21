import torch
from torch import nn
from onoffset_modules import *
from modules_exp2 import *
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import numpy as np
import sys
from argparse import ArgumentParser
import mir_eval
from sklearn.externals import joblib
from sklearn.mixture import GaussianMixture
from statistics import median

#----------------------------
# Smoothing Process
#----------------------------
def Smooth_prediction(predict_notes, threshold):
    Filter = np.ndarray(shape=(5,), dtype=float, buffer=np.array([0.25, 0.5, 1.0, 0.5, 0.25]))
    onset_times = []
    prob_seq = []
    for num in range(predict_notes.shape[1]):
        if num > 1 and num < predict_notes.shape[1]-2:
            prob_seq.append(np.dot(predict_notes[0,num-2:num+3], Filter) / 2.5)
        else:
            prob_seq.append(predict_notes[0][num])

    # find local min, mark 
    if prob_seq[0] > prob_seq[1] and prob_seq[0] > prob_seq[2] and prob_seq[0] > threshold:
        onset_times.append(0.01)
    if prob_seq[1] > prob_seq[0] and prob_seq[1] > prob_seq[2] and prob_seq[1] > prob_seq[3] and prob_seq[1] > threshold:
        onset_times.append(0.03)
    for num in range(len(prob_seq)):
        if num > 1 and num < len(prob_seq)-2:
            if prob_seq[num] > prob_seq[num-1] and prob_seq[num] > prob_seq[num-2] and prob_seq[num] > prob_seq[num+1] and prob_seq[num] > prob_seq[num+2] and prob_seq[num] > threshold:
                onset_times.append(0.02*num+0.01)
    if prob_seq[len(prob_seq)-1] > prob_seq[len(prob_seq)-2] and prob_seq[len(prob_seq)-1] > prob_seq[len(prob_seq)-3] and prob_seq[len(prob_seq)-1] > threshold:
        onset_times.append(0.02*(len(prob_seq)-1)+0.01)
    if prob_seq[len(prob_seq)-2] > prob_seq[len(prob_seq)-1] and prob_seq[len(prob_seq)-2] > prob_seq[len(prob_seq)-3] and prob_seq[len(prob_seq)-2] > prob_seq[len(prob_seq)-4] and prob_seq[len(prob_seq)-2] > threshold:
        onset_times.append(0.02*(len(prob_seq)-2)+0.01)


    prob_seq_np = np.ndarray(shape=(len(prob_seq),), dtype=float, buffer=np.array(prob_seq))

    return np.ndarray(shape=(len(onset_times),), dtype=float, buffer=np.array(onset_times)), prob_seq_np

def Naive_match(onset_times ,offset_times):
    est_intervals = np.zeros((onset_times.shape[0], 2))
    fit_offset_idx = 0
    all_checked = False
    last_onset_matched = 0

    #print(onset_times)
    #print(offset_times)
    #input()

    double_onset_count = 0
    onset_offset_pair_count = 0

    onset_times_sort = np.sort(onset_times, axis=0)
    offset_times_sort = np.sort(offset_times, axis=0)

    for i in range(onset_times.shape[0]):
        # Match for i-th onset time
        if i == onset_times.shape[0] - 1:
            while onset_times_sort[i] >= offset_times_sort[fit_offset_idx]:
                if fit_offset_idx == offset_times.shape[0]-1:
                    all_checked = True
                    break
                else:
                    fit_offset_idx += 1
            if all_checked:
                last_onset_matched = i
                break
            else:
                est_intervals[i][0] = onset_times_sort[i]
                est_intervals[i][1] = offset_times_sort[fit_offset_idx]
        
        else:
            est_intervals[i][0] = onset_times_sort[i]
            if onset_times_sort[i+1] < offset_times_sort[fit_offset_idx]:
                est_intervals[i][1] = onset_times_sort[i+1]
                double_onset_count += 1
            else:
                while onset_times_sort[i] >= offset_times_sort[fit_offset_idx]:
                    if fit_offset_idx == offset_times.shape[0]-1:
                        all_checked = True
                        break
                    else:
                        fit_offset_idx += 1

                if all_checked:
                    last_onset_matched = i
                    break
                else:
                    est_intervals[i][1] = offset_times_sort[fit_offset_idx]
                    fit_offset_idx = fit_offset_idx + 1 if fit_offset_idx < offset_times.shape[0]-1 else offset_times.shape[0]-1
                    onset_offset_pair_count += 1

    if all_checked:
        est_intervals = np.delete(est_intervals, np.s_[last_onset_matched:onset_times.shape[0]], axis=0)

    print('Pair ratio = %.4f' %(onset_offset_pair_count/(onset_offset_pair_count+double_onset_count)))

    return est_intervals

def Naive_pitch(pitch_step, pitch_intervals):
    interval_idx = 0
    pitch_buf = []
    pitch_est = np.zeros((pitch_intervals.shape[0],))
    onset_flag = False

    for i in range(pitch_intervals.shape[0]):
        start_frame = int((pitch_intervals[i][0]-0.01) / 0.02)
        end_frame = int((pitch_intervals[i][1]-0.01) / 0.02)
        pitch_est[i] = np.median(pitch_step[start_frame:end_frame]) if np.median(pitch_step[start_frame:end_frame]) != 0 else 1.0

    #for i in range(pitch_step.shape[0]):
    #    if i * 0.02 > pitch_intervals[interval_idx][0] and not onset_flag:
    #        pitch_buf.append(pitch_step[i])
    #        onset_flag = True
    #    elif i * 0.02 > pitch_intervals[interval_idx][1] and onset_flag:
    #        onset_flag = False
    #        pitch_est[interval_idx] = median(pitch_buf) if median(pitch_buf) != 0 else 1.0
    #        pitch_buf = []
    #        interval_idx += 1
    #        if interval_idx >= pitch_intervals.shape[0]:
    #            break
    #        if i * 0.02 > pitch_intervals[interval_idx][0]:
    #            onset_flag = True
    #            pitch_buf.append(pitch_step[i])

    #if interval_idx != pitch_intervals.shape[0]:
    #print(pitch_est)
    #print(pitch_intervals)
    #print(pitch_step.shape[0]*0.02)
    #input("Stop!!")

    return pitch_est

def HMM_prediction(hmm_model, onoffset_seq_np):
    ON = 1
    OFF = 2
    Z2 = hmm_model.predict(onoffset_seq_np)
    on_buf = []
    off_buf = []
    onset_times = []
    offset_times = []
    on_flag = False
    off_flag = False

    for i in range(Z2.shape[0]):
        if Z2[i] == ON:
            if not on_flag:
                on_flag = True
                on_buf = []
            on_buf.append(i*0.02+0.01)
            
            if off_flag:
                off_flag = False
                offset_times.append(median(off_buf))
                off_buf = []

        elif Z2[i] == OFF:
            if not off_flag:
                off_flag = True
                off_buf = []
            off_buf.append(i*0.02+0.01)
            if on_flag:
                on_flag = False
                onset_times.append(median(on_buf))
                on_buf = []

        else:
            if on_flag:
                on_flag = False
                onset_times.append(median(on_buf))
                on_buf = []
            if off_flag:
                off_flag = False
                offset_times.append(median(off_buf))
                off_buf = []

    onset_np = np.ndarray(shape=(len(onset_times),1), dtype=float, buffer=np.array(onset_times))
    offset_np = np.ndarray(shape=(len(offset_times),1), dtype=float, buffer=np.array(offset_times))
    onoffset_times = Naive_match(onset_np, offset_np)
    #onoffset_times = nearest_match(onset_times, offset_times)
    #if onset_np.shape[0] != offset_np.shape[0]:
    #    print(onset_times)
    #    print(offset_times)
    #    print("Error occurs !!")
    #    input("Stop!!")
    #return np.hstack((onset_np, offset_np))
    return onoffset_times

def nearest_match(on, off):
    buf_on = []
    buf_off = []
    buf_on_wait = 0
    on_in = False
    on_idx = 0
    off_idx = 0

    while on_idx < len(on) and off_idx < len(off):
        if on[on_idx] < off[off_idx]:
            on_in = True
            buf_on_wait = on[on_idx]
            on_idx += 1
            continue
        elif off[off_idx] < on[on_idx]:
            if on_in:
                on_in = False
                if buf_on_wait >= off[off_idx]:
                    print("Error occurs!!")
                    input()
                buf_on.append(buf_on_wait)
                buf_off.append(off[off_idx])
            
            off_idx += 1
            continue
        else:
            print(on[on_idx])
            print(off[off_idx])
            print("Infinite Loop!!")
            input()
    onset_np = np.ndarray(shape=(len(buf_on),1), dtype=float, buffer=np.array(buf_on))
    offset_np = np.ndarray(shape=(len(buf_off),1), dtype=float, buffer=np.array(buf_off))
    return np.hstack((onset_np, offset_np))


def pitch2freq(pitch_np):
    freq_l = [ (2**((pitch_np[i]-69)/12))*440 for i in range(pitch_np.shape[0]) ]
    return np.ndarray(shape=(len(freq_l),), dtype=float, buffer=np.array(freq_l))

def freq2pitch(freq_np):
    pitch_np = 69+12*np.log2(freq_np/440)
    return pitch_np

#----------------------------
# Parser
#----------------------------
parser = ArgumentParser()
#parser.add_argument("pos1", help="positional argument 1")
parser.add_argument("-d", help="data file position", dest="dfile", default="data.npy", type=str)
parser.add_argument("-a", help="label file position", dest="afile", default="ans.npy", type=str)
parser.add_argument("-pf", help="pitch file position", dest="pfile", default="p.npy", type=str)
parser.add_argument("-of", help="output est file position", dest="offile", default="o.npy", type=str)
parser.add_argument("-em1", help="encoder model file position", dest="emfile1", default="model/onset_v3_model", type=str)
#parser.add_argument("-em2", help="encoder model file position", dest="emfile2", default="model/onset_v3_model", type=str)
parser.add_argument("-dm1", help="decoder model file position", dest="dmfile1", default="model/onset_v3_model", type=str)
#parser.add_argument("-dm2", help="decoder model file position", dest="dmfile2", default="model/onset_v3_model", type=str)
parser.add_argument("-ef", help="eval file destination", dest="effile", default="eval/onset_v3_ef.csv", type=str)
parser.add_argument("-tf", help="total eval file destination", dest="tffile", default="eval/onset_v3_tf.csv", type=str)
parser.add_argument("-p", help="present file number", dest="present_file", default=0, type=int)
parser.add_argument("-l", help="learning rate", dest="lr", default=0.001, type=float)
parser.add_argument("--hs1", help="latent space size", dest="hidden_size1", default=50, type=int)
#parser.add_argument("--hs2", help="latent space size", dest="hidden_size2", default=50, type=int)
parser.add_argument("--hl1", help="LSTM layer depth", dest="hidden_layer1", default=3, type=int)
#parser.add_argument("--hl2", help="LSTM layer depth", dest="hidden_layer2", default=3, type=int)
parser.add_argument("--ws", help="input window size", dest="window_size", default=3, type=int)
parser.add_argument("--single-epoch", help="single turn training epoch", dest="single_epoch", default=5, type=int)
parser.add_argument("--bidir1", help="LSTM bidirectional switch", dest="bidir1", default=0, type=bool)
#parser.add_argument("--bidir2", help="LSTM bidirectional switch", dest="bidir2", default=0, type=bool)
parser.add_argument("--norm", help="normalization layer type", choices=["bn","ln","nb","nl","nn","bb", "bl", "lb","ll"], dest="norm_layer", default="nn", type=str)
parser.add_argument("--feat", help="feature cascaded", dest="feat_num", default=1, type=int)
parser.add_argument("--threshold", help="post-processing threshold", dest="threshold", default=0.5, type=float)

args = parser.parse_args()

#----------------------------
# Parameters
#----------------------------
#START_LIST = [1, 3, 5, 7, 9]
START_LIST = [1]
data_file = args.dfile # Z file
ans_file = args.afile  # marked onset/offset/pitch matrix file
p_file = args.pfile  # marked onset/offset/pitch matrix file
of_file = args.offile  # marked onset/offset/pitch matrix file
on_enc_model_file = args.emfile1 # e.g. model_file = "model/offset_v3_bi_k3"
#off_enc_model_file = args.emfile2 # e.g. model_file = "model/offset_v3_bi_k3"
on_dec_model_file = args.dmfile1
#off_dec_model_file = args.dmfile2
INPUT_SIZE = 174*args.feat_num
OUTPUT_SIZE = 4
on_HIDDEN_SIZE = args.hidden_size1
#off_HIDDEN_SIZE = args.hidden_size2
on_HIDDEN_LAYER = args.hidden_layer1
#off_HIDDEN_LAYER = args.hidden_layer2
on_BIDIR = args.bidir1
#off_BIDIR = args.bidir2
LR = args.lr
EPOCH = args.single_epoch
BATCH_SIZE = 10
WINDOW_SIZE = args.window_size
on_NORM_LAYER = args.norm_layer
THRESHOlD = args.threshold
PATIENCE = 700

hmm_file = "model/hmm/TONAS_hmm"

PRESENT_FILE = args.present_file

eval_score_file = args.effile # e.g. 'output/onset_v3_bi_k3_50_ISMIR2014_6m.csv'
total_eval_file = args.tffile # e.g. 'output/onset_v3_bi_k3_50_total_ISMIR2014_6m.csv'

#----------------------------
# Data Collection
#----------------------------
print("Evaluation File ", PRESENT_FILE)
try:
    myfile = open(data_file, 'r')
except IOError:
    print("Could not open file ", data_file)
    print()
    exit()

with open(data_file, 'r') as fd:
    with open(ans_file, 'r') as fa:
        data_np = np.loadtxt(fd)
        data_np = np.transpose(data_np)
        data_np = data_np.reshape((1,-1))
        ans_np = np.loadtxt(fa, delimiter=' ')
        pitch_ans = ans_np[:, 2].reshape((ans_np.shape[0],))
        freq_ans = pitch2freq(pitch_ans)

        ans_np = np.delete(ans_np, 2, axis=1)
        ans_np = ans_np.reshape((ans_np.shape[0],2))
        onset_ans_np = ans_np[:, 0].reshape((ans_np.shape[0],))

        data = torch.from_numpy(data_np).type(torch.FloatTensor).cuda()
        #ans = torch.from_numpy(ans_np).type(torch.FloatTensor).cuda()

with open(p_file, 'r') as fp:
    p_np = np.loadtxt(fp)
    p_np = np.delete(p_np, 0, axis=1)
    p_np = p_np.reshape((p_np.shape[0],))

try:
    with open(total_eval_file, 'r') as ft:
        eval_all = np.loadtxt(ft).reshape((1,-1))
        eval_count = eval_all[0][0]
        eval_all_acc = eval_all[0][1]
        eval_on_all_acc = eval_all[0][2]
        eval_p_all_acc = eval_all[0][3]
    if PRESENT_FILE in START_LIST:
        print("Reinitialize eval file ...")
        eval_count = 0
        eval_all_acc = 0
        eval_on_all_acc = 0
        eval_p_all_acc = 0
except:
    eval_count = 0
    eval_all_acc = 0
    eval_on_all_acc = 0
    eval_p_all_acc = 0

#input & target data
input_loader = data_utils.DataLoader(
    ConcatDataset(data), 
    batch_size=BATCH_SIZE,
    shuffle=False)

#----------------------------
# Model Initialize
#----------------------------
onoffset_hmm = joblib.load(hmm_file)
#####################################
# NOTE CLASSIFIER
#####################################
#note_classifier = Note_Classifier(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, HIDDEN_LAYER, BATCH_SIZE, WINDOW_SIZE, BIDIR)
#note_classifier.load_state_dict(torch.load(model_file))

#####################################
# ATTENTION MODEL
#####################################
if on_NORM_LAYER == "nb" or on_NORM_LAYER == "bb" or on_NORM_LAYER == "lb":
    onEnc = BNEncoder(INPUT_SIZE, on_HIDDEN_SIZE, on_HIDDEN_LAYER, BATCH_SIZE, 2*WINDOW_SIZE+1, on_BIDIR)
elif on_NORM_LAYER == "nl" or on_NORM_LAYER == "bl" or on_NORM_LAYER == "ll":
    onEnc = LNEncoder(INPUT_SIZE, on_HIDDEN_SIZE, on_HIDDEN_LAYER, BATCH_SIZE, 2*WINDOW_SIZE+1, on_BIDIR)
else:
    onEnc = Encoder(INPUT_SIZE, on_HIDDEN_SIZE, on_HIDDEN_LAYER, BATCH_SIZE, 2*WINDOW_SIZE+1, on_BIDIR)

if on_NORM_LAYER == "bn" or on_NORM_LAYER == "bb" or on_NORM_LAYER == "bl":
    onDec = AttentionBNClassifier(INPUT_SIZE, OUTPUT_SIZE, on_HIDDEN_SIZE, on_HIDDEN_LAYER, BATCH_SIZE, 2*WINDOW_SIZE+1, on_BIDIR)
elif on_NORM_LAYER == "ln" or on_NORM_LAYER == "ll" or on_NORM_LAYER == "lb":
    onDec = AttentionLNMultiClassifier(INPUT_SIZE, OUTPUT_SIZE, on_HIDDEN_SIZE, on_HIDDEN_LAYER, BATCH_SIZE, 2*WINDOW_SIZE+1, on_BIDIR)
else:
    onDec = AttentionClassifier(INPUT_SIZE, OUTPUT_SIZE, on_HIDDEN_SIZE, on_HIDDEN_LAYER, BATCH_SIZE, 2*WINDOW_SIZE+1, on_BIDIR)

off_enc_INPUT = on_HIDDEN_SIZE*2 if on_BIDIR else on_HIDDEN_SIZE
#offEnc = Encoder(off_enc_INPUT, off_HIDDEN_SIZE, off_HIDDEN_LAYER, BATCH_SIZE, 2*WINDOW_SIZE+1, off_BIDIR) 
#offDec = AttentionLNPrevClassifier(INPUT_SIZE, OUTPUT_SIZE, off_HIDDEN_SIZE, off_HIDDEN_LAYER, BATCH_SIZE, 2*WINDOW_SIZE+1, on_BIDIR)

onEnc.load_state_dict(torch.load(on_enc_model_file))
onDec.load_state_dict(torch.load(on_dec_model_file))
#offEnc.load_state_dict(torch.load(off_enc_model_file))
#offDec.load_state_dict(torch.load(off_dec_model_file))

#note_classifier.cuda()
onEnc.cuda()
onDec.cuda()
#offEnc.cuda()
#offDec.cuda()

#----------------------------
# Evaluation
#----------------------------
for step, xys in enumerate(input_loader):                 # gives batch data
    b_x = Variable(xys[0].contiguous().view(1, -1, INPUT_SIZE)).cuda() # reshape x to (batch, time_step, input_size)
    #b_y = Variable(xys[1].contiguous().view(BATCH_SIZE, -1, OUTPUT_SIZE)).cuda() # batch y

    #note_hidden = note_classifier.initHidden(BATCH_SIZE)

    predict_on_notes = []
    predict_off_notes = []

    input_time_step = b_x.shape[1]
    k = WINDOW_SIZE
    window_size = 2*k+1

    #for step in range(input_time_step):
    #    if step > k and step < input_time_step - (k+1):
            #####################################
            # NOTE CLASSIFIER
            #####################################
            #note_out, note_hidden = note_classifier(b_x[:, step-k:step+(k+1), :].contiguous().view(BATCH_SIZE, 2*k+1, INPUT_SIZE), note_hidden)
            #predict_note = note_out[:,k,:].view(BATCH_SIZE, 1, OUTPUT_SIZE).data[0][0][1]
            #####################################
    
    for step in range((input_time_step//BATCH_SIZE)+1):
        #####################################
        # Attention CLASSIFIER
        #####################################
        if BATCH_SIZE*step > k and BATCH_SIZE*step < input_time_step - (k+1) - BATCH_SIZE:

            onEncHidden = onEnc.initHidden(BATCH_SIZE)
            #offEncHidden = offEnc.initHidden(BATCH_SIZE)
            
            onEncOuts = torch.zeros(2*k+1, BATCH_SIZE, onEnc.hidden_size*2) if onEnc.bidir else torch.zeros(2*k+1, BATCH_SIZE, onEnc.hidden_size)
            #offEncOuts = torch.zeros(2*k+1, BATCH_SIZE, offEnc.hidden_size*2) if offEnc.bidir else torch.zeros(2*k+1, BATCH_SIZE, offEnc.hidden_size)

            # Onset Encode Step
            for ei in range(window_size):
                enc_out, onEncHidden = onEnc(b_x[:, BATCH_SIZE*step-k+ei:BATCH_SIZE*step-k+ei+BATCH_SIZE, :].contiguous().view(BATCH_SIZE, 1, INPUT_SIZE), onEncHidden)
                onEncOuts[ei] = enc_out.squeeze(1).data

            # To Onset Decoder
            onEncOuts = onEncOuts.transpose(0, 1)

            onDecAttnHidden = torch.cat((onEncHidden[0][2*onEnc.hidden_layer-1], onEncHidden[0][2*onEnc.hidden_layer-2]), 1) if onEnc.bidir else onEncHidden[0][onEnc.hidden_layer-1]
            onEncOuts = Variable(onEncOuts).cuda()

            # 1 step input (cause target only 1 time step)
            onDecOut1, onDecOut2, onDecAttn = onDec(onDecAttnHidden, onEncOuts)

            # To Offset Encoder
            #offEncOuts, offEncHidden = offEnc(onEncOuts, offEncHidden)

            # To Offset Decoder
            #offEncOuts = offEncOuts.transpose(0, 1)
            #offDecAttnHidden = torch.cat((offEncHidden[0][2*offEnc.hidden_layer-1], offEncHidden[0][2*offEnc.hidden_layer-2]), 1) if offEnc.bidir else offEncHidden[0][offEnc.hidden_layer-1]
            #offEncOuts = Variable(offEncOuts).cuda()

            #offDecOut, offDecAttn = offDec(onDecAttnHidden, onEncOuts, onDecOut)
            #offDecOut, offDecAttn = onDec(onDecAttnHidden, onEncOuts)
            
            for i in range(BATCH_SIZE):
                predict_on_note = onDecOut1.view(BATCH_SIZE, 1, OUTPUT_SIZE//2).data[i][0][1]
                predict_off_note = onDecOut2.view(BATCH_SIZE, 1, OUTPUT_SIZE//2).data[i][0][1]
                predict_on_notes.append(predict_on_note)
                predict_off_notes.append(predict_off_note)

            #####################################

            #predict_notes.append(predict_note)
        elif BATCH_SIZE*step <= k:
            for i in range(BATCH_SIZE):
                predict_on_notes.append(0)
                predict_off_notes.append(0)
        elif BATCH_SIZE*step >= input_time_step - (k+1) - BATCH_SIZE:
            for i in range(BATCH_SIZE):
                predict_on_notes.append(0)
                predict_off_notes.append(0)
        else:
            predict_on_notes.append(0)
            predict_off_notes.append(0)

    predict_on_notes_np = np.ndarray(shape=(1,len(predict_on_notes)), dtype=float, buffer=np.array(predict_on_notes))
    predict_off_notes_np = np.ndarray(shape=(1,len(predict_off_notes)), dtype=float, buffer=np.array(predict_off_notes))
    onset_times, probseq_on_np = Smooth_prediction(predict_on_notes_np, THRESHOlD) # list of onset secs, ndarray
    offset_times, probseq_off_np = Smooth_prediction(predict_off_notes_np, THRESHOlD) # list of onset secs, ndarray
    F_on, P_on, R_on = mir_eval.onset.f_measure(onset_ans_np, onset_times, window=0.05)
    pitch_intervals = Naive_match(onset_times ,offset_times)
    freq_est = Naive_pitch(p_np, pitch_intervals)
    pitch_est = freq2pitch(freq_est)
    
    # HMM find times
    #onoffset_times = HMM_prediction(onoffset_hmm, np.hstack((probseq_on_np.reshape((-1,1)), probseq_off_np.reshape((-1,1)))))
    #onset_times = onoffset_times[:,0].reshape((onoffset_times.shape[0],))
    #freq_est = Naive_pitch(p_np, onoffset_times)
    #pitch_est = freq2pitch(freq_est)
    #F_on, P_on, R_on = mir_eval.onset.f_measure(onset_ans_np, onset_times, window=0.05)
    #(P, R, F) =mir_eval.transcription.offset_precision_recall_f1(ans_np, onoffset_times, offset_ratio=0.2, offset_min_tolerance=0.05)
    #P_p, R_p, F_p, AOR = mir_eval.transcription.precision_recall_f1_overlap(ans_np, freq_ans, onoffset_times, freq_est, pitch_tolerance=1000000.0)

    #print(freq_est)
    #print(pitch_intervals)
    #print(freq_ans)
    #print(ans_np)
    #input()
    (P, R, F) = mir_eval.transcription.offset_precision_recall_f1(ans_np, pitch_intervals, offset_ratio=0.2, offset_min_tolerance=0.05)
    P_p, R_p, F_p, AOR = mir_eval.transcription.precision_recall_f1_overlap(ans_np, freq_ans, pitch_intervals, freq_est, pitch_tolerance=50.0)
    all_acc = (eval_all_acc*eval_count + F) / (eval_count+1)
    on_all_acc = (eval_on_all_acc*eval_count + F_on) / (eval_count+1)
    p_all_acc = (eval_p_all_acc*eval_count + F_p) / (eval_count+1)

    print('F-measure: %.4f / %.4f / %.4f' %(F_on, F, F_p))
    print('Precision: %.4f / %.4f / %.4f' %(P_on, P, P_p))
    print('Recall   : %.4f / %.4f / %.4f' %(R_on, R, R_p))
    print('Avg Overlap Ratio: %.4f' %(AOR))
    print('all_score: %.4f / %.4f / %.4f' %(on_all_acc, all_acc, p_all_acc))
    print()

    if PRESENT_FILE in START_LIST:
        with open(eval_score_file, 'w') as fo:
            fo.write("{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}\n".format(F, P, R, F_on, P_on, R_on, F_p, P_p, R_p))
    else:
        with open(eval_score_file, 'a') as fo:
        	fo.write("{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}\n".format(F, P, R, F_on, P_on, R_on, F_p, P_p, R_p))

    with open(total_eval_file, 'w') as fo2:
        fo2.write("{0}\n{1:.5f}\n{2:.5f}\n{3:.5f}\n".format(eval_count+1, all_acc, on_all_acc, p_all_acc))

    #out_est = np.hstack((pitch_intervals, freq_est.reshape((-1,1))))
    #np.savetxt(of_file, out_est)