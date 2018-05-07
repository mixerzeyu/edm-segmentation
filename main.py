#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 17:13:23 2018

Project: EDM segmentation

@author: Zeyu Li
"""

import annotation_reader
import segmentation
import csv
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import itertools
from scipy.spatial.distance import pdist,squareform

sr = 44100
hop_size = 512
minBPM = 65
maxBPM = 129.99
stepBPM = 0.01
window = 16
directory = "songs/"
INSERT_BOUNDARIES = True

SHOW_CHARTS = True

csv_precision = open('precision.csv', 'w')
csv_recall = open('recall.csv', 'w')
writer_precision = csv.writer(csv_precision)
writer_recall = csv.writer(csv_recall)

tempo_found_count = 0
beat_found_count = 0
downbeat_found_count = 0

#for filename in os.listdir(directory):
if True:
    filename = "02 Like This.wav"
    if filename.endswith(".wav"):
        print filename
        
        bpm_true = -1
        # read ground truth tags
        if os.path.isfile(directory + filename.replace(".wav", ".mp3")):
            cue_times, bpm_true = annotation_reader.mp3tagreader(directory + filename.replace(".wav", ".mp3"))
        else:
            cue_times = annotation_reader.sdifreader(directory + filename.replace(".wav", ".txt"))
        
#        if len(cue_times) == 0: continue
        
        cue_times = np.array(cue_times)
        
        y, sr = librosa.load(directory + filename, sr)
        
        # beat detection
        D = librosa.stft(y)
        times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr)

        valid_bpms = np.arange(minBPM, maxBPM, stepBPM)
        bpm, attack, bpm_collection, attack_collection = segmentation.tempo_beat_tracking(y, sr, hop_size, valid_bpms)
        
        if SHOW_CHARTS:
            plt.plot(valid_bpms, bpm_collection)
            plt.xlabel('BPM')
            plt.ylabel('Autocorrelation')
            plt.title('Tempo Estimation')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.show()
        
            plt.plot(attack_collection)
            plt.xlabel('Milliseconds')
            plt.ylabel('Average onset strength')
            plt.title('Beat Tracking')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.show()
        
        spb = 60./bpm #seconds per beat
        beat_size = spb * sr
        
        print "Tempo = " + str(bpm)
        print "Beat size = " + str(spb * 1000)
        print "Attack = " + str(attack)
        
        cue_points = [int(round((t / 1000 - attack) / spb)) for t in cue_times]
        
        beat_times = (np.arange(attack, float(len(y))/sr, spb).astype('single'))[:-1]
        beat_count = len(beat_times)
        
        # align the audio to the attack of the first beat (retaining 20 ms prior to the attack)
        y = y[int(max(0, attack - 0.02) * sr):]
        # retrieve the location of every beat
        beat_locs = np.round(np.arange(0, len(y), beat_size)).astype('int')[:-1]

        downbeat, strongbeat, novelty1 = segmentation.downbeat_detection(y, sr, beat_locs, beat_size, beat_count, window)
        
        downbeat_times = np.array(range(downbeat, beat_count, 4)) * spb + attack
        
        print "Downbeat: " + str(downbeat)
        print "Strong Beat: " + str(strongbeat)
        
        boundaries, novelty = segmentation.boundary_detection(y, beat_locs, beat_count, beat_size, window)
        
        # report raw boundaries before adjustment
        boundary_times = np.array(boundaries) * spb + attack
        
        print "Raw boundaries:"
        print boundaries
        print list(boundary_times)
        
        if SHOW_CHARTS:
            print "Downbeat:"
            plt.figure(figsize=(12, 3))
            plt.plot(beat_times, novelty1, label='Novelty')
            (markerline, stemlines, baseline) = plt.stem(downbeat_times, np.ones(len(downbeat_times))*max(novelty1), linefmt='g:', markerfmt=' ', label='Downbeats')
            plt.setp(baseline, visible=False)
            plt.xlabel('Seconds')
            plt.ylabel('Novelty')
            plt.title('Downbeat Detection')
            plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
            plt.show()
            
            print "Raw boundary:"
            plt.figure(figsize=(12, 3))
            plt.plot(beat_times, novelty, label='Novelty')
            (markerline, stemlines, baseline) = plt.stem(boundary_times, np.ones(len(boundary_times))*max(novelty), linefmt='g:', markerfmt=' ', label='Peaks')
            plt.setp(baseline, visible=False)
            (markerline, stemlines, baseline) = plt.stem(cue_times/1000., np.ones(len(cue_times))*max(novelty), linefmt='r:', markerfmt=' ', label='Ground truth')
            plt.setp(baseline, visible=False)
            plt.xlabel('Seconds')
            plt.ylabel('Novelty')
            plt.title('Raw Boundaries')
            plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
            plt.show()
        
#       boundaries = boundary_adjust(boundaries, novelty, beat_count)
        
        # adjust boundaries to downbeat
        boundaries = [int(round((b - downbeat) / 4.) * 4 + downbeat) for b in boundaries]
        
        # insert boundaries if too sparse
        if INSERT_BOUNDARIES:
            boundaries = list(itertools.chain.from_iterable([[a] if b - a < window*3 else list(np.arange(a, b - window, window*2)) for a, b in zip(boundaries, np.append(boundaries[1:], beat_count))]))
            boundaries = list(np.arange(boundaries[0], -1, -window*2)[::-1][:-1]) + boundaries
        
        if boundaries[-1] > beat_count: boundaries = boundaries[:-1]
        
        boundary_times = np.array(boundaries) * spb + attack
        
        print "Adjusted boundaries:"
        print [boundaries[0]] + [b - a for a, b in zip(boundaries, boundaries[1:])]
        print boundaries
        print list(boundary_times)
        
        # report recall
        print "Recall:"
        n = 0
        beat_found = False
        downbeat_found = False
        for t in cue_times:
            n += 1
            dist = min([abs(t / 1000. - t_) for t_ in boundary_times])
            msg = str(t/60000).zfill(2) + ":" + str((t/1000)%60).zfill(2) + "." + str(t%1000).zfill(3) + "|" + str(dist) + "|" 
            if dist <  3:
                msg += "*"
                if dist < 0.5:
                    msg += "*"
                    downbeat_found = True
                    if dist < 0.1:
                        msg += "*"
                        beat_found = True
                csvmsg = filename.replace(".wav","") + \
                    "|Found boundary at " + str(n) + \
                    "|" + msg
                writer_recall.writerow([csvmsg])
            else:
                writer_recall.writerow([filename.replace(".wav","") + \
                    "|Not found boundary at " + str(n)])
            print msg
        
        # report precision
        print "Precision:"
        n=0
        for t in boundary_times:
            n += 1
            dist = min([abs(t - t_ / 1000.) for t_ in cue_times])
            msg = str(int(t/60)).zfill(2) + ":" + str(int(t%60)).zfill(2) + "." + str(int(t)).zfill(3) + "|" + str(dist) + "|" 
            if dist < 3:
                msg += "*"
                if dist < 0.5:
                    msg += "*"
                    if dist < 0.1:
                        msg += "*"
                csvmsg = filename.replace(".wav","") + \
                    "|Found boundary at " + str(n) + \
                    "|" + msg
                writer_precision.writerow([csvmsg])
            else:
                writer_precision.writerow([filename.replace(".wav","") + \
                    "|Not found boundary at " + str(n)])
            print msg
        
        if bpm_true == -1:
            writer_recall.writerow([filename.replace(".wav","") + "|TEMPO = " + str(bpm)])
        elif abs(bpm - bpm_true) > 0.1 and abs(bpm * 2 - bpm_true) > 0.1 and abs(bpm/2 - bpm_true) > 0.1:
            msg = "|**********TEMPO DETECTION FAILED**********"
            print msg
            writer_recall.writerow([filename.replace(".wav","") + msg])
        else:
            tempo_found_count += 1
        
        if beat_found:
            beat_found_count += 1
        elif any((t/1000. - attack) % spb < 0.1 or (t/1000. - attack) % spb > spb - 0.1 for t in cue_times):
            beat_found_count += 1
        else:
            msg = "|**********BEAT DETECTION FAILED**********"
            print msg
            writer_recall.writerow([filename.replace(".wav","") + msg])
        
        if downbeat_found:
            downbeat_found_count += 1
        elif any((t/1000. - attack - spb * downbeat) % (spb * 4) < 0.5 or (t/1000. - attack - spb * downbeat) %  (spb * 4) > spb * 4 - 0.5 for t in cue_times):
            downbeat_found_count += 1
        else:
            msg = "|**********DOWNBEAT DETECTION FAILED**********"
            print msg
            writer_recall.writerow([filename.replace(".wav","") + msg])
        
        if SHOW_CHARTS:
            print "Novelty:"
            plt.figure(figsize=(12, 3))
            plt.plot(beat_times, novelty, label='Novelty')
            (markerline, stemlines, baseline) = plt.stem(boundary_times, np.ones(len(boundary_times))*max(novelty), linefmt='g:', markerfmt=' ', label='Segmentation')
            plt.setp(baseline, visible=False)
            (markerline, stemlines, baseline) = plt.stem(cue_times/1000., np.ones(len(cue_times))*max(novelty), linefmt='r:', markerfmt=' ', label='Ground truth')
            plt.setp(baseline, visible=False)
            plt.xlabel('Seconds')
            plt.ylabel('Novelty')
            plt.title('Adjusted Boundaries')
            plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
            plt.show()

            # plot average energy in obtained segment
            boundaries = sorted(boundaries + cue_points)
            if boundaries[0] < 0: boundaries = boundaries[1:]
            if boundaries[0] > 0: boundaries = np.concatenate([[0], boundaries])
            if boundaries[-1] > beat_count: boundaries = boundaries[:-1]
            if boundaries[-1] < beat_count: boundaries = np.concatenate([boundaries, [beat_count]])
            energy = []
            
            beat_size_int = int(beat_size)
            rms = [librosa.feature.rmse(y[i:i+beat_size_int-1], hop_length=beat_size_int)[0][0] for i in beat_locs]
            ssm_rms = squareform(1 - pdist(np.transpose([rms]), 'euclidean'))
    #        novelty_rms = np.concatenate([np.zeros(window), [window*2 + sum(sum(ssm_rms[i-window:i, i-window:i] + ssm_rms[i:i+window, i:i+window] - ssm_rms[i-window:i, i:i+window] - ssm_rms[i:i+window, i-window:i])) for i in beat_in_window], np.zeros(window)])
    #        novelty_rms = np.concatenate([np.zeros(window), [window*2 + sum(sum(ssm_rms[i-window:i+window, i-window:i+window] * kernel)) for i in beat_in_window], np.zeros(window)])
    #        rms_diff = [b - a for a, b in zip([0] + rms[:-1], rms)]
    
            for i in range(1, len(boundaries)):
                dist = boundaries[i] - boundaries[i-1]
                if dist > 4:
                    energy += list(np.ones(dist) * np.mean(rms[boundaries[i-1] + 2 : boundaries[i] - 2]))
                else:
                    energy += list(np.ones(dist) * np.mean(rms[boundaries[i-1] : boundaries[i]]))
        
            print "Energy:"
            plt.figure(figsize=(12, 3))
            plt.plot(beat_times, energy, label='RMS energy')
            (markerline, stemlines, baseline) = plt.stem(boundary_times, np.ones(len(boundary_times))*max(energy), linefmt='g:', markerfmt=' ', label='Segmentation')
            plt.setp(baseline, visible=False)
            (markerline, stemlines, baseline) = plt.stem(cue_times/1000., np.ones(len(cue_times))*max(energy), linefmt='r:', markerfmt=' ', label='Ground truth')
            plt.setp(baseline, visible=False)
            plt.xlabel('Seconds')
            plt.ylabel('Energy')
            plt.title('Energy in Segmentation')
            plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
            plt.show()
        
        print ""

msg = "Tempo Found:|" + str(tempo_found_count)
print msg
writer_recall.writerow([msg])

msg = "Beat Found:|" + str(beat_found_count)
print msg
writer_recall.writerow([msg])

msg = "Downbeat Found:|" + str(downbeat_found_count)
print msg
writer_recall.writerow([msg])

csv_precision.close()
csv_recall.close()
