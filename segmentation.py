#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 11:50:04 2018

Project: EDM segmentation

@author: Zeyu Li
"""

import numpy as np
import librosa
import scipy.signal
import itertools
from scipy.spatial.distance import pdist,squareform

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]

def adaptive_mean(x, N):
    return np.convolve(x, [1.0]*int(N), mode='same')/N

def tempo_beat_tracking(y, sr, hop_size, valid_bpms):
    # Step 1: compute onset curve
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_size, fmax=400, n_mels=8)
    
    # Step 2: half-wave rectify the result
    novelty_mean = adaptive_mean(onset_strength, 16.0)
    novelty_hwr = (onset_strength - novelty_mean).clip(min=0)
    
    # Step 3: then calculate the autocorrelation of this signal
    novelty_autocorr = autocorr(novelty_hwr)  
    
    # Step 4: Sum over constant intervals to detect most likely BPM
    bpm_collection = []
    for bpm in valid_bpms:
        fpb = (60.0 * sr)/(hop_size * bpm)
        frames = (np.round(np.arange(0,np.size(novelty_autocorr), fpb)).astype('int'))[:-1] # Discard last value to prevent reading beyond array (last value rounded up for example)
        bpm_collection.append(np.sum(novelty_autocorr[frames])/np.size(frames))
    
    bpm_max_index = np.argmax(bpm_collection)
    bpm = valid_bpms[bpm_max_index]
    bpm_energy = max(bpm_collection)
    
    # in uncertainty (more than one peak), take the bpm in 2/4 rhythm, not in 3
    bpm_collection[bpm_max_index] = 0
    bpm2 = valid_bpms[np.argmax(bpm_collection)]
    bpm_energy2 = max(bpm_collection)
    if bpm_energy2 > bpm_energy * 0.8:
        if abs(bpm * 3 / 4 - bpm2) < 1 or abs(bpm * 3 / 2 - bpm2) < 1:
            bpm = bpm2
            
    bpm_collection[bpm_max_index] = bpm_energy
    
    fpb = (60.0 * sr)/(hop_size * bpm)
    
    # Step 5: locate (the delay of) first beat
    delay = 0.0
    
    valid_delays = np.arange(0.0, 60.0/bpm, 0.001) # Valid delays in SECONDS
    delay_collection = []
    delay_count = []
    for p in valid_delays:
        # Convert delay from seconds to frames
        delay_frames = (p * sr) / hop_size
        frames = (np.round(np.arange(delay_frames,np.size(novelty_hwr), fpb)).astype('int'))[:-1] # Discard last value to prevent reading beyond array (last value rounded up for example)

        delay_collection.append(np.sum(novelty_hwr[frames])/np.size(frames))
        delay_count.append(np.count_nonzero(novelty_hwr[frames]))
    
    delay = valid_delays[np.argmax(delay_collection)]
    delay_energy = max(delay_collection)
    
    # in uncertainty (more than one prominent peak), take first peak
    delay2 = valid_delays[np.argmax(delay_collection[:len(delay_collection)/4])]
    delay_energy2 = max(delay_collection[:len(delay_collection)/4])
    if delay_energy2 > delay_energy * 0.8:
        delay = delay2
    
    return bpm, delay, bpm_collection, delay_collection

def downbeat_detection(y, sr, beat_locs, beat_size, beat_count, window):
    beat_size_int = int(round(beat_size / 2) * 2)
#    beat_onset = librosa.onset.onset_strength(y, sr=sr, hop_length=beat_size_int, feature=librosa.feature.mfcc, n_mfcc=13)[:beat_count]
    beat_onset = librosa.onset.onset_strength(y, sr=sr, hop_length=beat_size_int, feature=librosa.cqt, fmin=30, n_bins = 20)[:beat_count]
    beat_novelty = [b - a for a, b in zip([0] + list(beat_onset[:-1]), beat_onset)]
    novelty1 = beat_novelty
    
    # downbeat detection
    peak_in_measure = np.array(sorted(range(len(novelty1)), key=lambda x: novelty1[x])[-5:]) % 4
    downbeat = max(set(peak_in_measure), key=list(peak_in_measure).count)
    peak_in_measure = peak_in_measure % 2
    strongbeat = max(set(peak_in_measure), key=list(peak_in_measure).count)
    
    return downbeat, strongbeat, novelty1

def boundary_detection(y, beat_locs, beat_count, beat_size, window):
    beat_size_int = int(beat_size)
    # novelty curve of 16 frames
    beat_in_window = range(window, beat_count - window)
    
    kernel = np.kron(np.eye(2), np.ones((window,window))) - np.kron([[0, 1], [1, 0]], np.ones((window,window)))
    g = scipy.signal.gaussian(2*window, window)
    kernel = np.multiply(kernel, np.multiply.outer(g.T, g))
    
#    melspec = [np.transpose(librosa.feature.melspectrogram(y[i:i+beat_size_int-1], hop_length=beat_size_int, fmax=8000, n_mels=40))[0] for i in beat_locs]
#    melmax = np.max(melspec, axis=1)
#    melmax = melmax / max(melmax)
#    ssm_melmax = squareform(1 - pdist(np.transpose([melmax]), 'euclidean'))
#    novelty_melmax = np.concatenate([np.zeros(window), [window*2 + sum(sum(ssm_melmax[i-window:i+window, i-window:i+window] * kernel)) for i in beat_in_window], np.zeros(window)])
#    
#    # only take energy from low end
#    melmax_low = np.transpose(melspec)[0]
#    melmax_low = melmax_low / max(melmax_low)
#    melmax = list(melmax_low)
#    melmax_diff = [b - a for a, b in zip([0] + melmax[:-1], melmax)]

    mfcc = [np.transpose(librosa.feature.mfcc(y[i:i+beat_size_int-1], hop_length=beat_size_int)[:,0]) for i in beat_locs]
    ssm_mfcc = squareform(1 - pdist(mfcc, 'cosine'))
    novelty_mfcc = [window*2 + sum(sum(ssm_mfcc[i-window:i+window, i-window:i+window] * kernel)) for i in beat_in_window]
    novelty_mfcc = np.concatenate([np.full(window, novelty_mfcc[0]), novelty_mfcc, np.full(window, novelty_mfcc[-1])]) - window / 2
    
    novelty = novelty_mfcc
    
    novelty_thresh = max(novelty) / 16
    
    boundaries = []
    
    for i in beat_in_window:
        if novelty[i] > novelty_thresh \
        and all(n < novelty[i] for n in novelty[i - window/2:i]) \
        and all(n < novelty[i] for n in novelty[i + 1:i + window/2 + 1]) \
        and any(n < novelty[i] / 4 for n in novelty[i - window:i + window + 1]):
            boundaries.append(i)
    
    return boundaries, novelty

# discarded
def boundary_adjust(boundaries, novelty, beat_count, window):
    boundaries_dist = [boundaries[0]] + [b - a for a, b in zip(boundaries, boundaries[1:])]
    # preprocess boundaries before making adjustments
    errors = (np.array([b - a for a, b in zip(boundaries, boundaries[1:])]) + window/2) % window - window/2
    errors = np.array([(boundaries[0] + window/2) % window - window/2] + list(errors) + [beat_count - boundaries[-1]])
    
    adjustments = np.zeros(len(errors))

    adjustments[[i for i, v in enumerate(errors) if v == -window / 2]] = -window / 2
    errors[[i for i, v in enumerate(errors) if v == -window / 2]] = 0
    
    errors = list(errors)
    
    print boundaries_dist
    
    # looking for start position to adjust boundaries
    boundary_start = -1
    for ind in (i for i, v in enumerate(errors) if v==0):
        if errors[ind:ind+2]==[0,0]:
            boundary_start = ind
            break
    
#            if boundary_start == 0 and 0 in errors: boundary_start = errors.index(0)
    if boundary_start == -1: boundary_start = np.argmax(novelty[boundaries])
    
    # adjust boundaries to multiples of window, or slightly more
    order1 = range(boundary_start-1, -1, -1)
    order2 = range(boundary_start+1, len(errors)-1)
    order = [boundary_start] + [x for x in itertools.chain.from_iterable(itertools.izip_longest(order1, order2)) if x >= 0]
    
    print errors
    
    for i in order:
        if errors[i] < 0:
            suborder1 = range(i-1, max(i - len(errors)/2, -1), -1)
            suborder2 = range(i+1, min(i + len(errors)/2, len(errors)))
            if i < boundary_start: suborder1, suborder2 = suborder2, suborder1
            suborder = [x for x in itertools.chain.from_iterable(itertools.izip_longest(suborder1, suborder2)) if x >= 0]
            
            for j in suborder:
                if errors[j] > 0:
                    if errors[i] + errors[j] >= 0:
                        errors[j] = errors[i] + errors[j]
                        errors[i] = 0
                        break
                    else:
                        errors[i] = errors[i] + errors[j]
                        errors[j] = 0
        print "order: " + str(i)
        print errors
    
    # fix boundaries with length < 8 measures
    adj = sum([0 if x > 0 or x < -4 else x for x in errors[1:]])
    errors = [errors[0]] + [x if x > 0 or x < -6 else 0 for x in errors[1:]]
    
    for i in range(len(errors)):
        if errors[i] < -6:
            errors[i] += adj
            adj = 0
            break
    
    errors[-1] += adj
    print errors
    
    errors = np.array(errors) + adjustments
    errors = errors[:-1]
    
    boundaries_dist = ((np.array(boundaries_dist) + window/2) / window * window + errors).astype(int)
    boundaries = np.cumsum(boundaries_dist)
    
    return boundaries
