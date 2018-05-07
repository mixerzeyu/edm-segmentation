#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 11:48:33 2018

@author: Zeyu Li
"""

import eyed3

def tagdecode(c):
    i = ord(c)
    if c == '+':
        return 62
    elif c == '/':
        return 63
    elif i >= ord('0') and i <= ord('9'):
        return i - ord('0') + 52
    elif i >= ord('A') and i <= ord('Z'):
        return i - ord('A')
    elif i >= ord('a') and i <= ord('z'):
        return i - ord('a') + 26
    else:
        return -1

def mp3tagreader(path):
    mp3 = eyed3.load(path)
    bpm_true = mp3.tag.bpm
    
    for o in mp3.tag.objects:
        if o.description == "Serato Markers2" and "application/octet" in o.data:
            tag = o.data.replace("\n", "")
            break
    
    cue_times = []
    for subs in ["AAAAANAAA", "AAAAANAAE", "AAAAANAAI", "AAAAANAAM", "AAAAANAAQ", "AAAAANAAU", "AAAAANAAY", "AAAAANAAc"]:
        try:
            if subs not in tag: continue
            i = tag.index(subs)
            t = tagdecode(tag[i + 13]) + tagdecode(tag[i + 12]) * 64 + tagdecode(tag[i + 11]) * 64 * 64 + tagdecode(tag[i + 10]) * 64 * 64 * 64
            cue_times.append(t)
        except:
            break
    
    return cue_times, bpm_true

def sdifreader(path):
    cue_times = []
    
    fh = open(path)
    for line in fh:
        if line[:4] == "1MRK":
            cue_times.append(int(float(line.split()[-1]) * 1000))
    
    return cue_times
