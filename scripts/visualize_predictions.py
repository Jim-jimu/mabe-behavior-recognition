"""Render real prediction events and an optional held-out OOF example.

The demo uses published model predictions. Labels and trajectories are read from
the caller's local competition files and are not copied into the repository.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

BG, INK, MUTED, GRID = '#f7f9fa', '#172b35', '#617681', '#dce5e9'
TEAL, CORAL = '#087f8c', '#d85d50'
COLORS = {'rear':TEAL,'approach':'#3274ad','avoid':'#a569bd',
          'chase':'#df9b35','attack':CORAL,'submit':'#73828b','chaseattack':'#a53548'}


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,
        'text.color':INK,'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':INK,
        'axes.facecolor':BG,'figure.facecolor':BG,'savefig.facecolor':BG,
        'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,
        'axes.spines.bottom':False,'svg.fonttype':'none'})


def segments(frames, active):
    frames=np.asarray(frames,dtype=np.int64)
    active=np.asarray(active,dtype=bool)
    if not len(frames):return []
    starts=np.flatnonzero(np.r_[True,(active[1:]!=active[:-1])|(np.diff(frames)!=1)])
    stops=np.r_[starts[1:],len(frames)]
    return [(int(frames[a]),int(frames[b-1]+1)) for a,b in zip(starts,stops) if active[a]]


def finish(fig,path):
    fig.savefig(path,dpi=200)
    plt.close(fig)


def timeline(events,output,fps=30.,video=438887472):
    events=events[events.video_id==video].copy()
    if events.empty:raise ValueError('No predictions for this video')
    if (events.start_frame>=events.stop_frame).any():raise ValueError('Invalid intervals')
    pairs=sorted(set(zip(events.agent_id,events.target_id)),key=lambda p:(p[0],p[1]!='self',p[1]))
    fig,ax=plt.subplots(figsize=(12,7.4))
    fig.subplots_adjust(left=.17,right=.97,bottom=.18,top=.80)
    fig.text(.055,.935,'A social scene, decoded into events',fontsize=24,weight='bold')
    fig.text(.055,.884,f'Visible test video {video}  /  {len(events):,} predicted intervals  /  {fps:g} fps',color=MUTED,fontsize=13)
    end=float(events.stop_frame.max()/fps)
    for i,(agent,target) in enumerate(pairs):
        subset=events[(events.agent_id==agent)&(events.target_id==target)]
        ax.broken_barh([(0,end)],(i-.31,.62),facecolors='#eaf0f3')
        for action,group in subset.groupby('action'):
            ax.broken_barh(list(zip(group.start_frame/fps,(group.stop_frame-group.start_frame)/fps)),
                          (i-.31,.62),facecolors=COLORS.get(action,MUTED),edgecolors='none')
    labels=[f'{a.replace("mouse","M")} → {b.replace("mouse","M")}' for a,b in pairs]
    ax.set_yticks(range(len(pairs)),labels,fontsize=12)
    ax.set_ylim(len(pairs)-.4,-.7);ax.set_xlim(0,end)
    ax.set_xlabel('Video time (seconds)',fontsize=15,labelpad=12)
    ax.tick_params(axis='both',length=0)
    ax.set_xticks(np.arange(0,end+1,120));ax.grid(axis='x',color=GRID,linewidth=.8)
    ax.set_axisbelow(True)
    actions=[a for a in COLORS if a in set(events.action)]
    fig.legend(handles=[Patch(color=COLORS[a],label=a) for a in actions],
               loc='lower center',bbox_to_anchor=(.54,.055),ncol=6,frameon=False,fontsize=12,
               handlelength=1,handletextpad=.4,columnspacing=1.2)
    fig.text(.055,.022,'Each bar is an exported behavior interval. Self = a single-mouse action.',color=MUTED,fontsize=11)
    finish(fig,output/'prediction-timeline.png')


def oof_plot(path,metadata,output,threshold=.21,video=2054411054,agent='mouse1',target='mouse2',action='sniff'):
    full=pd.read_parquet(path)
    frame=full[(full.video_id==video)&(full.agent_id==agent)&(full.target_id==target)].sort_values('video_frame')
    if frame.empty or frame.fold.nunique()!=1:raise ValueError('Expected one held-out video/mouse pair')
    fps=float(metadata.set_index('video_id').loc[video,'frames_per_second'])
    episodes=segments(frame.video_frame,frame.label==1)
    longest=max(episodes,key=lambda v:v[1]-v[0])
    end=min(frame.video_frame.max()+1,longest[1]+int(30*fps))
    start=max(int(frame.video_frame.min()),int(end-90*fps))
    frame=frame[(frame.video_frame>=start)&(frame.video_frame<end)]
    fig,(prob,truth,pred)=plt.subplots(3,1,figsize=(12,5.8),sharex=True,
        gridspec_kw={'height_ratios':[3.6,.65,.65],'hspace':.20})
    fig.subplots_adjust(left=.16,right=.97,bottom=.15,top=.77)
    fig.text(.055,.93,'Sniffing, frame by frame',fontsize=25,weight='bold')
    fig.text(.055,.867,f'Held-out video {video}  /  {agent} → {target}  /  fold {int(frame.fold.iloc[0])+1} of 5',fontsize=13,color=MUTED)
    time=frame.video_frame.to_numpy()/fps
    probability=frame.prediction.to_numpy()
    prob.plot(time,probability,color=CORAL,lw=1.4)
    prob.fill_between(time,0,probability,color=CORAL,alpha=.12)
    prob.axhline(threshold,color=MUTED,lw=1.2,ls=(0,(5,4)))
    prob.text(end/fps,threshold+.04,f' threshold {threshold:.2f}',ha='right',fontsize=12,color=MUTED)
    prob.set_ylim(0,1.02);prob.set_yticks([0,.5,1]);prob.set_ylabel('OOF probability',fontsize=14,labelpad=16)
    prob.grid(axis='y',color=GRID,lw=.8);prob.tick_params(axis='x',labelbottom=False,length=0)
    for axis,active,color,label in ((truth,frame.label==1,TEAL,'Annotation'),
                                     (pred,frame.prediction>=threshold,CORAL,'Prediction')):
        spans=[(a/fps,(b-a)/fps) for a,b in segments(frame.video_frame,active)]
        axis.broken_barh([(start/fps,(end-start)/fps)],(.1,.8),facecolors='#e4ebef')
        axis.broken_barh(spans,(.1,.8),facecolors=color)
        axis.set_ylim(0,1);axis.set_yticks([.5],[label],fontsize=13);axis.tick_params(length=0)
    truth.tick_params(labelbottom=False)
    pred.set_xlim(start/fps,end/fps);pred.set_xlabel('Video time (seconds)',fontsize=15,labelpad=12)
    finish(fig,output/'oof-event-detail.png')
    provenance={'task':'9/pair/sniff','video_id':video,'agent_id':agent,'target_id':target,
        'fold_zero_based':int(frame.fold.iloc[0]),'fps':fps,'start_frame':start,'stop_frame':int(end),
        'threshold':threshold,'display':'Unsmoothed OOF probabilities and binary threshold decisions',
        'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'selection':'Illustrative 90-second window ending after the longest annotated sniff episode in video 2054411054; not an aggregate benchmark.'}
    (output/'oof-figure-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events',type=Path,default=Path('examples/predicted-events.csv'))
    parser.add_argument('--output',type=Path,default=Path('outputs/figures'))
    parser.add_argument('--oof',type=Path)
    parser.add_argument('--metadata',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);style()
    timeline(pd.read_csv(args.events),args.output)
    if args.oof:
        if not args.metadata:parser.error('--oof requires --metadata train.csv')
        oof_plot(args.oof,pd.read_csv(args.metadata),args.output)
    print('Figures saved to',args.output)


if __name__=='__main__':main()
