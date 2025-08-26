"""
26-August-2025 KB
Plot EPHYs data from sessions with probes targeting Dorsal Raphe (DR)
2 sessions
    A. 149 clusters
    B. 232 clusters
"""
#%% 
#====================================================================================================

#================================= 1. for DR for 1 session (total 2) ================================

#====================================================================================================
from iblatlas.atlas import AllenAtlas
import numpy as np 
import matplotlib.pyplot as plt
from one.api import ONE
one = ONE()

# Search for sessions with probes targeting Dorsal Raphe (DR)
eids = one.search(
    task_protocol='ephys',
    atlas_acronym='DR',
    project='brainwide'
)

print("Sessions with ephys in DR:", eids)




eid = eids[0]


PRE_TIME, POST_TIME = 1, 2  # peri-event window

# ----------------------
# LOAD DATA
# ----------------------
spikes = one.load_object(eid, 'spikes')
clusters = one.load_object(eid, 'clusters')
channels = one.load_object(eid, 'channels')
trials = one.load_object(eid, 'trials')



print(clusters.keys())

print(channels.keys())



cluster_channels = clusters['channels']  # channel index for each cluster
cluster_region_ids = channels['brainLocationIds_ccf_2017'][cluster_channels]


ba = AllenAtlas(25)  # 25 µm resolution atlas

# Map IDs to acronyms
cluster_acronyms = np.array([ba.regions.id2acronym(rid) for rid in cluster_region_ids])

# Filter DR clusters
dr_clusters = np.where(cluster_acronyms == 'DR')[0]
print(f"Found {len(dr_clusters)} DR clusters")


def compute_psth(spike_times, event_times, pre_time=1, post_time=2, bin_size=0.01):
    """
    Compute peri-event time histogram (PSTH).
    
    spike_times: 1D array of spike times (for one neuron)
    event_times: array of event timestamps
    pre_time, post_time: window in seconds relative to event
    bin_size: bin width in seconds
    """
    bins = np.arange(-pre_time, post_time + bin_size, bin_size)
    counts = np.zeros(len(bins) - 1)

    for et in event_times:
        aligned = spike_times - et
        hist, _ = np.histogram(aligned, bins=bins)
        counts += hist

    # Convert counts → firing rate (Hz)
    counts = counts / (len(event_times) * bin_size)
    return counts, bins[:-1] + bin_size/2  # return bin centers


EVENT = 'stimOnTrigger_times' #"feedback_times"


PRE_TIME, POST_TIME = 1, 2
BIN_SIZE = 0.01

all_peths = []
for clu in dr_clusters:
    st = spikes.times[spikes.clusters == clu]
    counts, t = compute_psth(st, trials[EVENT],
                             pre_time=PRE_TIME, post_time=POST_TIME, bin_size=BIN_SIZE)
    all_peths.append(counts)

all_peths = np.vstack(all_peths)
mean_peth = all_peths.mean(axis=0)

plt.figure(figsize=(8, 4))
plt.plot(t, mean_peth, color='k')
plt.axvline(0, color='red', linestyle='--', label=EVENT)
plt.xlabel('Time from stimOn (s)')
plt.ylabel('Firing rate (Hz)')
plt.title(f'Dorsal Raphe (DR) population response, n={len(dr_clusters)} clusters')
plt.legend()
plt.tight_layout()
plt.show() 







#%% 
#========================================================================================================

#================================= 2. for VTA per session ================================

#======================================================================================================== 
# %%
# %%
session_peths = []

for eid in eids:
    print(f"\nProcessing session: {eid}")
    try:
        # Load session data
        spikes = one.load_object(eid, 'spikes')
        clusters = one.load_object(eid, 'clusters')
        channels = one.load_object(eid, 'channels')
        trials = one.load_object(eid, 'trials')
    except Exception as e:
        print(f"Skipping session {eid} due to error: {e}")
        continue  # go to next session
    
    # Map clusters to regions
    cluster_channels = clusters['channels']
    cluster_region_ids = channels['brainLocationIds_ccf_2017'][cluster_channels]
    cluster_acronyms = np.array([ba.regions.id2acronym(rid) for rid in cluster_region_ids])
    
    # Filter VTA clusters
    vta_clusters = np.where(cluster_acronyms == 'VTA')[0]
    print(f"Found {len(vta_clusters)} VTA clusters")
    
    if len(vta_clusters) == 0:
        continue
    
    # Compute PSTH per cluster
    all_peths = []
    for clu in vta_clusters:
        st = spikes.times[spikes.clusters == clu]
        counts, t = compute_psth(st, trials[EVENT],
                                 pre_time=PRE_TIME, post_time=POST_TIME, bin_size=BIN_SIZE)
        all_peths.append(counts)
    
    all_peths = np.vstack(all_peths)
    mean_peth = all_peths.mean(axis=0)
    
    session_peths.append(mean_peth)
    
    # Plot per session
    plt.figure(figsize=(6, 3))
    plt.plot(t, mean_peth, color='k')
    plt.axvline(0, color='red', linestyle='--', label=EVENT)
    plt.xlabel('Time from stimOn (s)')
    plt.ylabel('Firing rate (Hz)')
    plt.title(f'VTA response, session {eid}, n={len(vta_clusters)} clusters')
    plt.legend()
    plt.tight_layout()
    plt.show()





#%%
"""
plot split by contrast
"""

from iblatlas.atlas import AllenAtlas
import numpy as np
import matplotlib.pyplot as plt
from one.api import ONE
from matplotlib import cm

one = ONE()

# ----------------------
# SEARCH SESSIONS WITH VTA
# ----------------------
eids = one.search(task_protocol='ephys', atlas_acronym='VTA', project='brainwide')
print(f"Found {len(eids)} sessions with VTA: {eids}")

# ----------------------
# PARAMETERS
# ----------------------
PRE_TIME, POST_TIME = 1, 2
BIN_SIZE = 0.01
EVENTS = ['stimOnTrigger_times', 'feedback_times']

ba = AllenAtlas(25)  # 25µm resolution

# ----------------------
# FUNCTION TO COMPUTE PSTH
# ----------------------
def compute_psth(spike_times, event_times, pre_time=1, post_time=2, bin_size=0.01):
    bins = np.arange(-pre_time, post_time + bin_size, bin_size)
    counts = np.zeros(len(bins) - 1)
    for et in event_times:
        aligned = spike_times - et
        hist, _ = np.histogram(aligned, bins=bins)
        counts += hist
    counts = counts / (len(event_times) * bin_size)  # firing rate in Hz
    return counts, bins[:-1] + bin_size / 2

# ----------------------
# LOAD ALL SESSIONS
# ----------------------
all_peths = {event: [] for event in EVENTS}

for eid in eids:
    try:
        print(f"Processing session: {eid}")
        spikes = one.load_object(eid, 'spikes')
        clusters = one.load_object(eid, 'clusters')
        channels = one.load_object(eid, 'channels')
        trials = one.load_object(eid, 'trials')

        cluster_channels = clusters['channels']
        cluster_region_ids = channels['brainLocationIds_ccf_2017'][cluster_channels]
        cluster_acronyms = np.array([ba.regions.id2acronym(rid) for rid in cluster_region_ids])
        vta_clusters = np.where(cluster_acronyms == 'VTA')[0]

        if len(vta_clusters) == 0:
            continue

        # Merge contrastLeft and contrastRight
        contrast_values = []
        for l, r in zip(trials['contrastLeft'], trials['contrastRight']):
            if not np.isnan(l):
                contrast_values.append(float(l))
            elif not np.isnan(r):
                contrast_values.append(float(r))
            else:
                contrast_values.append(np.nan)
        contrast_values = np.array(contrast_values)
        unique_contrasts = np.unique(contrast_values[~np.isnan(contrast_values)])

        for event in EVENTS:
            for contrast in unique_contrasts:
                try:
                    trial_mask = contrast_values == contrast
                    if trial_mask.sum() == 0:
                        continue

                    session_counts = []
                    for clu in vta_clusters:
                        st = spikes.times[spikes.clusters == clu]
                        counts, t = compute_psth(st, trials[event][trial_mask],
                                                 pre_time=PRE_TIME, post_time=POST_TIME, bin_size=BIN_SIZE)
                        session_counts.append(counts)

                    all_peths[event].append({'eid': str(eid), 'contrast': contrast, 'counts': np.vstack(session_counts)})
                except Exception as e:
                    print(f"Skipping cluster {clu} in session {eid} for {event} contrast {contrast} due to {e}")
                    continue

    except Exception as e:
        print(f"Skipping session {eid} due to error: {e}")
        continue


# ----------------------
# PLOT ALL SESSIONS BY CONTRAST
# ----------------------
for event in EVENTS:
    plt.figure(figsize=(10, 5))
    contrasts = np.unique([d['contrast'] for d in all_peths[event]])
    colors = cm.Greys(np.linspace(0.3, 1.0, len(contrasts)))  # 0.3 = light gray, 1.0 = black
    
    for i, contrast in enumerate(contrasts):
        try:
            contrast_counts = [d['counts'].mean(axis=0) for d in all_peths[event] if d['contrast'] == contrast]
            if len(contrast_counts) == 0:
                continue
            mean_counts = np.vstack(contrast_counts).mean(axis=0)
            plt.plot(t, mean_counts, label=f"Contrast {contrast}", color=colors[i])
        except Exception as e:
            print(f"Skipping plotting contrast {contrast} due to {e}")
            continue

    plt.axvline(0, color='red', linestyle='--', label=event)
    plt.xlabel('Time from event (s)')
    plt.ylabel('Firing rate (Hz)')
    plt.title(f'VTA population response - {event}')
    plt.legend()
    plt.tight_layout()
    plt.show()







#%% 
"""
plot feedbackType at 2 events: stimOnTrigger_times and feedback_times
""" 
import numpy as np

feedback = np.array(trials.feedbackType)  # shape (n_trials,)
mask_correct = feedback == 1
mask_incorrect = feedback == -1

# Now apply the mask to each field you care about
correct_trials = {k: v[mask_correct] for k, v in trials.items()}
incorrect_trials = {k: v[mask_incorrect] for k, v in trials.items()}

# Then compute PSTHs separately
# Example:
for event in ['stimOnTrigger_times', 'feedback_times']:
    
    # Correct trials
    all_counts_correct = []
    for clu in vta_clusters:
        st = spikes.times[spikes.clusters == clu]
        counts, t = compute_psth(st, correct_trials[event], pre_time=PRE_TIME, post_time=POST_TIME, bin_size=BIN_SIZE)
        all_counts_correct.append(counts)
    mean_correct = np.vstack(all_counts_correct).mean(axis=0)
    
    plt.figure(figsize=(8,4))
    plt.plot(t, mean_correct, label='Correct')
    plt.axvline(0, color='red', linestyle='--')
    plt.title(f'{event} - Correct trials')
    plt.show()

    # Incorrect trials
    all_counts_incorrect = []
    for clu in vta_clusters:
        st = spikes.times[spikes.clusters == clu]
        counts, t = compute_psth(st, incorrect_trials[event], pre_time=PRE_TIME, post_time=POST_TIME, bin_size=BIN_SIZE)
        all_counts_incorrect.append(counts)
    mean_incorrect = np.vstack(all_counts_incorrect).mean(axis=0)
    
    plt.figure(figsize=(8,4))
    plt.plot(t, mean_incorrect, label='Incorrect', color='orange')
    plt.axvline(0, color='red', linestyle='--')
    plt.title(f'{event} - Incorrect trials')
    plt.show()









# %%
"""
ALL, BOTH FROM ABOVE
""" 
#======================
# VTA PSTH by event, feedback, contrast
#======================
from iblatlas.atlas import AllenAtlas
import numpy as np
import matplotlib.pyplot as plt
from one.api import ONE
from matplotlib import cm
from scipy.ndimage import gaussian_filter1d

one = ONE()

# ----------------------
# PARAMETERS
# ----------------------
PRE_TIME, POST_TIME = 1, 2
BIN_SIZE = 0.01
EVENTS = ['stimOnTrigger_times', 'feedback_times']
SMOOTHING_SIGMA = 2  # adjust smoothing

ba = AllenAtlas(25)  # 25µm resolution

# ----------------------
# PSTH FUNCTION
# ----------------------
def compute_psth(spike_times, event_times, pre_time=1, post_time=2, bin_size=0.01):
    bins = np.arange(-pre_time, post_time + bin_size, bin_size)
    counts = np.zeros(len(bins)-1)
    for et in event_times:
        aligned = spike_times - et
        hist, _ = np.histogram(aligned, bins=bins)
        counts += hist
    counts = counts / (len(event_times) * bin_size)
    return counts, bins[:-1] + bin_size/2

# ----------------------
# SEARCH SESSIONS WITH VTA
# ----------------------
eids = one.search(task_protocol='ephys', atlas_acronym='VTA', project='brainwide')
print(f"Found {len(eids)} sessions with VTA")

# ----------------------
# FUNCTION TO MERGE LEFT/RIGHT CONTRASTS
# ----------------------
def merge_contrasts(left, right):
    merged = []
    for l, r in zip(left, right):
        if not np.isnan(l):
            merged.append(float(l))
        elif not np.isnan(r):
            merged.append(float(r))
        else:
            merged.append(np.nan)
    return np.array(merged)

# ----------------------
# PROCESS SESSIONS
# ----------------------
all_data = []

for eid in eids:
    try:
        print(f"Processing session: {eid}")
        spikes = one.load_object(eid, 'spikes')
        clusters = one.load_object(eid, 'clusters')
        channels = one.load_object(eid, 'channels')
        trials = one.load_object(eid, 'trials')

        # Map clusters to brain regions
        if 'brainLocationIds_ccf_2017' not in channels.keys():
            print(f"Skipping session {eid}: missing brainLocationIds_ccf_2017")
            continue

        cluster_channels = clusters['channels']
        cluster_region_ids = channels['brainLocationIds_ccf_2017'][cluster_channels]
        cluster_acronyms = np.array([ba.regions.id2acronym(rid) for rid in cluster_region_ids])
        vta_clusters = np.where(cluster_acronyms == 'VTA')[0]
        if len(vta_clusters) == 0:
            continue

        # Merge contrasts
        contrast_values = merge_contrasts(trials['contrastLeft'], trials['contrastRight'])

        # Separate correct/incorrect trials
        feedback = np.array(trials['feedbackType'], dtype=float)
        mask_correct = feedback == 1
        mask_incorrect = feedback == -1

        for event in EVENTS:
            for fb_label, mask in zip(['Correct','Incorrect'], [mask_correct, mask_incorrect]):
                unique_contrasts = np.unique(contrast_values[mask])
                for c in unique_contrasts:
                    trial_mask = mask & (contrast_values == c)
                    if trial_mask.sum() == 0:
                        continue

                    session_counts = []
                    for clu in vta_clusters:
                        st = spikes.times[spikes.clusters == clu]
                        counts, t = compute_psth(st, trials[event][trial_mask],
                                                 pre_time=PRE_TIME, post_time=POST_TIME,
                                                 bin_size=BIN_SIZE)
                        if SMOOTHING_SIGMA>0:
                            counts = gaussian_filter1d(counts, sigma=SMOOTHING_SIGMA)
                        session_counts.append(counts)

                    if len(session_counts) > 0:
                        all_data.append({'event': event,
                                         'feedback': fb_label,
                                         'contrast': c,
                                         'counts': np.vstack(session_counts).mean(axis=0),
                                         't': t})

    except Exception as e:
        print(f"Skipping session {eid} due to error: {e}")
        continue

# ----------------------
# PLOT ALL
# ----------------------
for event in EVENTS:
    for fb_label in ['Correct','Incorrect']:
        plt.figure(figsize=(8,4))
        subset = [d for d in all_data if d['event']==event and d['feedback']==fb_label]
        if len(subset)==0:
            continue
        contrasts = np.unique([d['contrast'] for d in subset])
        colors = cm.Greys(np.linspace(0.3,1.0,len(contrasts)))  # gradient gray

        for i, c in enumerate(contrasts):
            c_data = [d['counts'] for d in subset if d['contrast']==c]
            if len(c_data)==0:
                continue
            mean_counts = np.vstack(c_data).mean(axis=0)
            t = subset[0]['t']
            plt.plot(t, mean_counts, color=colors[i], label=f"Contrast {c:.1f}")

        plt.axvline(0, color='red', linestyle='--', label='Event')
        plt.xlabel('Time from event (s)')
        plt.ylabel('Firing rate (Hz)')
        plt.title(f'VTA PSTH - {event} - {fb_label}')
        plt.legend()
        plt.tight_layout()
        plt.show()
