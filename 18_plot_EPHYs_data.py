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
#====================================================================================================

#================================= 2. for VTA for 5 sessions (total 10?) ================================

#====================================================================================================