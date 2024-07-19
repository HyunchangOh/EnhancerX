import numpy as np
import heapq

link_source = "../../../data/3DIV/GM12878_in-situ_Mbol_HiCaptureSeq_GMiM_cutoff_10.tsv"
f = open(link_source,"r")
f.readline()

chromosome_converter = {
    "chr1": 0,
    "chr2": 1,
    "chr3": 2,
    "chr4": 3,
    "chr5": 4,
    "chr6": 5,
    "chr7": 6,
    "chr8": 7,
    "chr9": 8,
    "chr10": 9,
    "chr11": 10,
    "chr12": 11,
    "chr13": 12,
    "chr14": 13,
    "chr15": 14,
    "chr16": 15,
    "chr17": 16,
    "chr18": 17,
    "chr19": 18,
    "chr20": 19,
    "chr21": 20,
    "chr22": 21,
    "chrX":22,
    "chrY":23
}

links = []
for i in range(24):
    links.append([])

for l in f.readlines():
    l= l.strip().split()
    links[chromosome_converter[l[-3]]].append((int(l[-2]),int(l[-1])))

f.close()

def dijkstra_1d_with_links(array, links):
    """Calculate the shortest distance to the closest True value in a 1D array with links."""
    n = len(array)
    distances = [float('inf')] * n
    adjacency_list = {i: [] for i in range(n)}
    
    # Create the adjacency list
    for i in range(n):
        if i > 0:
            adjacency_list[i].append((1, i-1))
        if i < n-1:
            adjacency_list[i].append((1, i+1))
    
    for link in links:
        loc1, loc2 = link
        adjacency_list[loc1].append((1, loc2))
        adjacency_list[loc2].append((1, loc1))
    
    # Initialize the priority queue with all True indices
    priority_queue = []
    for i in range(n):
        if array[i]:
            distances[i] = 0
            heapq.heappush(priority_queue, (0, i))
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        if current_distance > distances[current_node]:
            continue
        
        for edge_distance, neighbor in adjacency_list[current_node]:
            distance = current_distance + edge_distance
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances

path = "../../../la_grande_table/"

for i in range(24):
    c = list(chromosome_converter.keys())[i]
    array = np.load(path+c+"/h3k27me3.npy")
    distances = dijkstra_1d_with_links(array, links[i])
    np.save(path+c+"/h3k27me3_3D_Dist.npy",distances)

