'''
chromosome  start   end     name    score   strand  thick(transcribed start)    thick(transcribed end)
chr1	    894625	894685	NOC2L_1	900	    -	    894625	                    894636
'''

import numpy as np
#lengths of chromosomes, from 1 to 22, and then X and Y.
lengths = [249250621, 243199373, 198022430, 191154276, 180915260, 171115067,159138663, 146364022, 141213431, 135534747, 135006516, 133851895, 115169878, 107349540, 102531392, 90354753, 81195210, 78077248,59128983,63025520,48129895,51304566,155270560,59373566]
f = open("../../data/EPD/Hs_EPDnew_006_hg19.bed")

forwards = []
reverses = []
forwards_thick = []
reverses_thick = []

for x in lengths:
    forwards.append(np.full(x,False))
    reverses.append(np.full(x,False))
    forwards_thick.append(np.full(x,False))
    reverses_thick.append(np.full(x,False))

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

for l in f.readlines():
    l = l.strip().split("\t")
    # 0: chr1 / 1:start / 2:end / 3:name / 4:score / 5:strand / 6:think_start / 7:thick_end
    start = int(l[1])
    end = int(l[2])
    thick_start = int(l[6])
    thick_end = int(l[7])
    if l[5]=="+":
        chromosome = chromosome_converter[l[0]]
        for i in range(start,end+1):
            forwards[chromosome][i] = True
        for j in range(thick_start,thick_end+1):
            forwards_thick[chromosome][j] = True
    elif l[5]=="-":
        chromosome = chromosome_converter[l[0]]
        for i in range(start,end+1):
            reverses[chromosome][i] = True
        for j in range(thick_start,thick_end+1):
            reverses_thick[chromosome][j] = True

path = "../../la_grande_table/"
for i in range(len(list(chromosome_converter.keys()))):
    np.save(path+list(chromosome_converter)[i]+"/promoter_forward.npy",forwards[i])
    np.save(path+list(chromosome_converter)[i]+"/promoter_reverse.npy",reverses[i])
    np.save(path+list(chromosome_converter)[i]+"/promoter_transcribed_forward.npy",forwards_thick[i])
    np.save(path+list(chromosome_converter)[i]+"/promoter_transcribed_reverse.npy",reverses_thick[i])
    print(list(chromosome_converter.keys())[i])