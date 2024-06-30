'''
chromosome  start   end     name            score   strand      thickstart  thickend    RGB(for genomebrowser)
chr1	    10244	10357	EH37E1055273	0	    .	        10244	    10357	    225,225,225
'''


import numpy as np
#lengths of chromosomes, from 1 to 22, and then X and Y.
lengths = [249250621, 243199373, 198022430, 191154276, 180915260, 171115067,159138663, 146364022, 141213431, 135534747, 135006516, 133851895, 115169878, 107349540, 102531392, 90354753, 81195210, 78077248,59128983,63025520,48129895,51304566,155270560,59373566]

forwards = []

for x in lengths:
    forwards.append(np.full(x,False))

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

path = "../../../../../../scratch/ohh98/data/hg19/"
target = "../../../../../../scratch/ohh98/la_grande_table/"
for x in list(chromosome_converter.keys()):
# for x in ['chr11']:
    p = path + x + ".fa"
    f = open(p,"r")
    f.readline()
    boolean=[]
    for l in f.readlines():
        for c in l.strip():
            if c.isupper and c != "N":
                boolean.append(True)
            else:
                boolean.append(False)
    np.save(target+x+"/cod.npy",boolean)

