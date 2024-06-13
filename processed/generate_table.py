g = open("enrichment_GC_PER_sequence.tsv","r")
f = open("../data/enhancer_atlas/GM12878.txt","r")
h = open("la_grande_table.tsv","w")

f_l = f.readlines()
g_l = g.readlines()

h.write("Reference\tChromosome\tStart\tEnd\tSequence\tGC_content\tEnrichment_Score_GM12878\n")
for i in range(len(g_l)):
    row = "hg19\tchr1\t"
    f_ll = f_l[i].strip().split("\t")
    row+= f_ll[1]+"\t"+f_ll[2]+"\t"
    row+= g_l[i]
    h.write(row)