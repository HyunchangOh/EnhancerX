# enhancer_atlas
# chromsome start   end     enrichment_score 
# chr1	    773300	774100	7.86608754738979

# calculate only for chr1
f = open("../../data/enhancer_atlas/GM12878.txt","r")

enhancer_atlas_db = []
for l in f.readlines():
    l = l.strip().split("\t")
    if l[0]!="chr1":
        break
    enhancer_atlas_db.append((int(l[1]),int(l[2]),float(l[3])))

f.close()
print("part f done")

g = open("../../data/hg19/chr1.fa","r")
g.readline()
lines = g.readlines()
print("lines are read")
length = len(lines)
chromosome_1 = ""
for i in range(length):
    chromosome_1+=lines[i].strip()
    if i%100000 == 0:
        print(i/length, "percent read")

g.close()
print("part g done")

sequence_enrichment = []
for segment in enhancer_atlas_db:
    sequence = chromosome_1[segment[0]:segment[1]]
    sequence_enrichment.append((sequence,segment[2]))

h = open("../../processed/enrichment_score_PER_sequence.tsv","w")
for sequence in sequence_enrichment:
    h.write(str(sequence[0]))
    h.write("\t")
    h.write(str(sequence[1]))
    h.write("\n")
h.close()
