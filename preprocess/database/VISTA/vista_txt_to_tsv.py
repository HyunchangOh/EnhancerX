f = open("../../data/VISTA/vista.txt","r")
g = open("../../processed/VISTA/vista.tsv","w")

lines = f.readlines()

g.write("Reference\tChromosome\tStart\tEnd\tElementNumber\tPatternExpression\tActiveTissues\tSequence\n")
for l in lines:
    # start of a sequence
    if l[0]==">":
        l = l.strip().split("|")
        if l[0]==">Human":
            reference = "hg19"
        else:
            reference = "mm9"
        chromosome = l[1].split(":")[0]
        start = l[1].split(":")[1].split("-")[0].strip()
        end = l[1].split(":")[1].split("-")[1].strip()
        ElementNumber = l[2].strip()
        PatternExpression = l[3].strip()
        ActiveTissues = " "
        if len(l)>=5:
            ActiveTissues = "|".join(l[4:])
        header = [reference,chromosome,start,end,ElementNumber,PatternExpression,ActiveTissues]
        g.write("\t".join(header))
        sequence = ""
    elif l=="\n":
        g.write("\t")
        g.write(sequence)
        g.write("\n")
    else:
        sequence += l.strip()


