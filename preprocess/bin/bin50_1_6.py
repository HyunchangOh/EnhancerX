import numpy as np

bin_size = 50

chromosomes = [
        'chr1',
        'chr2',
        'chr3',
        'chr4',
        'chr5',
        'chr6',
        # 'chr7',
        # 'chr8',
        # 'chr9',
        # 'chr10',
        # 'chr11',
        # 'chr12',
        # 'chr13',
        # 'chr14',
        # 'chr15',
        # 'chr16',
        # 'chr17',
        # 'chr18',
        # 'chr19',
        # 'chr20',
        # 'chr21',
        # 'chr22',
        # 'chrX'
    ]

features = ["enhancer_atlas","CTCF","DHS","EP300Conservative","h3k4me1","h3k4me2","h3k9me3","h3k27ac","h3k27me3","h3k36me3"]


data_path = "../../../../../scratch/ohh98/la_grande_table/"

# for c in chromosomes:
#     for f in features:
#         data = np.load(data_path+c+"/"+f+".npy")
#         binned_data = []
#         bin_counts = len(data)//bin_size
#         for i in range(bin_counts):
#             bin_value = 0
#             for j in range(i*bin_size,i*bin_size+bin_size):
#                 if data[j]:
#                     bin_value+=1
#             binned_data.append(bin_value)
#         np.save(data_path+c+"/BIN50_"+f+".npy",binned_data)

#         f_1d = f+"_1D_Dist"
#         data = np.load(data_path+c+"/"+f_1d+".npy")
#         binned_data = []
#         bin_counts = len(data)//bin_size
#         for i in range(bin_counts):
#             bin_value = 0
#             for j in range(i*bin_size,i*bin_size+bin_size):
#                 bin_value+=data[j]
#             binned_data.append(bin_value)
#         np.save(data_path+c+"/BIN50_"+f_1d+".npy",binned_data)

#         f_3d = f+"_3D_Dist"
#         data = np.load(data_path+c+"/"+f_3d+".npy")
#         binned_data = []
#         bin_counts = len(data)//bin_size
#         for i in range(bin_counts):
#             bin_value = 0
#             for j in range(i*bin_size,i*bin_size+bin_size):
#                 bin_value+=data[j]
#             binned_data.append(bin_value)
#         np.save(data_path+c+"/BIN50_"+f_3d+".npy",binned_data)

# chromosomes = ["chr7"]

# manual_features = [
    
# "promoter_1D_Dist_any.npy",
# "promoter_1D_Dist_forward.npy",
# # "promoter_1D_Dist_reverse.npy",
# "promoter_3D_Dist_any.npy",
# "cod_1D_Dist.npy"
# ]
# # manual_features = ["Interaction_1D_Dist.npy"]
# for c in chromosomes:
#     for f in manual_features:
#         data = np.load(data_path+c+"/"+f)
#         print(c+f+"loaded")
#         binned_data = []
#         bin_counts = len(data)//bin_size
#         for i in range(bin_counts):
#             bin_value = 0
#             for j in range(i*bin_size,i*bin_size+bin_size):
#                 bin_value+=data[j]
#             binned_data.append(bin_value)
#         np.save(data_path+c+"/BIN50_"+f,binned_data)

manual_features_boolean = [
    "cod",
    "promoter_any",
    "promoter_forward",
    "promoter_reverse",
]

for c in chromosomes:
    for f in manual_features_boolean:
        data = np.load(data_path+c+"/"+f+".npy")
        binned_data = []
        bin_counts = len(data)//bin_size
        for i in range(bin_counts):
            bin_value = 0
            for j in range(i*bin_size,i*bin_size+bin_size):
                if data[j]:
                    bin_value+=1
            binned_data.append(bin_value)
        np.save(data_path+c+"/BIN50_"+f+".npy",binned_data)