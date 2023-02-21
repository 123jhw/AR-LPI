#!/usr/bin/python3 
BASE_PATH = "data/"
SEQ_PATH = BASE_PATH + "sequence/"
STR_PATH = BASE_PATH + "structure/"
NPZ_PATH = {
    "NPInter" : "npz/NPInter.npz",
    "RPI" : {
        1807 : "npz/RPI1807.npz",
        2241 : "npz/RPI2241.npz",
        369  : "npz/RPI369.npz",
        488  : "npz/RPI488.npz",
        999  : "npz/RPI999.npz",
        0    : "npz/RPI0.npz",
        1    : "npz/RPI1.npz",
        2    : "npz/RPI2.npz",
        3    : "npz/RPI3.npz",
        4    : "npz/RPI4.npz",
        5    : "npz/RPI5.npz",
        6    : "npz/RPI6.npz",
        7    : "npz/RPI7.npz",
        8    : "npz/RPI8.npz",
        9    : "npz/RPI9.npz",
        10    : "npz/RPI10.npz",
        11    : "npz/RPI11.npz",
        12    : "npz/RPI12.npz",
        13    : "npz/RPI13.npz",
        14    : "npz/RPI14.npz",
        15    : "npz/RPI15.npz",
        16    : "npz/RPI16.npz",
        17    : "npz/RPI17.npz",
        18    : "npz/RPI18.npz",
        19    : "npz/RPI19.npz",
        20    : "npz/RPI20.npz",
        21    : "npz/RPI21.npz",
        22    : "npz/RPI22.npz",
        23    : "npz/RPI23.npz",
        24    : "npz/RPI24.npz",
        25    : "npz/RPI25.npz",
        26    : "npz/RPI26.npz",
        27    : "npz/RPI27.npz",
        28    : "npz/RPI28.npz",
        29    : "npz/RPI29.npz",
        30    : "npz/RPI30.npz",
        31    : "npz/RPI31.npz",
        32    : "npz/RPI32.npz"
        
    }
}

PAIRS_PATH = {
    "NPInter" : BASE_PATH + "NPInter_pairs.txt",
    "RPI" : {
        1807 : BASE_PATH + "RPI1807_pairs.txt",
        2241 : BASE_PATH + "RPI2241_pairs.txt",
        369  : BASE_PATH + "RPI369_pairs.txt",
        488  : BASE_PATH + "RPI488_pairs.txt",
        999  : BASE_PATH + "interacting_pairs.txt",
        0  : BASE_PATH + "negative_pairs0.txt",
        1  : BASE_PATH + "negative_pairs1.txt",
        2  : BASE_PATH + "negative_pairs2.txt",
        3  : BASE_PATH + "negative_pairs3.txt",
        4  : BASE_PATH + "negative_pairs4.txt",
        5  : BASE_PATH + "negative_pairs5.txt",
        6  : BASE_PATH + "negative_pairs6.txt",
        7  : BASE_PATH + "negative_pairs7.txt",
        8  : BASE_PATH + "negative_pairs8.txt",
        9  : BASE_PATH + "negative_pairs9.txt",
        10  : BASE_PATH + "negative_pairs10.txt",
        11  : BASE_PATH + "negative_pairs11.txt",
        12  : BASE_PATH + "negative_pairs12.txt",
        13  : BASE_PATH + "negative_pairs13.txt",
        14  : BASE_PATH + "negative_pairs14.txt",
        15  : BASE_PATH + "negative_pairs15.txt",
        16  : BASE_PATH + "negative_pairs16.txt",
        17  : BASE_PATH + "negative_pairs17.txt",
        18  : BASE_PATH + "negative_pairs18.txt",
        19  : BASE_PATH + "negative_pairs19.txt",
        20  : BASE_PATH + "negative_pairs20.txt",
        21  : BASE_PATH + "negative_pairs21.txt",
        22  : BASE_PATH + "negative_pairs22.txt",
        23  : BASE_PATH + "negative_pairs23.txt",
        24  : BASE_PATH + "negative_pairs24.txt",
        25  : BASE_PATH + "negative_pairs25.txt",
        26  : BASE_PATH + "negative_pairs26.txt",
        27  : BASE_PATH + "negative_pairs27.txt",
        28  : BASE_PATH + "negative_pairs28.txt",
        29  : BASE_PATH + "negative_pairs29.txt",
        30  : BASE_PATH + "negative_pairs30.txt",
        31  : BASE_PATH + "negative_pairs31.txt",
        32  : BASE_PATH + "negative_pairs32.txt"
    }
}
SEQ_PATH = {
    "NPInter" : {
        "RNA"     : SEQ_PATH + "NPinter_rna_seq.fa",
        "Protein" : SEQ_PATH + "NPinter_protein_seq.fa"
    },
    "RPI" : {
        1807 : {
            "RNA"     : SEQ_PATH + "RPI1807_rna_seq.fa",
            "Protein" : SEQ_PATH + "RPI1807_protein_seq.fa"
        },
        2241 : {
            "RNA"     : SEQ_PATH + "RPI2241_rna_seq.fa",
            "Protein" : SEQ_PATH + "RPI2241_protein_seq.fa"
        },
        369  : {
            "RNA"     : SEQ_PATH + "RPI369_rna_seq.fa",
            "Protein" : SEQ_PATH + "RPI369_protein_seq.fa"
        },
        488  : {
            "RNA"     : SEQ_PATH + "RPI488_rna_seq.fa",
            "Protein" : SEQ_PATH + "RPI488_protein_seq.fa"
        },
        999  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        0  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        1  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        2  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        3  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        4  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        5  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        6  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        7  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        8  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        9  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        10  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        11  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        12  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        13  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        14  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        15  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        16  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        17  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        18  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        19  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        20  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        21  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        22  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        23  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        23  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        24  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        25  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        26  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        27  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        28  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        29  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        30  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        31  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        },
        32  : {
            "RNA"     : SEQ_PATH + "lncRNA.fa",
            "Protein" : SEQ_PATH + "protein.fa"
        }
    }
}

STR_PATH = {
    "NPInter" : {
        "RNA"     : STR_PATH + "NPinter_rna_struct.fa",
        "Protein" : STR_PATH + "NPinter_protein_struct.fa"
    },
    "RPI" : {
        999 : {
            "Protein" : STR_PATH + "RPI999_protein_struct.fa"
        },
        1807 : {
            "RNA"     : STR_PATH + "RPI1807_rna_struct.fa",
            "Protein" : STR_PATH + "RPI1807_protein_struct.fa"
        },
        2241 : {
            "RNA"     : STR_PATH + "RPI2241_rna_struct.fa",
            "Protein" : STR_PATH + "RPI2241_protein_struct.fa"
        },
        369  : {
            "RNA"     : STR_PATH + "RPI369_rna_struct.fa",
            "Protein" : STR_PATH + "RPI369_protein_struct.fa"
        },
        488  : {
            "RNA"     : STR_PATH + "RPI488_rna_struct.fa",
            "Protein" : STR_PATH + "RPI488_protein_struct.fa"
        }
    }
}