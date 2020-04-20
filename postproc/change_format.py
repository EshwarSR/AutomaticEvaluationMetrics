import os
import pandas as pd
import matplotlib.pyplot as plt

fn = os.listdir('../results/candidates_data_sim_dist/')
for an in fn:
	inpath = "../results/candidates_data_sim_dist/" + an
	odf = pd.read_csv(inpath, delimiter='\t')
	# print(len(odf))
	# print(odf.columns)

	groups = odf.groupby('candidate_id')
	# print(len(groups))
	sim = []
	sc = []
	# ref_id = []
	cand_id = []
	for cid, group in groups:
	    cand_id.append(cid)
	    a = group['similarity'].idxmax()
	    # ref_id.append(group['reference_id'][a])
	    sim.append(group['similarity'].max())
	    sc.append(group['score'][a])

	fd = pd.DataFrame()
	fd['cand id'] = cand_id
	fd['similarity score'] = sim
	fd['score'] = sc

	# print(fd.columns)
	outpath = "../results/candidates_data_sim_dist_f/" + an
	fd.to_csv(outpath, sep="\t", index=False, header=True)