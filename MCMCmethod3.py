import networkx as nx
import numpy as np
import random
import csv

import copy

import math
import matplotlib.pyplot as plt
from scipy import stats

boundary_edges = set()
G = nx.grid_2d_graph(9,9)
DList = []

for i in G.nodes():
	G.node[i]['district'] = i[0]
	
def populationscore():
	n = len(G.nodes())
	d = len(DList)
	ideal = float(n) / float(d)
	sum = 0
	for g in DList:
		sum = sum + (len(g.nodes()) - ideal)**2
	Jp = math.sqrt(sum) / ideal
	return Jp
	
def contiguityscore():
	try:
		Jc = 0
		for g in DList:
			if not nx.is_connected(g):
				Jc = Jc + 100000000
	except: 
		return 100000000
	return Jc

def isoperimetricscore():
	sum = 0
	for g in DList:
		p = len(nx.edge_boundary(G, g.nodes()))**2
		q = len(g.nodes())
		if q != 0:
			sum = sum + float(p) / float(q)
		else:
			sum += 10000000
	return sum	
	
def spectralgapscore(subgraphs):
	sum = 0
	for g in subgraphs:
		S = nx.laplacian_spectrum(g)
		gap = (S[len(S) - 1] - S[len(S) - 2])/ S[len(S) - 1]
	sum = sum + gap
	return sum

def makeDList():
	global DList
	DList = []
	for k in range(9):
		nbunch = [i for i in G.nodes() if G.node[i]['district'] == k]
		#print(nbunch)
		DList.append(G.subgraph(nbunch))
		
def score():
	Jp = populationscore()
	Jc = contiguityscore()
	Ji = isoperimetricscore() /270
	return (Jp + Jc + Ji + 1)**4

#Compute ballot data

def ballot(id):
	#id is ideology
	#for now m = 5
	#this outputs the list of nearest voters
	candidates = set([0,.25,.5,.75,1])
	ballot = []
	while len(ballot) < 3:
		pick = -10
		for c in candidates:
			if math.fabs(c - id) < math.fabs(id - pick):
				pick = c
		candidates.remove(pick)
		ballot.append(pick)
	return ballot
	
# Social choice functions:

def BordaCount(ballots, num_cand = 5, num_win = 3):
	candidates = [0, .25, .5, .75, 1]
	candidate_totals = {cand:0 for cand in candidates}
	for cand in candidates:
		list = []
		for ballot in ballots:
			if cand in ballot:
				list.append(ballot.index(cand))
		candidate_totals[cand] += sum(list)*-1
	#print(candidate_totals)
	results = [x[0] for x in sorted(candidate_totals.items(), key=lambda x: x[1])[:3]]
#	print(results)
	return results

def InstantRunoff(ballots, num_cand = 5, num_win = 3, candidates = {0:[], 0.25:[], 0.5:[], 0.75:[], 1:[], "sink":[]}):
	print("start counting!")
	print(ballots)
	candidates = {0:[], 0.25:[], 0.5:[], 0.75:[], 1:[], "sink":[]}
	ballot_list = [{"ranking":ballot, "weight": 1} for ballot in ballots]
	winners_list = set([])
	while len(winners_list) < 3:
		eliminated = False
		numberwinners = len(winners_list)

		print("Candidates and Weights:", candidates)
		print("Candidate Scores", [CandidateScore(candidates[c]) for c in candidates], len(ballots)/4)
		print("Ballots:", ballot_list)
		for c in candidates:
			candidates[c] = []
		for x in ballot_list: 
			candidates[x["ranking"][0]].append(x)
		for c in candidates:
			if CandidateScore(candidates[c]) > len(ballots) / 4:
				winners_list.add(c)
				for x in ballot_list:
				#remove c from all ballots
					if c in x["ranking"]:
						x["ranking"].remove(c)
						x["ranking"].append("sink")
		if len(winners_list) == numberwinners:
			worstscore = 1000000
			worst = "extra"
			#prefering 0 to be elimanted if all tied
			for c in candidates:
				if c != "sink":
					score = CandidateScore(candidates[c])
					if score <= worstscore:
						worstscore = score
						worst = c
			if worst != "extra":
				for x in ballot_list:
					#remove worst from all ballots
					if worst in x["ranking"]:
						x["ranking"].remove(worst)
						x["ranking"].append("sink")
				candidates.pop(worst)
			eliminated = True
		if eliminated == False:
			if len(winners_list) != 0:
				for c in winners_list:
					if c in candidates and (CandidateScore(candidates[c]) != 0):
						score = CandidateScore(candidates[c])
						excess = score - len(ballots)/4
						#epsilon?
						extraweight = excess / score
						if extraweight != 0:
							for x in candidates[c]:
								temp = copy.deepcopy(x["ranking"])
								temp.pop(0)
								temp.append("sink")
								oldweight = x["weight"]
								x["weight"] = x["weight"]*(1 - extraweight)
								ballot_list.append({"ranking":temp, "weight": oldweight*extraweight})
		print("Winners:", winners_list)
	print("VOTING COMPLETE",winners_list)
	return winners_list

#temp.append("sink")
def CandidateScore(c_ballot_lst):
    return sum([x["weight"] for x in c_ballot_lst])	
 
#Boundaries:

def InitializePreference(mean):
	#Random for now
	for p in G.nodes():
		G.node[p]['id'] = np.random.normal(mean,.1)

def ComputeBallots():
	for p in G.nodes():
		G.node[p]['ballot'] = ballot(G.node[p]['id'])
		
def ComputeBinaryBallots():
	for p in G.nodes():
		if G.node[p]['id'] >= .5:
			G.node[p]['ballot'] = 1 
		else:
			G.node[p]['ballot'] = 0

def vote(type):
	if type == "IRV":
		for g in DList:
			ballots = []
			ComputeBallots()
			for x in g.nodes():
				ballots.append(g.node[x]['ballot'])
			g.graph['reps'] = InstantRunoff(ballots)
	if type == "BORDA":
		for g in DList:
			ballots = []
			ComputeBallots()
			for x in g.nodes():
				ballots.append(g.node[x]['ballot'])
			g.graph['reps'] = BordaCount(ballots)   
		
def peoplewill():
	sum = 0
	for p in G.nodes():
		sum += G.node[p]['id']
	return sum / len(G.nodes())
	
def repwill():
	sum = 0
	for g in DList:
		for r in g.graph['reps']:
			if r != "sink":
				sum += r
	return sum / float(3 * len(DList))
	

def InitializeBoundary():
	global boundary_edges
	m = 9
	for i in range(m):
		Y = nx.edge_boundary(G,DList[i].nodes())
		X =set(Y)
		boundary_edges = boundary_edges.union(X)
		
def MCMCstep():
	currentscore = score()
	#Pick random boundary edge
	edge = random.sample(boundary_edges,1)
	#Pick random node of that edge
	node = random.choice(edge[0])
	othernode = 0
	for x in edge[0]:
		if x != node:
			othernode = x
		
	old_district = G.node[node]["district"]
	new_district = G.node[othernode]["district"]
	#Compute these variables
	updatebandd(edge, node, old_district, new_district)
	newscore = score()
	if newscore > currentscore :
		coin = np.random.rand(1)
		if coin > (currentscore / newscore):
			updatebandd(edge, node, new_district, old_district)
		
	
def updatebandd(edge, node, old_district, new_district):
	#Update boundary and districts
	global boundary_edges
	global DList
	
	K = nx.edge_boundary(DList[old_district], [node])
	for x in K:
		x = x
		boundary_edges = boundary_edges.union(set([x]))
		boundary_edges = boundary_edges.union(set([(x[1],x[0])]))
		#adding x adds the directed edge x
	
	new_oldnodes = set(DList[old_district].nodes())
	new_oldnodes.remove(node)
	new_newnodes = set(DList[new_district].nodes())
	new_newnodes.add(node)
	
	G.node[node]["district"] = new_district
	
	DList[old_district] = G.subgraph(new_oldnodes)
	DList[new_district] = G.subgraph(new_newnodes)
	
	#print(nx.edge_boundary(DList[new_district],[node]))
	K= nx.edge_boundary(DList[new_district],[node])
	for x in K:
		x = x
		boundary_edges.discard(x)
		boundary_edges.discard((x[1],x[0]))
		#Need to be careful that undirectd edges are stored as (a,b),...,(b,a) in the edge list
	
pw_list = []
rw_list = []
for k in range(200):
    makeDList()
    InitializeBoundary()
    InitializePreference(np.random.rand(1)[0])
    for i in range(20):
        for i in range(20):
            MCMCstep()
        vote("BORDA")
        print(score())
        pw_list.append(peoplewill())
        rw_list.append(repwill())
plt.plot(pw_list, rw_list, 'ro')
plt.axis([0,1,0,1])
plt.show()
slope, intercept, r_value, p_value, std_err = stats.linregress(pw_list,rw_list)
print("Slope:", slope)
print("r_value:", r_value)
