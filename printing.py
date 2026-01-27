import numpy as np
from colorama import Fore
from datetime import datetime

from gradientDescent import heuristic


# print out details for a given round
# used in gradeRounds and fightLookup
def printRoundDetails(data, i, prediction, parameters, rank, mlp, dampener=30.0, sharpness=2.0):
    """
    print out details for a given round
    used in gradeRounds and fightLookup
    """
    name = data['name'][i]
    nameO = data['nameO'][i]
    roundNum = data['round'][i]

    if rank != None:
        rankStr = f"{rank}. "
    else:
        rankStr = ""

    if data['scores'][i] > 0:
        print(f"\n{rankStr}Round {roundNum} : {Fore.LIGHTRED_EX}{name}{Fore.WHITE} wins over {Fore.LIGHTBLUE_EX}{nameO}{Fore.WHITE}")
        # print judge score with right number of decimals
        if data['scores'][i] == 1 or data['scores'][i] == -1:
            print(f"\tJudge Score : {Fore.LIGHTRED_EX}{int(data['scores'][i])}{Fore.WHITE}")
        else:
            print(f"\tJudge Score : {Fore.LIGHTRED_EX}{data['scores'][i]:.2f}{Fore.WHITE}")

    else:
        print(f"\n{rankStr}Round {roundNum} : {Fore.LIGHTBLUE_EX}{nameO}{Fore.WHITE} wins over {Fore.LIGHTRED_EX}{name}{Fore.WHITE}")
        # print judge score with right number of decimals
        if data['scores'][i] == 1 or data['scores'][i] == -1:
            print(f"\tJudge Score : {Fore.LIGHTBLUE_EX}{int(data['scores'][i])}{Fore.WHITE}")
        else:
            print(f"\tJudge Score : {Fore.LIGHTBLUE_EX}{data['scores'][i]:.2f}{Fore.WHITE}")

    if prediction > 0:
        print(f"\tPrediction  : {Fore.LIGHTRED_EX}{prediction:.2f}{Fore.WHITE}")
    else:
        print(f"\tPrediction  : {Fore.LIGHTBLUE_EX}{prediction:.2f}{Fore.WHITE}")

    cost = (prediction - data['scores'][i])**2
    print(f"\tCost        : {cost:.2f}")
    
    # print out parameters and their raw values
    maxLength = 1 + max(len(param) for param in parameters)
    for param in parameters:
        value = data[param][i]
        if 'total' in param:
            print(f"\t{param:<{maxLength}}: ",end="")
            print(f"{value:>8.2f}")
        else:
            valueO = data[param + 'O'][i]
            print(f"\t{param:<{maxLength}}: ",end="")
            print(f"{Fore.LIGHTRED_EX}{value:>8.2f}\t{Fore.LIGHTBLUE_EX}{valueO:>8.2f}{Fore.WHITE}")
    return cost
    
        


# finds and prints the rounds with the biggest difference between heuristic value and score
def gradeRounds(data, mlpPredictions, parameters, optimizedParams, top=10, dampener=50.0, sharpness=2.0):
    
    differences = {}  # dictionary to store round differences

    for i in range(0, len(data['round']), 2):  # step by 2 to process both perspectives at once
        # create a dictionary with both fighter and opponent data for the current round
        roundData = {}
        for param in parameters:
            if 'total' in param or 'Share' in param:
                # for total/share metrics, create pair with red at i, blue at i+1
                roundData[param] = [data[param][i], data[param + 'O'][i]]
            else:
                # for differential metrics, create pair with red at i, blue at i+1
                roundData[param] = [data[param][i], data[param + 'O'][i]]
                # opponent metrics need to be reversed for blue fighter perspective
                roundData[param + 'O'] = [data[param + 'O'][i], data[param][i]]
        
        # gd : calculate the heuristic value for the current round
        if mlpPredictions == None:
            prediction = heuristic(optimizedParams, roundData, parameters, dampener, sharpness)[0]
        # mlp : use the prediction value
        else:
            prediction = mlpPredictions[i]
        
        # calculate difference for both perspectives
        diff = abs(prediction - data['scores'][i])
        
        # create a unique key for this round
        roundKey = (data['name'][i], data['nameO'][i], data['date'][i], data['round'][i])
        
        # store the larger difference and the index
        differences[roundKey] = (diff, i)

    # sort differences by difference in descending order
    sortedDifferences = sorted(differences.items(), key=lambda x: x[1][0], reverse=True)

    print(f"\nTop {min(top, len(sortedDifferences))} Rounds with Biggest Prediction / Score Differences:")
    rank = 1

    for (name, nameO, date, roundNum), (diff, i) in sortedDifferences[:top]:
        roundData = {}
        for param in parameters:
            if 'total' in param:
                roundData[param] = [data[param][i]]
            else:
                roundData[param] = [data[param][i]]
                roundData[param + 'O'] = [data[param + 'O'][i]]
        
        # calculate the predicted value for the current round
        if mlpPredictions == None:     # gd :  plug optimized params into heuristic function
            prediction = heuristic(optimizedParams, roundData, parameters, dampener, sharpness)[0]
        else:                       # mlp : use mlpPredictions array
            prediction = mlpPredictions[i]
        
        # print the round details
        printRoundDetails(data, i, prediction, parameters, rank, (mlpPredictions!=None), dampener, sharpness)

        rank += 1


# prints the 10 worst judges fights
def gradeFights(data, parameters, optimizedParams, predictions, top=15, dampener=50.0, sharpness=2.0):
    """
    prints the worst judged fights
    """
    from gradientDescent import heuristic
    
    fights = {
        'red': [],
        'blue': [],
        'date': [],
        'grade': [],
        'roundDetails': []
    }

    # helper function to find or create a fight entry
    def findOrCreateFight(fighter1, fighter2, date):
        for i, (f1, f2, d) in enumerate(zip(fights['red'], fights['blue'], fights['date'])):
            if d == date and ((f1 == fighter1 and f2 == fighter2) or (f1 == fighter2 and f2 == fighter1)):
                return i
        fights['red'].append(fighter1)
        fights['blue'].append(fighter2)
        fights['date'].append(date)
        fights['grade'].append(0)
        fights['roundDetails'].append([])
        return len(fights['red']) - 1

    # process each round
    for i in range(0, len(data['round']), 2):
        red = data['name'][i]
        blue = data['nameO'][i]
        date = data['date'][i]
        roundNum = data['round'][i]

        # use predictions if available (mlp), otherwise calculate using heuristic (gd)
        if predictions is not None:
            prediction = predictions[i]
        else:
            roundData = {param: [data[param][i]] for param in parameters}
            roundData.update({param + 'O': [data[param + 'O'][i]] for param in parameters if 'total' not in param})
            prediction = heuristic(optimizedParams, roundData, parameters, dampener, sharpness)[0]

        cost = (data['scores'][i] - prediction)**2

        # find or create the fight entry
        fightIndex = findOrCreateFight(red, blue, date)

        # update the fight's grade and round details
        fights['grade'][fightIndex] += cost
        fights['roundDetails'][fightIndex].append((roundNum, data['scores'][i], prediction, cost))

    # sort fights by grade in descending order and get top fights
    sortedFights = sorted(zip(fights['red'], fights['blue'], fights['date'], fights['grade'], fights['roundDetails']), 
                          key=lambda x: x[3], reverse=True)[:top]

    # print worst judged fights with round details
    print(f"\nTop {top} fights by highest cost:")
    for i, (red, blue, date, grade, rounds) in enumerate(sortedFights, 1):
        print(f"\n{i}. {Fore.LIGHTRED_EX}{red}{Fore.WHITE} vs {Fore.LIGHTBLUE_EX}{blue}{Fore.WHITE}    {date} : Total Cost = {grade:.2f}")
        print("\tRound | Judge Score | Predicted Score | Round Cost")
        print("\t--------------------------------------------------")
        for roundNum, judgeScore, predictedScore, roundGrade in sorted(rounds):
            print(f"\t{roundNum:5d} | ", end="")
            if judgeScore > 0:
                print(f"{Fore.LIGHTRED_EX}{judgeScore:11.2f}{Fore.WHITE} | ", end="")
            else:
                print(f"{Fore.LIGHTBLUE_EX}{judgeScore:11.2f}{Fore.WHITE} | ", end="")
            if predictedScore > 0:
                print(f"{Fore.LIGHTRED_EX}{predictedScore:15.2f}{Fore.WHITE} | ", end="")
            else:
                print(f"{Fore.LIGHTBLUE_EX}{predictedScore:15.2f}{Fore.WHITE} | ", end="")
            print(f"{roundGrade:.2f}")

    return sortedFights



# print the percentage accuracy for a set of parameters
def printAccuracy(pmData, disagreeThreshold):

    unanTotal = 0
    unanCorrect = 0
    splitTotal = 0
    splitCorrect = 0

    for i in range(0, len(pmData['scores']), 2):

        if abs(pmData['heuristic'][i]) >= disagreeThreshold:

            # split decision round
            if pmData['scores'][i] < 1 and pmData['scores'][i] > -1:
                splitTotal += 1
                if np.sign(pmData['heuristic'][i]) == np.sign(pmData['scores'][i]):
                    splitCorrect += 1
            # unanimous decision round
            else:
                unanTotal += 1
                if np.sign(pmData['heuristic'][i]) == np.sign(pmData['scores'][i]):
                    unanCorrect += 1

    print(f"\n\tUnanimous Rounds:\t\t\t\t{ unanTotal }")
    print(f"\tSplit Rounds:\t\t\t\t\t{ splitTotal }\n")

    if disagreeThreshold != 0.0:    print(f"   ( disagreement threshold of {disagreeThreshold} )")
    print(f"\tMajority Agreement for Unanimous Rounds:\t{ round(100* unanCorrect / unanTotal,3) }%")
    if splitTotal == 0:
        print(f"\tMajority Agreement for Split Rounds:\t\tN/A (no split rounds)")
    else:
        print(f"\tMajority Agreement for Split Rounds:\t\t{ round(100* splitCorrect / splitTotal,3) }%")
    print(f"\tMajority Agreement:\t\t\t\t{ round(100* (unanCorrect+splitCorrect) / (unanTotal+splitTotal),3) }%")

    # calculate individual judge agreement from existing metrics
    if (unanTotal + splitTotal) > 0:
        correctDecisions = (3 * unanCorrect + 2 * splitCorrect + 1 * (splitTotal - splitCorrect))
        totalDecisions = 3 * (unanTotal + splitTotal)
        individualJudgeAccuracy = correctDecisions / totalDecisions
        print(f"\tIndividual Judge Agreement:\t\t\t{round(100 * individualJudgeAccuracy, 3)}%\n")


# print out optimized parameters
def printValues(parameters, optimizedParams, bestValues, best, includeMappingParams=False):
    """
    print out optimized parameters including mapping parameters if applicable
    """
    # ensure optimizedParams is a numpy array
    if not isinstance(optimizedParams, np.ndarray):
        optimizedParams = np.array(optimizedParams)
    
    # flatten if needed (in case it's multidimensional)
    if optimizedParams.ndim > 1:
        optimizedParams = optimizedParams.flatten()
    
    # print coefficient values
    maxLength = max(len(param) for param in parameters)
    
    # determine how many parameters are weights vs mapping params
    numWeights = len(parameters)
    
    # if different number of parameters than for bestValues or best flag is true
    if len(bestValues) != numWeights or best:
        for i, param in enumerate(parameters):
            # ensure we have enough parameters
            if i < len(optimizedParams):
                # extract scalar value if needed
                value = float(optimizedParams[i]) if hasattr(optimizedParams[i], '__float__') else optimizedParams[i]
                print(f"\t{param:<{maxLength}} : {value:.7f}")
            else:
                print(f"\t{param:<{maxLength}} : ERROR - parameter missing")
        
        # print mapping parameters if included
        if includeMappingParams and len(optimizedParams) > numWeights:
            dampenerValue = float(optimizedParams[-2]) if len(optimizedParams) > numWeights else 0.0
            sharpnessValue = float(optimizedParams[-1]) if len(optimizedParams) > numWeights + 1 else 0.0
            print(f"\t{'dampener (D)':<{maxLength}} : {dampenerValue:.7f}")
            print(f"\t{'sharpness (S)':<{maxLength}} : {sharpnessValue:.7f}")
    
    # same params as bestValue - show percentage changes
    else:
        for i, param in enumerate(parameters):
            if i < len(optimizedParams):
                # extract scalar value
                value = float(optimizedParams[i]) if hasattr(optimizedParams[i], '__float__') else optimizedParams[i]
                print(f"\t{param:<{maxLength}} : {value:.7f}\t ",end="")
                
                # find if coefficients sign changed
                signChange = False
                if np.sign(bestValues[param]) != np.sign(value):
                    signChange = True
                
                # print with sign change
                if signChange:
                    percentDiff = 100*(abs(value)-abs(bestValues[param]))/abs(bestValues[param])
                    if percentDiff > 0: 
                        print("+",end="")
                    print(f"{percentDiff:.3f}%\t(sign change)")
                
                # print without sign change
                else:
                    percentDiff = 100*(value-bestValues[param])/bestValues[param]
                    if percentDiff > 0: 
                        print(f"{Fore.GREEN}+",end="")
                    else:
                        print(f"{Fore.RED}",end="")
                    print(f"{percentDiff:.3f}%{Fore.WHITE}")
            else:
                print(f"\t{param:<{maxLength}} : ERROR - parameter missing")


# prints out the win rate for orthodox vs southpaw rounds
def printStanceWinRate(data):
    southpawJudgeWins = 0       # the number of times a judge scored for a southpaw fighter
    southpawWins = 0            # the times a southpaw fighter overall won a round
    southpawTotal = 0

    ordthodoxJudgeWins = 0      # the number of times a judge scored for an orthodox fighter
    orthodoxWins = 0            # the times an orthodox fighter overall won a round
    orthodoxTotal = 0

    for i in range(0, len(data['round'])):

        # ortodox
        if data['orthodox'][i] > 50:
            orthodoxTotal += 1
            # check if the orthodox fighter won the round (overall)
            if data['scores'][i] > 0:
                orthodoxWins += 1
            # check how many judges scored for the orthodox fighter
            for j in range(1,4):
                if data['score'+str(j)][i] > 0:
                    ordthodoxJudgeWins += 1

        # southpaw
        elif data['southpaw'][i] > 50:
            southpawTotal += 1
            if data['scores'][i] > 0:
                southpawWins += 1
            # check how many judges scored for the southpaw fighter
            for j in range(1,4):
                if data['score'+str(j)][i] > 0:
                    southpawJudgeWins += 1
    
    if orthodoxTotal == 0 or southpawTotal == 0:
        print(f"{Fore.BLUE}WARNING:\tStance win rate skipped to avoid div by 0{Fore.WHITE}")
        return

    print("Orthodox vs Southpaw Win Rate")
    print("--------------------------------------------------------------------------------")
    print("\nOrthodox:")
    print(f"{round(100*orthodoxWins/orthodoxTotal,2)}% of rounds won                      \t({orthodoxWins}/{orthodoxTotal})")
    print(f"{round(100*ordthodoxJudgeWins/(orthodoxTotal*3),2)}% of individual judge decisions won\t({ordthodoxJudgeWins}/{orthodoxTotal*3})")
    print("\nSouthpaw:")
    print(f"{round(100*southpawWins/southpawTotal,2)}% of rounds won                      \t({southpawWins}/{southpawTotal})")
    print(f"{round(100*southpawJudgeWins/(southpawTotal*3),2)}% of individual judge decisions won\t({southpawJudgeWins}/{southpawTotal*3})\n")




# prints out pearson & spearman correlation metrics for each parameter
# shows how each param's +/- is correlated with the judges' scores
def printCorrelations(correlations):
    print("\nCorrelation Analysis:")
    print("-" * 80)
    print(f"{'Attribute':<20} {'Pearson r':<12} {'Pearson p':<12} {'Spearman r':<12} {'Spearman p':<12}")
    print("-" * 80)
    
    # sort attributes by absolute Pearson correlation
    sortedAttrs = sorted(correlations.items(), 
                        key=lambda x: abs(x[1]['pearson']['r']), 
                        reverse=True)
    
    for attr, corr in sortedAttrs:
        print(f"{attr:<22} {corr['pearson']['r']:>10.3f}   {corr['pearson']['p']:>10.3f}   "
              f"{corr['spearman']['r']:>10.3f}   {corr['spearman']['p']:>10.3f}")



# judge isolation functions
# prints names of judges who have scored the most rounds
def judgeFrequency(data, judgeName, top=50):

    if judgeName != None:   return      # dont run if only looking at data for a single judge 

    judgeFreq = {}
    total = 0
    for i in range(0, len(data['round']), 2):
        total += 1
        judges = [data['judge1'][i], data['judge2'][i], data['judge3'][i]]
        # check if a judge is listed twice
        if len(set(judges)) != 3:
            print(f"{Fore.YELLOW}Warning:\tFight between {data['name'][i]} and {data['nameO'][i]} has duplicate judges{Fore.WHITE}")
        # increment count for each judge
        for judge in judges:
            judgeFreq[judge] = judgeFreq.get(judge, 0) + 1

    sortedJudges = sorted(judgeFreq.items(), key=lambda x: x[1], reverse=True)
    topList = sortedJudges[:top]

    print(f"Judges Sorted By Number of Rounds (Total: {total})")
    print("------------------------------------")
    maxNameLength = max(len(judge) for judge, _ in topList)
    for judge, count in topList:
        print(f"\t{judge:<{maxNameLength}} : {count}")

    return topList

# counts the number of unique judges in the dataset after round exclusion
def countUniqueJudges(data):
    uniqueJudges = set()
    
    # loop through every other entry since each round has 2 fighters
    for i in range(0, len(data['round']), 2):
        # add all three judges for this round to the set
        uniqueJudges.add(data['judge1'][i])
        uniqueJudges.add(data['judge2'][i])
        uniqueJudges.add(data['judge3'][i])
    
    return len(uniqueJudges)


# print list of judges ranked by their accuracy (ie. how often they agree with prediction) or cost
def rankJudges(data, predictions, disagreeThreshold, judgeName, sampleRank=False, recursive=False):

    # only include judges with more than 75 rounds
    roundThreshold = 20

    if judgeName is not None:   
        return      # don't run if only looking at data for a single judge 

    # if there is disagreement threshold, get ranked list w/o threshold to compare rankings later    
    if disagreeThreshold != 0:
        noThresh = rankJudges(data, predictions, 0, judgeName, sampleRank)        

    judgeFreq = {}          # name : [totalRounds, roundsCorrect, accuracy%, ignoredRounds]     --- accuracy rank
                            # name : [totalRounds, totalCost, averageCost, 0]                   --- cost rank
    
    # error check that predictions lines up with data 
    if len(predictions) != len(data['score1']):
        print(f"\n{Fore.RED}ERROR: prediction array isn't same size as score array")
        print(f"{len(predictions)} != {len(data['score1'])}{Fore.WHITE}\n")

    # first pass - gather all rounds for each judge
    judgeRounds = {}  # judge name -> list of round indices
    
    # loop through every other entry
    for i in range(0, len(data['round']), 2):
        prediction = predictions[i]     # get heuristic for this round

        for judgeNum in range(1, 4):
            name = data[f'judge{judgeNum}'][i]
            if name not in judgeRounds:
                judgeRounds[name] = []
            
            # only include rounds that meet the disagreement threshold
            if disagreeThreshold == -1 or abs(prediction) >= disagreeThreshold:
                judgeRounds[name].append((i, judgeNum))

    # Remove judges who have less than roundThreshold rounds
    judgeRounds = {judge: rounds for judge, rounds in judgeRounds.items() if len(rounds) >= roundThreshold}

    # second pass - use equal sample sizes if sampleRank is True, otherwise use all rounds
    for name, rounds in judgeRounds.items():
        # randomly select roundThreshold rounds only if sampleRank is True
        if sampleRank and len(rounds) > roundThreshold:
            np.random.seed(42)  # use fixed seed for reproducibility
            selectedRounds = np.random.choice(len(rounds), roundThreshold, replace=False)
            rounds = [rounds[i] for i in selectedRounds]
        
        # initialize judge stats
        judgeFreq[name] = [0, 0, 0.0, 0]
        
        # process each round
        for i, judgeNum in rounds:
            prediction = predictions[i]
            
            # increment total rounds
            judgeFreq[name][0] += 1
            
            # if cost rank
            if disagreeThreshold == -1:
                # increment total cost
                cost = (prediction - data[f'score{judgeNum}'][i])**2
                judgeFreq[name][1] += cost
            # if acc rank & judges agree with prediction, increment correct rounds
            elif np.sign(prediction) == np.sign(data[f'score{judgeNum}'][i]):
                judgeFreq[name][1] += 1
        
        # update total accuracy (or avg cost)
        judgeFreq[name][2] = judgeFreq[name][1]/judgeFreq[name][0]

    # if cost, sort ascending
    if disagreeThreshold == -1:
        sortedJudges = sorted(judgeFreq.items(), key=lambda x: x[1][2], reverse=False)
        print("\nJudges Sorted By Average Cost" + (" (using equal sample sizes)" if sampleRank else "") + "\n------------------------------------")

    # if accuracy, sort descending
    else:
        sortedJudges = sorted(judgeFreq.items(), key=lambda x: x[1][2], reverse=True)
        if recursive:   # if recursive call, return sorted list w/o printing anything
            return sortedJudges
        else:
            print("\nJudges Sorted By Accuracy" + (" (using equal sample sizes)" if sampleRank else "") + "\n------------------------------------")
        
    maxNameLength = max(len(name) for name, _ in sortedJudges)
    rank = 1
    for name, (totalRounds, correctRounds, accuracy, ignoredRounds) in sortedJudges:

        # print judge w cost
        if disagreeThreshold == -1:
            print(f"\t{name:<{maxNameLength}} : {accuracy:.3f} ", end="")
        # print judge w accuracy
        else:
            print(f"\t{name:<{maxNameLength}} : {100*accuracy:.2f}% ", end="")

        if disagreeThreshold != 0:
            # loop through noThres to find matching judge and save its rank
            for i in range(0, len(noThresh)):
                if name == noThresh[i][0]:
                    rankDiff = i+1 - rank
                    break
            # print positions moved up or down
            if rankDiff > 0:
                print(f"{Fore.GREEN}+{rankDiff}{Fore.WHITE} ", end="")
            elif rankDiff < 0:
                print(f"{Fore.RED}{rankDiff}{Fore.WHITE} ", end="")
            else:
                print(f" - ", end="")
        
        # print total rounds
        print(f"({totalRounds} rounds)")
        rank += 1

    return sortedJudges


# calculate optimal shrinkage parameter k using Method of Moments
# estimates between-judge variance from the data and uses it to compute principled k
def calculateOptimalK(judgeAccuracy):
	# extract human judges only (exclude models and aggregates)
	judgeNames = [name for name in judgeAccuracy.keys()
	             if name not in ['prediction', 'prediction (test set only)', 'all judges']]

	# collect observed accuracies and round counts for each judge
	accuracies = []
	roundCounts = []
	for name in judgeNames:
		totalComparisons, correctComparisons = judgeAccuracy[name]
		if totalComparisons > 0:
			accuracy = correctComparisons / totalComparisons
			numRounds = totalComparisons / 2  # judges get 2 comparisons per round
			accuracies.append(accuracy)
			roundCounts.append(numRounds)

	if len(accuracies) < 2:
		print(f"{Fore.YELLOW}Warning: Not enough judges for Method of Moments estimation. Using k=50.{Fore.WHITE}")
		return 50.0

	# calculate overall mean accuracy
	overallMean = np.mean(accuracies)

	# calculate sample variance of observed accuracies
	sampleVariance = np.var(accuracies, ddof=1)  # use unbiased estimator

	# calculate expected within-judge variance (measurement noise)
	# for binomial data: Var(p_i) ≈ p(1-p)/n_i
	withinVariances = [overallMean * (1 - overallMean) / n for n in roundCounts]
	meanWithinVariance = np.mean(withinVariances)

	# estimate between-judge variance using Method of Moments
	# τ² = sample variance - expected within variance
	betweenVariance = max(0.0, sampleVariance - meanWithinVariance)

	# calculate optimal k = p(1-p) / τ²
	# if betweenVariance is very small, judges are homogeneous → large k (strong shrinkage)
	# if betweenVariance is large, judges vary widely → small k (weak shrinkage)
	if betweenVariance > 0:
		optimalK = (overallMean * (1 - overallMean)) / betweenVariance
	else:
		# if no between-judge variance detected, use strong shrinkage
		optimalK = 1000.0  # effectively treat all judges as identical

	# print diagnostic information
	print(f"\n{Fore.GREEN}Method of Moments Estimation:{Fore.WHITE}")
	print(f"\tOverall mean accuracy:\t\t{100*overallMean:.2f}%")
	print(f"\tSample variance:\t\t{sampleVariance:.6f}")
	print(f"\tExpected within-judge variance:\t{meanWithinVariance:.6f}")
	print(f"\tBetween-judge variance (τ²):\t{betweenVariance:.6f}")
	print(f"\tOptimal k:\t\t\t{optimalK:.2f} virtual rounds")

	return optimalK


# evaluate judges and prediction system using exhaustive pairwise comparisons
# for each judge: compare against both other judges on each round they scored
# for the model: compare against all 3 judges on each round
# this eliminates randomness and uses all available comparison information
# optionally applies Empirical Bayes shrinkage to stabilize accuracy estimates
def pairwiseComparison(data, predictions, testData, testPredictions, roundThreshold=20, shrinkageK=None):

	judgeAccuracy = {}  # judge name -> [total comparisons, correct comparisons]

	judgeAccuracy['prediction'] = [0, 0]
	judgeAccuracy['all judges'] = [0, 0]

	# loop through every round
	for i in range(0, len(data['round']), 2):

		# for each judge, compare against the other 2 judges
		for judgeNum in range(1, 4):

			# add judge to accuracy dict if not already present
			name = data[f'judge{judgeNum}'][i]
			if name not in judgeAccuracy:
				judgeAccuracy[name] = [0, 0]

			# compare against each of the other 2 judges
			for otherJudgeNum in range(1, 4):
				if otherJudgeNum != judgeNum:
					# increment total comparisons
					judgeAccuracy[name][0] += 1
					judgeAccuracy['all judges'][0] += 1

					# check if the scores match
					if np.sign(data[f'score{judgeNum}'][i]) == np.sign(data[f'score{otherJudgeNum}'][i]):
						judgeAccuracy[name][1] += 1
						judgeAccuracy['all judges'][1] += 1

		# for the prediction system, compare against all 3 judges
		for judgeNum in range(1, 4):
			# increment total comparisons
			judgeAccuracy['prediction'][0] += 1

			# check if the prediction matches this judge
			if np.sign(predictions[i]) == np.sign(data[f'score{judgeNum}'][i]):
				judgeAccuracy['prediction'][1] += 1

	# if we are using -split then create a second entry for the prediction system, using only the testing set
	if testData is not None:
		judgeAccuracy['prediction (test set only)'] = [0, 0]
		for i in range(0, len(testData['round']), 2):
			# for the prediction system, compare against all 3 judges
			for judgeNum in range(1, 4):
				# increment total comparisons
				judgeAccuracy['prediction (test set only)'][0] += 1

				# check if the prediction matches this judge
				if np.sign(testPredictions[i]) == np.sign(testData[f'score{judgeNum}'][i]):
					judgeAccuracy['prediction (test set only)'][1] += 1

	# apply Empirical Bayes shrinkage if requested
	shrunkAccuracy = {}  # name -> shrunken accuracy (for sorting and display)
	if shrinkageK is not None:
		# if shrinkageK is 'opt', calculate optimal k using Method of Moments
		if shrinkageK == 'opt':
			shrinkageK = calculateOptimalK(judgeAccuracy)

		# calculate overall mean accuracy from human judges only (exclude models and aggregate)
		judgeNames = [name for name in judgeAccuracy.keys()
		             if name not in ['prediction', 'prediction (test set only)', 'all judges']]

		# calculate mean using raw observed accuracies
		totalJudgeAccuracy = 0.0
		for name in judgeNames:
			totalComparisons, correctComparisons = judgeAccuracy[name]
			if totalComparisons > 0:
				totalJudgeAccuracy += correctComparisons / totalComparisons
		overallMean = totalJudgeAccuracy / len(judgeNames) if len(judgeNames) > 0 else 0.5

		# apply shrinkage to human judges only (not to models)
		for name, (totalComparisons, correctComparisons) in judgeAccuracy.items():
			if totalComparisons > 0:
				# calculate raw accuracy
				rawAccuracy = correctComparisons / totalComparisons

				# only apply shrinkage to human judges
				if name not in ['prediction', 'prediction (test set only)', 'all judges']:
					# convert comparisons to rounds (judges get 2 comparisons per round)
					numRounds = totalComparisons / 2

					# apply shrinkage formula: (n * observed + k * mean) / (n + k)
					shrunkAcc = (numRounds * rawAccuracy + shrinkageK * overallMean) / (numRounds + shrinkageK)
					shrunkAccuracy[name] = shrunkAcc
				else:
					# for models and aggregates, use raw accuracy (no shrinkage)
					shrunkAccuracy[name] = rawAccuracy
			else:
				shrunkAccuracy[name] = 0.0

		# sort by shrunken accuracy
		sortedJudges = sorted(judgeAccuracy.items(), key=lambda x: shrunkAccuracy[x[0]], reverse=True)
	else:
		# no shrinkage - sort by raw accuracy
		sortedJudges = sorted(judgeAccuracy.items(), key=lambda x: x[1][1]/x[1][0] if x[1][0] > 0 else 0, reverse=True)

	# print out judges ranked by accuracy if they have more than roundThreshold rounds
	# note: judges get 2 comparisons per round, predictions get 3 comparisons per round
	if shrinkageK is not None:
		print(f"\nJudges Ranked By Pairwise Comparison Accuracy (with Empirical Bayes Shrinkage, k={shrinkageK})")
	else:
		print("\nJudges Ranked By Pairwise Comparison Accuracy")
	print("------------------------------------")
	maxNameLength = max(len(name) for name, _ in sortedJudges)+1
	rank = 1
	for name, (totalComparisons, correctComparisons) in sortedJudges:
		# convert comparisons back to rounds for threshold check
		# judges get 2 comparisons per round, prediction systems get 3
		if name in ['prediction', 'prediction (test set only)']:
			numRounds = totalComparisons / 3
		else:
			numRounds = totalComparisons / 2

		# only print judges with more than roundThreshold rounds
		if numRounds < roundThreshold:
			continue

		if rank == 10:  # adjust max name length at rank 10 to adjust for longer rank number
			maxNameLength -= 1

		# calculate raw accuracy
		rawAccuracy = correctComparisons / totalComparisons

		# format output based on whether shrinkage is applied
		if shrinkageK is not None and name not in ['prediction', 'prediction (test set only)', 'all judges']:
			# show both raw and shrunken accuracy for human judges only
			shrunkAcc = shrunkAccuracy[name]
			accuracyStr = f"{100*rawAccuracy:.2f}% → {100*shrunkAcc:.2f}%"
		else:
			# show only raw accuracy for models and when shrinkage is disabled
			accuracyStr = f"{100*rawAccuracy:.2f}%"

		# print with appropriate color
		if name == 'prediction':
			print(f"{Fore.CYAN}\t{rank}. {name:<{maxNameLength}} : {accuracyStr} ({correctComparisons}/{totalComparisons}){Fore.WHITE}")
		elif name == 'prediction (test set only)':
			print(f"{Fore.MAGENTA}\t{rank}. {name:<{maxNameLength}} : {accuracyStr} ({correctComparisons}/{totalComparisons}){Fore.WHITE}")
		elif name == 'all judges':
			print(f"{Fore.YELLOW}\t{rank}. {name:<{maxNameLength}} : {accuracyStr} ({correctComparisons}/{totalComparisons}){Fore.WHITE}")
		else:
			print(f"\t{rank}. {name:<{maxNameLength}} : {accuracyStr} ({correctComparisons}/{totalComparisons})")
		rank += 1

	return
    


# prompts the user for a fight and prints out round by round details
# loops until user enters 'exit'
def fightLookup(data, mlpPredictions, parameters, optimizedParameters, dampener=50.0, sharpness=2.0):    
    while True:
        lookupString = input("\nEnter boxers' full names and date in m/d/yr format: \nEx: Floyd Mayweather Jr vs Manny Pacquiao (5/2/2015)\n")

        # exit condition
        if lookupString == 'exit':
            break

        # error check if the format isn't correct
        if 'vs' not in lookupString or '(' not in lookupString or ')' not in lookupString:
            print(f"{Fore.RED}ERROR:\tInvalid format{Fore.WHITE}")
            return fightLookup(data, mlpPredictions, parameters, optimizedParameters, dampener, sharpness)

        # isolate names and date from input
        name1 = lookupString.split('vs')[0].strip()
        name2 = lookupString.split('vs')[1].split('(')[0].strip()
        dateStr = lookupString.split('(')[1].split(')')[0].strip()

        found = False
        totalCost = 0.0

        # check if fight exists in data
        for i in range(0, len(data['round']), 2):
            # check fighter names
            if (name1 == data['name'][i] and name2 == data['nameO'][i]) or (name2 == data['name'][i] and name1 == data['nameO'][i]):
                # adjust format of date strings
                inputDate = datetime.strptime(dateStr, "%m/%d/%Y")
                fightDate = datetime.strptime(data['date'][i], "%m/%d/%Y")

                # check if dates line up
                if inputDate == fightDate:
                    # if mlp is being used, get prediction from mlpPredictions
                    if mlpPredictions != None:
                        prediction = mlpPredictions[i]
                    # if gd is being used, calculate using heuristic()
                    else:
                        # isolate data for just this round
                        roundData = {}
                        for param in parameters:
                            if 'total' in param:
                                roundData[param] = [data[param][i]]
                            else:
                                roundData[param] = [data[param][i]]
                                roundData[param + 'O'] = [data[param + 'O'][i]]
                        # get prediction using heuristic()
                        prediction = heuristic(optimizedParameters, roundData, parameters, dampener, sharpness)[0]

                    # print out round by round details
                    mlp = (mlpPredictions != None)
                    totalCost += printRoundDetails(data, i, prediction, parameters, None, mlp, dampener, sharpness)
                    found = True
                else:
                    print(f"{Fore.RED}Found under different date{Fore.WHITE}")
        
        # print total cost for the fight if found
        if found:
            print(f"\nTotal Cost : {totalCost:.2f}\n")
        # if no match found, print error message
        else:
            print(f"{Fore.RED}ERROR:\tNo match found{Fore.WHITE}")