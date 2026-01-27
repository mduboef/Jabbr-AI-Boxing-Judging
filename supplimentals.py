import numpy as np
from colorama import Fore


def printScoreDistribution(fights):
    # count different types of scores and rounds
    totalRounds = 0
    tenNineRounds = 0
    tenTenRounds = 0
    deductionRounds = 0
    
    # count individual judge decisions
    totalJudgeDecisions = 0
    redCornerWins = 0
    blueCornerWins = 0
    drawDecisions = 0
    
    # track unique score combinations
    scoreComboCount = {}
    
    # count unanimous vs split decisions
    unanimousRounds = 0
    splitDecisionRounds = 0
    
    for fight in fights:
        # skip fights without proper score data
        if fight.rScore1 is None:
            continue
            
        # process each round that wasn't excluded
        for roundNum in range(len(fight.rScore1)):
            if roundNum + 1 not in fight.excludedRounds and fight.rScore1[roundNum] != '':
                totalRounds += 1
                
                # get scores for all three judges
                redScores = [fight.rScore1[roundNum], fight.rScore2[roundNum], fight.rScore3[roundNum]]
                blueScores = [fight.bScore1[roundNum], fight.bScore2[roundNum], fight.bScore3[roundNum]]
                
                # count individual judge decisions and categorize round type
                roundRedWins = 0
                roundBlueWins = 0
                roundDraws = 0
                
                # determine round category based on first judge's score
                redScore = int(fight.rScore1[roundNum])
                blueScore = int(fight.bScore1[roundNum])
                
                if (redScore == 10 and blueScore == 9) or (redScore == 9 and blueScore == 10):
                    tenNineRounds += 1
                elif redScore == 10 and blueScore == 10:
                    tenTenRounds += 1
                else:
                    deductionRounds += 1
                
                # analyze each judge's decision for this round
                for i in range(3):
                    if redScores[i] != '' and blueScores[i] != '':
                        totalJudgeDecisions += 1
                        
                        redScoreInt = int(redScores[i])
                        blueScoreInt = int(blueScores[i])
                        
                        if redScoreInt == blueScoreInt:
                            drawDecisions += 1
                            roundDraws += 1
                        elif redScoreInt > blueScoreInt:
                            redCornerWins += 1
                            roundRedWins += 1
                        else:
                            blueCornerWins += 1
                            roundBlueWins += 1
                
                # track score combinations for each judge (for interesting patterns)
                for i in range(3):
                    if redScores[i] != '' and blueScores[i] != '':
                        judgeRedScore = int(redScores[i])
                        judgeBlueScore = int(blueScores[i])
                        scoreCombo = f"{judgeRedScore}-{judgeBlueScore}"
                        scoreComboCount[scoreCombo] = scoreComboCount.get(scoreCombo, 0) + 1
                
                # determine if round was unanimous or split
                if roundRedWins == 3 or roundBlueWins == 3 or roundDraws == 3:
                    unanimousRounds += 1
                else:
                    splitDecisionRounds += 1
    
    # print comprehensive score distribution analysis
    print(f"{Fore.CYAN}Score Distribution Analysis{Fore.WHITE}")
    print("=" * 60)
    print(f"Total fights analyzed:\t\t\t{len([f for f in fights if f.rScore1 is not None])}")
    print(f"Total rounds analyzed:\t\t\t{totalRounds}")
    print()
    
    print("Round Categories:")
    print(f"\t10-9/9-10 rounds:\t\t{tenNineRounds:>6} ({100 * tenNineRounds / totalRounds:>5.1f}%)")
    print(f"\t10-10 rounds:\t\t\t{tenTenRounds:>6} ({100 * tenTenRounds / totalRounds:>5.1f}%)")
    print(f"\tRounds with deductions:\t\t{deductionRounds:>6} ({100 * deductionRounds / totalRounds:>5.1f}%)")
    print()
    
    print("Judge Decision Analysis:")
    print(f"\tTotal individual decisions:\t{totalJudgeDecisions:>6}")
    print(f"\tRed corner wins:\t\t{redCornerWins:>6} ({100 * redCornerWins / totalJudgeDecisions:>5.1f}%)")
    print(f"\tBlue corner wins:\t\t{blueCornerWins:>6} ({100 * blueCornerWins / totalJudgeDecisions:>5.1f}%)")
    print(f"\tDraw decisions:\t\t\t{drawDecisions:>6} ({100 * drawDecisions / totalJudgeDecisions:>5.1f}%)")
    print()
    
    print("Round Consensus:")
    print(f"\tUnanimous rounds:\t\t{unanimousRounds:>6} ({100 * unanimousRounds / totalRounds:>5.1f}%)")
    print(f"\tSplit decision rounds:\t\t{splitDecisionRounds:>6} ({100 * splitDecisionRounds / totalRounds:>5.1f}%)")
    print()
    
    # show most common score combinations
    print("Most Common Score Combinations:")
    sortedCombos = sorted(scoreComboCount.items(), key=lambda x: x[1], reverse=True)
    for combo, count in sortedCombos:
        percentage = 100 * count / (totalRounds*3)
        print(f"\t{combo}:\t\t\t\t{count:>6} ({percentage:>5.1f}%)")
    
    print()




# prints the average and standard deviation of all round-specific metrics
def getAverageStatValues(data):
    
    # specify the stats in data to analyze
    statsToAnalyze = [
        'totalThrown', 'totalLanded','thrownShare', 'landedShare',
        'thrown', 'landed', 'missed', 'accuracy', 'highImpact',
        'min', 'low', 'mid', 'high', 'max',
        'minMiss', 'lowMiss', 'midMiss', 'highMiss', 'maxMiss', 'lowCommitMiss', 'highCommitMiss',
        'pressure', 'pressureDistance', 'pressureMovement', 'pressurePosition',
        'aggression', 'aggressionCombinations', 'aggressionExchanges', 'aggressionPower',
        'singles', 'doubles', 'triples', 'quadsPlus',
        'outside', 'midrange', 'inside', 'clinch',
        'backfoot', 'neutral', 'frontfoot',
        'southpaw', 'squared', 'orthodox',
        'lead', 'rear', 'head', 'body',
        'straights', 'hooks', 'overhands', 'uppercuts'
    ]
    
    print(f"\nAverage Stat Values and Standard Deviations")
    print("=" * 58)

    # find the maximum length for formatting
    maxLength = max(len(stat) for stat in statsToAnalyze)
    
    # calculate and print stats for each parameter
    for stat in statsToAnalyze:
        if stat in data:
            values = np.array(data[stat])
            avgValue = np.mean(values)
            stdDev = np.std(values)
            
            print(f"{stat:<{maxLength}} : avg = {avgValue:>8.2f}    std = {stdDev:>8.2f}")
        else:
            print(f"{stat:<{maxLength}} : {Fore.RED}NOT FOUND IN DATA{Fore.WHITE}")
    
    print("=" * 58)


# prints the average and standard deviation of all round-specific metrics by judge score
def getStatValuesByScore(data):
    
    # specify the stats in data to analyze
    statsToAnalyze = [
        'totalThrown', 'totalLanded','thrownShare', 'landedShare',
        'thrown', 'landed', 'missed', 'accuracy', 'highImpact',
        'min', 'low', 'mid', 'high', 'max',
        'minMiss', 'lowMiss', 'midMiss', 'highMiss', 'maxMiss', 'lowCommitMiss', 'highCommitMiss',
        'pressure', 'pressureDistance', 'pressureMovement', 'pressurePosition',
        'aggression', 'aggressionCombinations', 'aggressionExchanges', 'aggressionPower',
        'singles', 'doubles', 'triples', 'quadsPlus',
        'outside', 'midrange', 'inside', 'clinch',
        'backfoot', 'neutral', 'frontfoot',
        'southpaw', 'squared', 'orthodox',
        'lead', 'rear', 'head', 'body',
        'straights', 'hooks', 'overhands', 'uppercuts'
    ]
    
    # organize data by score
    dataByScore = {
        -1: {stat: [] for stat in statsToAnalyze},
        -1/3: {stat: [] for stat in statsToAnalyze},
        1/3: {stat: [] for stat in statsToAnalyze},
        1: {stat: [] for stat in statsToAnalyze}
    }
    
    # populate data by score
    for i, score in enumerate(data['scores']):
        if score in dataByScore:
            for stat in statsToAnalyze:
                if stat in data:
                    dataByScore[score][stat].append(data[stat][i])
    
    print(f"\nAverage Stat Values by Judge Score")
    print("=" * 84)
    
    # find the maximum length for formatting
    maxLength = max(len(stat) for stat in statsToAnalyze)
    
    # print header
    print(f"{'Statistic':<{maxLength}} | {'0/3 Judges':<12} | {'1/3 Judges':<12} | {'2/3 Judges':<12} | {'3/3 Judges':<12}")
    print("-" * 84)
    
    # calculate and print stats for each parameter
    for stat in statsToAnalyze:
        if stat in data:
            print(f"{stat:<{maxLength}} | ", end="")
            
            # calculate averages for each score category
            for score in [-1, -1/3, 1/3, 1]:
                if len(dataByScore[score][stat]) > 0:
                    avgValue = np.mean(dataByScore[score][stat])
                    print(f"{avgValue:>10.2f}   | ", end="")
                else:
                    print(f"{'N/A':>10}   | ", end="")
            print()  # newline after each stat
        else:
            print(f"{stat:<{maxLength}} | {Fore.RED}NOT FOUND IN DATA{Fore.WHITE}")
    
    print("=" * 84)



def sortDataConsistently(data):
    """
    sort all data arrays by fighter names, date, and round for consistent ordering
    """
    # create list of indices with sort keys
    indexedRounds = []
    for i in range(0, len(data['scores']), 2):
        sortKey = (data['name'][i], data['nameO'][i], data['date'][i], data['round'][i])
        indexedRounds.append((sortKey, i))
    
    # sort by the sort key  
    indexedRounds.sort(key=lambda x: x[0])
    
    # create new sorted data dictionary
    sortedData = {}
    for key in data.keys():
        sortedData[key] = []
    
    # reorder all arrays according to sorted indices
    for _, originalIndex in indexedRounds:
        for key in data.keys():
            # add both fighter perspectives for this round
            sortedData[key].extend([data[key][originalIndex], data[key][originalIndex + 1]])
    
    return sortedData



# create a data dictionary from Fight objects without requiring score data
def fightToDataNoScores(fights):
    data = dict(
        # general info
        name=[],
        nameO=[],
        date=[],
        round=[],
        color=[],

        # total stats
        totalLanded=[],
        totalThrown=[],

        # fighter ratio stats
        thrownShare=[],    # thrown/totalThrown
        landedShare=[],    # landed/totalLanded

        # fighter stats
        thrown=[],
        landed=[],
        missed=[],
        accuracy=[],
        highImpact=[],
        pressure=[],
        pressureDistance=[],
        pressureMovement=[],
        pressurePosition=[],
        aggression=[],
        aggressionCombinations=[],
        aggressionExchanges=[],
        aggressionPower=[],
        min=[],
        low=[],
        mid=[],
        high=[],
        max=[],
        minMiss=[],
        lowMiss=[],
        midMiss=[],
        highMiss=[],
        maxMiss=[],
        lowCommitMiss=[],     # min & low missed
        highCommitMiss=[],    # med, high & max missed
        orthodox=[],
        southpaw=[],
        squared=[],
        # additional stats
        singles=[],
        doubles=[],
        triples=[],
        quadsPlus=[],
        outside=[],
        midrange=[],
        inside=[],
        clinch=[],
        backfoot=[],
        frontfoot=[],
        neutral=[],
        lead=[],
        rear=[],
        straights=[],
        hooks=[],
        overhands=[],
        uppercuts=[],
        head=[],
        body=[],

        # opponent ratio stats
        thrownShareO=[],    # thrown/totalThrown
        landedShareO=[],    # landed/totalLanded

        # opponent stats
        thrownO=[],
        landedO=[],
        missedO=[],
        accuracyO=[],
        highImpactO=[],
        pressureO=[],
        pressureDistanceO=[],
        pressureMovementO=[],
        pressurePositionO=[],
        aggressionO=[],
        aggressionCombinationsO=[],
        aggressionExchangesO=[],
        aggressionPowerO=[],
        minO=[],
        lowO=[],
        midO=[],
        highO=[],
        maxO=[],
        minMissO=[],
        lowMissO=[],
        midMissO=[],
        highMissO=[],
        maxMissO=[],
        lowCommitMissO=[],    # min & low missed
        highCommitMissO=[],   # med, high & max missed
        orthodoxO=[],
        southpawO=[],
        squaredO=[],
        # opponent additional stats
        singlesO=[],
        doublesO=[],
        triplesO=[],
        quadsPlusO=[],
        outsideO=[],
        midrangeO=[],
        insideO=[],
        clinchO=[],
        backfootO=[],
        frontfootO=[],
        neutralO=[],
        leadO=[],
        rearO=[],
        headO=[],
        bodyO=[],
        straightsO=[],
        hooksO=[],
        overhandsO=[],
        uppercutsO=[]
    )

    for fight in fights:
        # get the number of rounds for this fight
        numRounds = len(fight.pR.landed)
        
        for roundNum in range(numRounds):
            # store general info for both fighters
            data["name"].append(fight.red)
            data["name"].append(fight.blue)
            data["nameO"].append(fight.blue)
            data["nameO"].append(fight.red)
            data["date"].append(fight.date)
            data["date"].append(fight.date)
            data["round"].append(roundNum + 1)
            data["round"].append(roundNum + 1)
            data["color"].append("red")
            data["color"].append("blue")

            # store stats that apply to both fighters
            totalThrown = fight.pR.thrown[roundNum] + fight.pB.thrown[roundNum]
            data["totalThrown"].append(totalThrown)
            data["totalThrown"].append(totalThrown)
            totalLanded = fight.pR.landed[roundNum] + fight.pB.landed[roundNum]
            data["totalLanded"].append(totalLanded)
            data["totalLanded"].append(totalLanded)

            # store ratio stats for primary fighter
            tt = totalThrown if totalThrown != 0 else 1     # avoid division by zero
            tl = totalLanded if totalLanded != 0 else 1
            data["thrownShare"].append(fight.pR.thrown[roundNum] / tt)
            data["thrownShare"].append(fight.pB.thrown[roundNum] / tt)
            data["landedShare"].append(fight.pR.landed[roundNum] / tl)
            data["landedShare"].append(fight.pB.landed[roundNum] / tl)

            # store stats for primary fighter
            data["thrown"].append(fight.pR.thrown[roundNum])
            data["thrown"].append(fight.pB.thrown[roundNum])
            data["landed"].append(fight.pR.landed[roundNum])
            data["landed"].append(fight.pB.landed[roundNum])
            data["missed"].append(fight.pR.thrown[roundNum] - fight.pR.landed[roundNum])
            data["missed"].append(fight.pB.thrown[roundNum] - fight.pB.landed[roundNum])
            
            if fight.pR.thrown[roundNum] < 1:
                data["accuracy"].append(0)
            else:
                data["accuracy"].append(100 * (fight.pR.landed[roundNum] / fight.pR.thrown[roundNum]))
            if fight.pB.thrown[roundNum] < 1:
                data["accuracy"].append(0)
            else:
                data["accuracy"].append(100 * (fight.pB.landed[roundNum] / fight.pB.thrown[roundNum]))
            
            data["highImpact"].append(fight.pR.highImpact[roundNum])
            data["highImpact"].append(fight.pB.highImpact[roundNum])
            data["pressure"].append(fight.pR.pressure[roundNum])
            data["pressure"].append(fight.pB.pressure[roundNum])
            data["pressureDistance"].append(fight.pR.pressureDistance[roundNum])
            data["pressureDistance"].append(fight.pB.pressureDistance[roundNum])
            data["pressureMovement"].append(fight.pR.pressureMovement[roundNum])
            data["pressureMovement"].append(fight.pB.pressureMovement[roundNum])
            data["pressurePosition"].append(fight.pR.pressurePosition[roundNum])
            data["pressurePosition"].append(fight.pB.pressurePosition[roundNum])
            data["aggression"].append(fight.pR.aggression[roundNum])
            data["aggression"].append(fight.pB.aggression[roundNum])
            data["aggressionCombinations"].append(fight.pR.aggressionCombinations[roundNum])
            data["aggressionCombinations"].append(fight.pB.aggressionCombinations[roundNum])
            data["aggressionExchanges"].append(fight.pR.aggressionExchanges[roundNum])
            data["aggressionExchanges"].append(fight.pB.aggressionExchanges[roundNum])
            data["aggressionPower"].append(fight.pR.aggressionPower[roundNum])
            data["aggressionPower"].append(fight.pB.aggressionPower[roundNum])
            data["orthodox"].append(fight.pR.orthodox[roundNum])
            data["orthodox"].append(fight.pB.orthodox[roundNum])
            data["southpaw"].append(fight.pR.southpaw[roundNum])
            data["southpaw"].append(fight.pB.southpaw[roundNum])
            data["squared"].append(fight.pR.squared[roundNum])
            data["squared"].append(fight.pB.squared[roundNum])
            data["singles"].append(fight.pR.singles[roundNum])
            data["singles"].append(fight.pB.singles[roundNum])
            data["doubles"].append(fight.pR.doubles[roundNum])
            data["doubles"].append(fight.pB.doubles[roundNum])
            data["triples"].append(fight.pR.triples[roundNum])
            data["triples"].append(fight.pB.triples[roundNum])
            data["quadsPlus"].append(fight.pR.quadsPlus[roundNum])
            data["quadsPlus"].append(fight.pB.quadsPlus[roundNum])
            data["outside"].append(fight.pR.outside[roundNum])
            data["outside"].append(fight.pB.outside[roundNum])
            data["midrange"].append(fight.pR.midrange[roundNum])
            data["midrange"].append(fight.pB.midrange[roundNum])
            data["inside"].append(fight.pR.inside[roundNum])
            data["inside"].append(fight.pB.inside[roundNum])
            data["clinch"].append(fight.pR.clinch[roundNum])
            data["clinch"].append(fight.pB.clinch[roundNum])
            data["backfoot"].append(fight.pR.backfoot[roundNum])
            data["backfoot"].append(fight.pB.backfoot[roundNum])
            data["frontfoot"].append(fight.pR.frontfoot[roundNum])
            data["frontfoot"].append(fight.pB.frontfoot[roundNum])
            data["neutral"].append(fight.pR.neutral[roundNum])
            data["neutral"].append(fight.pB.neutral[roundNum])
            data["lead"].append(fight.pR.lead[roundNum])
            data["lead"].append(fight.pB.lead[roundNum])
            data["rear"].append(fight.pR.rear[roundNum])
            data["rear"].append(fight.pB.rear[roundNum])
            data["head"].append(fight.pR.head[roundNum])
            data["head"].append(fight.pB.head[roundNum])
            data["body"].append(fight.pR.body[roundNum])
            data["body"].append(fight.pB.body[roundNum])
            data["straights"].append(fight.pR.straights[roundNum])
            data["straights"].append(fight.pB.straights[roundNum])
            data["hooks"].append(fight.pR.hooks[roundNum])
            data["hooks"].append(fight.pB.hooks[roundNum])
            data["overhands"].append(fight.pR.overhands[roundNum])
            data["overhands"].append(fight.pB.overhands[roundNum])
            data["uppercuts"].append(fight.pR.uppercuts[roundNum])
            data["uppercuts"].append(fight.pB.uppercuts[roundNum])

            data["min"].append(fight.pR.min[roundNum])
            data["min"].append(fight.pB.min[roundNum])
            data["low"].append(fight.pR.low[roundNum])
            data["low"].append(fight.pB.low[roundNum])
            data["mid"].append(fight.pR.mid[roundNum])
            data["mid"].append(fight.pB.mid[roundNum])
            data["high"].append(fight.pR.high[roundNum])
            data["high"].append(fight.pB.high[roundNum])
            data["max"].append(fight.pR.max[roundNum])
            data["max"].append(fight.pB.max[roundNum])
            data["minMiss"].append(fight.pR.minMiss[roundNum])
            data["minMiss"].append(fight.pB.minMiss[roundNum])
            data["lowMiss"].append(fight.pR.lowMiss[roundNum])
            data["lowMiss"].append(fight.pB.lowMiss[roundNum])
            data["midMiss"].append(fight.pR.midMiss[roundNum])
            data["midMiss"].append(fight.pB.midMiss[roundNum])
            data["highMiss"].append(fight.pR.highMiss[roundNum])
            data["highMiss"].append(fight.pB.highMiss[roundNum])
            data["maxMiss"].append(fight.pR.maxMiss[roundNum])
            data["maxMiss"].append(fight.pB.maxMiss[roundNum])
            data["lowCommitMiss"].append(fight.pR.minMiss[roundNum] + fight.pR.lowMiss[roundNum])
            data["lowCommitMiss"].append(fight.pB.minMiss[roundNum] + fight.pB.lowMiss[roundNum])
            data["highCommitMiss"].append(fight.pR.midMiss[roundNum] + fight.pR.highMiss[roundNum] + fight.pR.maxMiss[roundNum])
            data["highCommitMiss"].append(fight.pB.midMiss[roundNum] + fight.pB.highMiss[roundNum] + fight.pB.maxMiss[roundNum])

            # store ratio stats for opponent fighter
            data["thrownShareO"].append(fight.pB.thrown[roundNum] / tt)
            data["thrownShareO"].append(fight.pR.thrown[roundNum] / tt)
            data["landedShareO"].append(fight.pB.landed[roundNum] / tl)
            data["landedShareO"].append(fight.pR.landed[roundNum] / tl)

            # store stats for opponent fighter
            data["thrownO"].append(fight.pB.thrown[roundNum])
            data["thrownO"].append(fight.pR.thrown[roundNum])
            data["landedO"].append(fight.pB.landed[roundNum])
            data["landedO"].append(fight.pR.landed[roundNum])
            data["missedO"].append(fight.pB.thrown[roundNum] - fight.pB.landed[roundNum])
            data["missedO"].append(fight.pR.thrown[roundNum] - fight.pR.landed[roundNum])
            
            if fight.pB.thrown[roundNum] < 1:
                data["accuracyO"].append(0)
            else:
                data["accuracyO"].append(100 * (fight.pB.landed[roundNum] / fight.pB.thrown[roundNum]))
            if fight.pR.thrown[roundNum] < 1:
                data["accuracyO"].append(0)
            else:
                data["accuracyO"].append(100 * (fight.pR.landed[roundNum] / fight.pR.thrown[roundNum]))
            
            data["highImpactO"].append(fight.pB.highImpact[roundNum])
            data["highImpactO"].append(fight.pR.highImpact[roundNum])
            data["pressureO"].append(fight.pB.pressure[roundNum])
            data["pressureO"].append(fight.pR.pressure[roundNum])
            data["pressureDistanceO"].append(fight.pB.pressureDistance[roundNum])
            data["pressureDistanceO"].append(fight.pR.pressureDistance[roundNum])
            data["pressureMovementO"].append(fight.pB.pressureMovement[roundNum])
            data["pressureMovementO"].append(fight.pR.pressureMovement[roundNum])
            data["pressurePositionO"].append(fight.pB.pressurePosition[roundNum])
            data["pressurePositionO"].append(fight.pR.pressurePosition[roundNum])
            data["aggressionO"].append(fight.pB.aggression[roundNum])
            data["aggressionO"].append(fight.pR.aggression[roundNum])
            data["aggressionCombinationsO"].append(fight.pB.aggressionCombinations[roundNum])
            data["aggressionCombinationsO"].append(fight.pR.aggressionCombinations[roundNum])
            data["aggressionExchangesO"].append(fight.pB.aggressionExchanges[roundNum])
            data["aggressionExchangesO"].append(fight.pR.aggressionExchanges[roundNum])
            data["aggressionPowerO"].append(fight.pB.aggressionPower[roundNum])
            data["aggressionPowerO"].append(fight.pR.aggressionPower[roundNum])
            data["orthodoxO"].append(fight.pB.orthodox[roundNum])
            data["orthodoxO"].append(fight.pR.orthodox[roundNum])
            data["southpawO"].append(fight.pB.southpaw[roundNum])
            data["southpawO"].append(fight.pR.southpaw[roundNum])
            data["squaredO"].append(fight.pB.squared[roundNum])
            data["squaredO"].append(fight.pR.squared[roundNum])
            data["singlesO"].append(fight.pB.singles[roundNum])
            data["singlesO"].append(fight.pR.singles[roundNum])
            data["doublesO"].append(fight.pB.doubles[roundNum])
            data["doublesO"].append(fight.pR.doubles[roundNum])
            data["triplesO"].append(fight.pB.triples[roundNum])
            data["triplesO"].append(fight.pR.triples[roundNum])
            data["quadsPlusO"].append(fight.pB.quadsPlus[roundNum])
            data["quadsPlusO"].append(fight.pR.quadsPlus[roundNum])
            data["outsideO"].append(fight.pB.outside[roundNum])
            data["outsideO"].append(fight.pR.outside[roundNum])
            data["midrangeO"].append(fight.pB.midrange[roundNum])
            data["midrangeO"].append(fight.pR.midrange[roundNum])
            data["insideO"].append(fight.pB.inside[roundNum])
            data["insideO"].append(fight.pR.inside[roundNum])
            data["clinchO"].append(fight.pB.clinch[roundNum])
            data["clinchO"].append(fight.pR.clinch[roundNum])
            data["backfootO"].append(fight.pB.backfoot[roundNum])
            data["backfootO"].append(fight.pR.backfoot[roundNum])
            data["frontfootO"].append(fight.pB.frontfoot[roundNum])
            data["frontfootO"].append(fight.pR.frontfoot[roundNum])
            data["neutralO"].append(fight.pB.neutral[roundNum])
            data["neutralO"].append(fight.pR.neutral[roundNum])
            data["leadO"].append(fight.pB.lead[roundNum])
            data["leadO"].append(fight.pR.lead[roundNum])
            data["rearO"].append(fight.pB.rear[roundNum])
            data["rearO"].append(fight.pR.rear[roundNum])
            data["headO"].append(fight.pB.head[roundNum])
            data["headO"].append(fight.pR.head[roundNum])
            data["bodyO"].append(fight.pB.body[roundNum])
            data["bodyO"].append(fight.pR.body[roundNum])
            data["straightsO"].append(fight.pB.straights[roundNum])
            data["straightsO"].append(fight.pR.straights[roundNum])
            data["hooksO"].append(fight.pB.hooks[roundNum])
            data["hooksO"].append(fight.pR.hooks[roundNum])
            data["overhandsO"].append(fight.pB.overhands[roundNum])
            data["overhandsO"].append(fight.pR.overhands[roundNum])
            data["uppercutsO"].append(fight.pB.uppercuts[roundNum])
            data["uppercutsO"].append(fight.pR.uppercuts[roundNum])

            data["minO"].append(fight.pB.min[roundNum])
            data["minO"].append(fight.pR.min[roundNum])
            data["lowO"].append(fight.pB.low[roundNum])
            data["lowO"].append(fight.pR.low[roundNum])
            data["midO"].append(fight.pB.mid[roundNum])
            data["midO"].append(fight.pR.mid[roundNum])
            data["highO"].append(fight.pB.high[roundNum])
            data["highO"].append(fight.pR.high[roundNum])
            data["maxO"].append(fight.pB.max[roundNum])
            data["maxO"].append(fight.pR.max[roundNum])
            data["minMissO"].append(fight.pB.minMiss[roundNum])
            data["minMissO"].append(fight.pR.minMiss[roundNum])
            data["lowMissO"].append(fight.pB.lowMiss[roundNum])
            data["lowMissO"].append(fight.pR.lowMiss[roundNum])
            data["midMissO"].append(fight.pB.midMiss[roundNum])
            data["midMissO"].append(fight.pR.midMiss[roundNum])
            data["highMissO"].append(fight.pB.highMiss[roundNum])
            data["highMissO"].append(fight.pR.highMiss[roundNum])
            data["maxMissO"].append(fight.pB.maxMiss[roundNum])
            data["maxMissO"].append(fight.pR.maxMiss[roundNum])
            data["lowCommitMissO"].append(fight.pB.minMiss[roundNum] + fight.pB.lowMiss[roundNum])
            data["lowCommitMissO"].append(fight.pR.minMiss[roundNum] + fight.pR.lowMiss[roundNum])
            data["highCommitMissO"].append(fight.pB.midMiss[roundNum] + fight.pB.highMiss[roundNum] + fight.pB.maxMiss[roundNum])
            data["highCommitMissO"].append(fight.pR.midMiss[roundNum] + fight.pR.highMiss[roundNum] + fight.pR.maxMiss[roundNum])

    # validate that all arrays have the same length
    for key, value in data.items():
        if len(value) != len(data["round"]):
            print(f"{Fore.RED}ERROR: Unequal list length for key: {key} ({len(value)} != {len(data['round'])}){Fore.WHITE}")

    return data







def printFightRoundCounts(data):
    # group rounds by fight using name, opponent name, and date as unique identifiers
    fightRounds = {}
    
    # iterate through data (every 2 entries represent one round from both fighter perspectives)
    for i in range(0, len(data['round']), 2):
        fighterName = data['name'][i]
        opponentName = data['nameO'][i] 
        fightDate = data['date'][i]
        roundNumber = data['round'][i]
        
        # create unique fight identifier
        fightKey = f"{fighterName} vs {opponentName} ({fightDate})"
        
        # track rounds for this fight
        if fightKey not in fightRounds:
            fightRounds[fightKey] = set()
        
        fightRounds[fightKey].add(roundNumber)
    
    # sort fights by name for consistent output
    sortedFights = sorted(fightRounds.items())
    
    print(f"\nFight Round Counts ({len(sortedFights)} fights, {len(data['round'])//2} total rounds):")
    print("=" * 80)
    print(f"{'Fight':<60} | {'Number of rounds'}")
    print("-" * 80)
    
    totalRounds = 0
    for fightKey, rounds in sortedFights:
        numRounds = len(rounds)
        totalRounds += numRounds
        print(f"{fightKey:<60} | {numRounds}")
    
    print("-" * 80)
    print(f"{'Total':<60} | {totalRounds}")
    print("=" * 80)