import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress all TF messages except errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # disable oneDNN optimizations

import sys
import numpy as np
from colorama import Fore
from dataProcessing import readStats, readCards, pairFightsToCards, getTenTenCount, fightToData, getPMs, normalizeData
from gradientDescent import costCalc, gradientDescent, combos, comboStart
from mlp import mlp, dictSplit
from graphing import plotCorrMatrix, plotScatters, plotHistograms, plotNormalHistograms
from printing import gradeRounds, gradeFights, printAccuracy, printValues, printStanceWinRate, printCorrelations, judgeFrequency, rankJudges, pairwiseComparison, fightLookup, countUniqueJudges
from supplimentals import getAverageStatValues, getStatValuesByScore, printScoreDistribution, sortDataConsistently, fightToDataNoScores, printFightRoundCounts
from l1Log import runL1Analysis, compareL1ToOtherMethods
from validation import readQuadStats, excludeSpecificRounds



# calculate and return the 10 fights with highest total impact
def getHighestImpactFights(data):
    
    # impact coefficients
    aCoeff = 0.092  # min coefficient
    bCoeff = 0.091  # low coefficient  
    cCoeff = 0.147  # mid coefficient
    dCoeff = 0.266  # high coefficient
    eCoeff = 0.513  # max coefficient
    
    # dictionary to store fight impacts
    fightImpacts = {}
    
    # iterate through data (every 2 entries represent one round from both fighter perspectives)
    for i in range(0, len(data['round']), 2):
        fighter1Name = data['name'][i]
        fighter2Name = data['nameO'][i] 
        fightDate = data['date'][i]
        roundNumber = data['round'][i]
        
        # create unique fight identifier
        fightKey = f"{fighter1Name} vs {fighter2Name} ({fightDate})"
        
        # calculate impact for fighter 1 (index i)
        fighter1Impact = (aCoeff * data['min'][i] + 
                         bCoeff * data['low'][i] + 
                         cCoeff * data['mid'][i] + 
                         dCoeff * data['high'][i] + 
                         eCoeff * data['max'][i])
        
        # calculate impact for fighter 2 (index i+1)  
        fighter2Impact = (aCoeff * data['min'][i+1] + 
                         bCoeff * data['low'][i+1] + 
                         cCoeff * data['mid'][i+1] + 
                         dCoeff * data['high'][i+1] + 
                         eCoeff * data['max'][i+1])
        
        # total round impact
        roundTotalImpact = fighter1Impact + fighter2Impact
        
        # add to fight total impact
        if fightKey not in fightImpacts:
            fightImpacts[fightKey] = {'totalImpact': 0.0, 'roundCount': 0}
        
        fightImpacts[fightKey]['totalImpact'] += roundTotalImpact
        fightImpacts[fightKey]['roundCount'] += 1
    
    # convert to list of tuples for sorting
    fightList = []
    for fightName, fightData in fightImpacts.items():
        fightList.append((fightName, fightData['totalImpact'], fightData['roundCount']))
    
    # sort by total impact in descending order
    sortedFights = sorted(fightList, key=lambda x: x[1], reverse=True)
    
    # return top 10 fights
    return sortedFights[:10]


# print the top 10 highest impact fights with details
def printHighestImpactFights(data):
    
    topFights = getHighestImpactFights(data)
    
    print(f"\nTop 10 Fights by Total Impact:")
    print("=" * 90)
    print(f"{'Rank':<4} | {'Fight':<50} | {'Total Impact':<12} | {'Rounds':<6}")
    print("-" * 90)
    
    for rank, (fightName, totalImpact, roundCount) in enumerate(topFights, 1):
        print(f"{rank:<4} | {fightName:<50} | {totalImpact:>10.2f}   | {roundCount:<6}")
    
    print("=" * 90)
    
    # count total unique fights
    uniqueFights = set()
    for i in range(0, len(data['round']), 2):
        fightKey = data['name'][i] + ' vs ' + data['nameO'][i] + ' (' + data['date'][i] + ')'
        uniqueFights.add(fightKey)
    
    print(f"Total fights analyzed: {len(uniqueFights)}")
    print(f"Impact formula: 0.092*min + 0.091*low + 0.147*mid + 0.266*high + 0.513*max")


def main():


    # new mapping parameters for the ratio-based scoring equation
    dampener = 150.0  # dampening parameter D - can be made configurable via command line
    sharpness = 9.0   # sharpness parameter S - can be made configurable via command line
    includeMappingParams = False  # whether to optimize D and S during gradient descent
    
    # check for new command line flags for mapping parameters
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '-dampener' or sys.argv[i] == '-d':
            try:
                dampener = float(sys.argv[i+1])
                if dampener <= 0:
                    print(f"{Fore.RED}ERROR:\tDampener must be positive{Fore.WHITE}")
                    return
            except (ValueError, IndexError):
                print(f"{Fore.RED}ERROR:\tDampener must be a valid positive number{Fore.WHITE}")
                return
        
        elif sys.argv[i] == '-sharpness' or sys.argv[i] == '-s':
            try:
                sharpness = float(sys.argv[i+1])
                if sharpness <= 0:
                    print(f"{Fore.RED}ERROR:\tSharpness must be positive{Fore.WHITE}")
                    return
            except (ValueError, IndexError):
                print(f"{Fore.RED}ERROR:\tSharpness must be a valid positive number{Fore.WHITE}")
                return
        
        elif sys.argv[i] == '-optimizemapping':
            includeMappingParams = True
    
    print(f"Using mapping parameters - Dampener (D): {dampener}, Sharpness (S): {sharpness}")
    if includeMappingParams:
        print("Mapping parameters will be optimized during gradient descent")
    else:
        print("Mapping parameters are fixed")



    # check for -quadcam flag
    useQuadcam = '-quadcam' in sys.argv
    useSinglecam = '-singlecam' in sys.argv
    ignoreScores = '-ignorescores' in sys.argv

    # validate flag combinations
    if ignoreScores and not (useQuadcam or useSinglecam):
        print(f"{Fore.RED}ERROR: -ignorescores flag can only be used with -quadcam or -singlecam{Fore.WHITE}")
        return

    # create list of Fight objects
    if useQuadcam:                                  # get quad-cam stats from json files
        fights = readQuadStats(sys.argv)
        print("\nCompleted:\tQuad-cam stats imported\n")
    else:                                           # get single-cam stats from Excel files (default)
        fights = readStats(sys.argv)
        print("\nCompleted:\tSingle-cam stats imported\n")

    # if -ignorescores is used, skip scorecard processing
    # This is for comparing quadcam stats to singlecam stats (validation)
    if ignoreScores:
        print(f"{Fore.YELLOW}INFO: Ignoring scorecard data as requested{Fore.WHITE}")
        
        # exclude specific rounds for single-cam analysis
        excludeSpecificRounds(fights, ignoreScores, useSinglecam)
        
        # create data dictionary without scores
        data = fightToDataNoScores(fights)
        print(f"\nCompleted:\tData conversion without scores (using {len(fights)} fights)\n")
        
        # only run analyses that don't require score data
        parameters = ['min', 'low', 'mid', 'high', 'max', 'accuracy',
                    'minMiss', 'lowMiss', 'midMiss', 'highMiss', 'maxMiss',
                    'pressureDistance', 'pressureMovement', 'pressurePosition',
                    'aggressionCombinations', 'aggressionExchanges', 'aggressionPower',
                    'singles', 'doubles', 'triples', 'quadsPlus',
                    'outside', 'midrange', 'inside', 'clinch',
                    'backfoot', 'neutral', 'frontfoot',
                    'southpaw', 'squared', 'orthodox',
                    'lead', 'rear', 'head', 'body',
                    'straights', 'hooks', 'overhands', 'uppercuts']
        
        # get average stat values & standard deviations
        getAverageStatValues(data)
        
        # plot a correlation matrix of the stats
        plotCorrMatrix(data, parameters)

        printFightRoundCounts(data)
        
        print(f"\nCompleted:\tAnalysis without scores - used {len(fights)} fights, {int(len(data['round'])/2)} rounds")
        return

    # normal flow with scorecard data
    # create list of Card objects
    cards = readCards(useQuadcam, useSinglecam)

    fights = pairFightsToCards(fights, cards)
    print("\nCompleted:\tScores imported\n")

    printScoreDistribution(fights)

    data = dict(         # twice values per round, 1 for each fighter
        # general info
        name = [],
        nameO = [],
        date = [],
        round = [],
        color = [],
        scores = [],        # combined scores for all 3 judges

        # judge specific scores
        score1 = [],
        score2 = [],
        score3 = [],

        # judges names
        judge1 = [],
        judge2 = [],
        judge3 = [],

        # total stats
        totalLanded = [],
        totalThrown = [],

        # fighter ratio stats
        thrownShare = [],    # thrown/totalThrown
        landedShare = [],    # landed/totalLanded

        # fighter stats
        thrown = [],
        landed = [],
        missed = [],
        accuracy = [],
        highImpact = [],
        pressure = [],
        pressureDistance = [],
        pressureMovement = [],
        pressurePosition = [],
        aggression = [],
        aggressionCombinations = [],
        aggressionExchanges = [],
        aggressionPower = [],
        min = [],
        low = [],
        mid = [],
        high = [],
        max = [],
        minMiss = [],
        lowMiss = [],
        midMiss = [],
        highMiss = [],
        maxMiss = [],
        lowCommitMiss = [],     # min & low missed
        highCommitMiss = [],    # med, high & max missed
        orthodox = [],
        southpaw = [],
        squared = [],
        # addtional states
        singles = [],
        doubles = [],
        triples = [],
        quadsPlus = [],
        outside = [],
        midrange = [],
        inside = [],
        clinch = [],
        backfoot = [],
        frontfoot = [],
        neutral = [],
        lead = [],
        rear = [],
        straights = [],
        hooks = [],
        overhands = [],
        uppercuts = [],
        head = [],
        body = [],

        # opponent ratio stats
        thrownShareO = [],    # thrown/totalThrown
        landedShareO = [],    # landed/totalLanded

        # opponent stats
        thrownO = [],
        landedO = [],
        missedO = [],
        accuracyO = [],
        highImpactO = [],
        pressureO = [],
        pressureDistanceO = [],
        pressureMovementO = [],
        pressurePositionO = [],
        aggressionO = [],
        aggressionCombinationsO = [],
        aggressionExchangesO = [],
        aggressionPowerO = [],
        minO = [],
        lowO = [],
        midO = [],
        highO = [],
        maxO = [],
        minMissO = [],
        lowMissO = [],
        midMissO = [],
        highMissO = [],
        maxMissO = [],
        lowCommitMissO = [],    # min & low missed
        highCommitMissO = [],   # med, high & max missed
        orthodoxO = [],
        southpawO = [],
        squaredO = [],
        # opponent addtional states
        singlesO = [],
        doublesO = [],
        triplesO = [],
        quadsPlusO = [],
        outsideO = [],
        midrangeO = [],
        insideO = [],
        clinchO = [],
        backfootO = [],
        frontfootO = [],
        neutralO = [],
        leadO = [],
        rearO = [],
        headO = [],
        bodyO = [],
        straightsO = [],
        hooksO = [],
        overhandsO = [],
        uppercutsO = []
    )

    # stores name of judge you want data for
    judgeName = None

    # optional threshold for judge accuracy percentages
    disagreeThreshold = 0.0     # if abs(predictions) < threshold round wont be considered

    # stores ratio of data assigned to testing set (also applies to validation set in mlp)
    testSplit = 0.0
    validationSplit = 0.0

    # stores random seed for reproducible data split
    randomSeed = 42  # default seed

    # stores shrinkage parameter for Empirical Bayes shrinkage in rankings
    shrinkageK = None  # default no shrinkage

    # set disagreeThreshold, judgeName & testSplit based on command line args
    for i in range(1, len(sys.argv)):

        if sys.argv[i] == '-j':
            judgeName = sys.argv[i+1]

        elif sys.argv[i] == '-dt':
            disagreeThreshold = float(sys.argv[i+1])
            # make sure dt value is valid
            if disagreeThreshold < 0 or disagreeThreshold > 1:
                print(f"{Fore.RED}ERROR:\tDisagreement threshold must be between 0 and 1{Fore.WHITE}")
                return

        elif sys.argv[i] == '-split':
            testSplit = float(sys.argv[i+1])
            
            # check if there's a second value for validation split
            if i+2 < len(sys.argv) and not sys.argv[i+2].startswith('-'):
                validationSplit = float(sys.argv[i+2])
                        
            # make sure split values are valid
            if testSplit < 0 or testSplit > 1:
                print(f"{Fore.RED}ERROR:\tRatio of data assigned to testing set must be between 0 and 1{Fore.WHITE}")
                return
            if validationSplit < 0 or validationSplit > 1:
                print(f"{Fore.RED}ERROR:\tRatio of data assigned to validation set must be between 0 and 1{Fore.WHITE}")
                return
            if testSplit + validationSplit >= 1:
                print(f"{Fore.RED}ERROR:\tTest and validation ratios cannot sum to 1 or more{Fore.WHITE}")
                return

        elif sys.argv[i] == '-seed':
            try:
                randomSeed = int(sys.argv[i+1])
                if randomSeed < 0:
                    print(f"{Fore.RED}ERROR:\tSeed must be a non-negative integer{Fore.WHITE}")
                    return
            except (ValueError, IndexError):
                print(f"{Fore.RED}ERROR:\tSeed must be a valid integer{Fore.WHITE}")
                return

        elif sys.argv[i] == '-shrink':
            try:
                # check if user wants optimal k via Method of Moments
                if sys.argv[i+1].lower() == 'opt':
                    shrinkageK = 'opt'
                else:
                    # parse as numeric k value
                    shrinkageK = float(sys.argv[i+1])
                    if shrinkageK <= 0:
                        print(f"{Fore.RED}ERROR:\tShrinkage parameter must be positive{Fore.WHITE}")
                        return
            except (ValueError, IndexError):
                print(f"{Fore.RED}ERROR:\tShrinkage parameter must be a valid number or 'opt'{Fore.WHITE}")
                return


    # set global random seed for reproducibible data split
    np.random.seed(randomSeed)
    print(f"Using random seed: {randomSeed}")

    # print shrinkage parameter if set
    if shrinkageK is not None:
        print(f"Using Empirical Bayes shrinkage with k = {shrinkageK}")


    # all data
    if judgeName == None:
        for i in range(0, len(fights)):
            fightToData(fights[i], data, None)

    # only for specified judge
    else:
        for i in range(0, len(fights)):
            if judgeName == fights[i].judge1:
                fightToData(fights[i], data, 1)
            elif judgeName == fights[i].judge2:
                fightToData(fights[i], data, 2)
            elif judgeName == fights[i].judge3:
                fightToData(fights[i], data, 3)

    if useQuadcam or useSinglecam:
        data = sortDataConsistently(data)           # data needs to be sorted if we want to compare single & quad via random comparison


    # split data into training & testing sets for gradient descent
    if testSplit != 0.0 and '-mlp' not in sys.argv:
        trainingRatio = 1 - testSplit - validationSplit
        training, testing, validation = dictSplit(data, trainingRatio, testSplit, validationSplit, randomSeed=randomSeed)


    # best gd weights for 39 parameters
    # avg training cost: 0.3802
    # avg testing cost: 0.36961
    bestValues = {
        'min'                    : -0.0129120,
        'low'                    : 0.0264717,
        'mid'                    : 0.0728543,
        'high'                   : 0.1707474,
        'max'                    : 0.3067023,
        'accuracy'               : 0.0480160,
        'minMiss'                : 0.0358430,
        'lowMiss'                : 0.0021280,
        'midMiss'                : 0.0398667,
        'highMiss'               : 0.0127287,
        'maxMiss'                : -0.0827359,
        'pressureDistance'       : 0.0090346,
        'pressureMovement'       : -0.0026960,
        'pressurePosition'       : 0.0014184,
        'aggressionCombinations' : -0.0377719,
        'aggressionExchanges'    : -0.0052481,
        'aggressionPower'        : 0.1072505,
        'singles'                : 0.0411366,
        'doubles'                : 0.0473676,
        'triples'                : 0.0579751,
        'quadsPlus'              : 0.0591963,
        'outside'                : 0.1158242,
        'midrange'               : 0.1494002,
        'inside'                 : 0.0461852,
        'clinch'                 : 0.1252064,
        'backfoot'               : 0.7826371,
        'neutral'                : 0.8506273,
        'frontfoot'              : 0.7658356,
        'southpaw'               : -0.1728379,
        'squared'                : -0.1831820,
        'orthodox'               : -0.1762143,
        'lead'                   : 0.0875230,
        'rear'                   : 0.0992274,
        'head'                   : -0.1856465,
        'body'                   : -0.2276947,
        'straights'              : 0.0922259,
        'hooks'                  : 0.1123113,
        'overhands'              : 0.0117116,
        'uppercuts'              : 0.0858006,
    }


    # best weights for 39 param new formula
    # S = 8.0
    # D = 50.0
    bestValues = {
        "min"                    : -1.8603244,
        "low"                    : -0.6993520,
        "mid"                    : 1.1096152,
        "high"                   : 4.4485583,
        "max"                    : 9.8109060,
        "accuracy"               : 0.9480478,
        "minMiss"                : 0.7956453,
        "lowMiss"                : -0.1748967,
        "midMiss"                : 0.8617502,
        "highMiss"               : -0.2070899,
        "maxMiss"                : -3.0540421,
        "pressureDistance"       : 0.9023145,
        "pressureMovement"       : 0.0005613,
        "pressurePosition"       : 0.0397009,
        "aggressionCombinations" : -1.0257037,
        "aggressionExchanges"    : -0.1118902,
        "aggressionPower"        : 3.7195118,
        "singles"                : -0.0673367,
        "doubles"                : 0.0976586,
        "triples"                : 0.3449049,
        "quadsPlus"              : 0.3680929,
        "outside"                : 0.1890353,
        "midrange"               : 1.3274392,
        "inside"                 : -1.9952067,
        "clinch"                 : 0.5603484,
        "backfoot"               : -0.6363375,
        "neutral"                : 1.4720695,
        "frontfoot"              : -1.2419811,
        "southpaw"               : -0.2119466,
        "squared"                : -0.4130478,
        "orthodox"               : -0.3329574,
        "lead"                   : 0.6077495,
        "rear"                   : 0.5964239,
        "head"                   : 0.1467953,
        "body"                   : -1.1962542,
        "straights"              : 1.3427060,
        "hooks"                  : 2.4635050,
        "overhands"              : -2.1919277,
        "uppercuts"              : 1.5405886,
    }


    # # d = 1.0
    # # s = 8.78837
    # bestValues = {
    #     "landedShare"     : 7.2041973,
    #     "head"            : 0.1312695,
    #     "aggressionPower" : 0.4606006,
    #     "highImpact"      : 1.2812556,
    #     "neutral"         : 0.4287188,
    #     "mid"             : 0.4479602,
    # }


    # # best values with weights rescaled in accordance with quadcam data
    # bestValues = {
    #     'min'                    : -0.009411079,
    #     'low'                    : 0.020741895,
    #     'mid'                    : 0.054539726,
    #     'high'                   : 0.107304559,
    #     'max'                    : 0.141083058,
    #     'accuracy'               : 0.037948346,
    #     'minMiss'                : 0.047563289,
    #     'lowMiss'                : 0.002417155,
    #     'midMiss'                : 0.042479752,
    #     'highMiss'               : 0.011651997,
    #     'maxMiss'                : -0.031921725,
    #     'pressureDistance'       : 0.010202571,
    #     'pressureMovement'       : -0.003339729,
    #     'pressurePosition'       : 0.001481323,
    #     'aggressionCombinations' : -0.033037334,
    #     'aggressionExchanges'    : -0.004634001,
    #     'aggressionPower'        : 0.082526086,
    #     'singles'                : 0.043401143,
    #     'doubles'                : 0.047719645,
    #     'triples'                : 0.052842397,
    #     'quadsPlus'              : 0.050429351,
    #     'outside'                : 0.115732407,
    #     'midrange'               : 0.162763366,
    #     'inside'                 : 0.045361934,
    #     'clinch'                 : 0.120198144,
    #     'backfoot'               : 0.927868727,
    #     'neutral'                : 0.884484275,
    #     'frontfoot'              : 0.59013845,
    #     'southpaw'               : -0.177511108,
    #     'squared'                : -0.204512693,
    #     'orthodox'               : -0.173824217,
    #     'lead'                   : 0.06592555,
    #     'rear'                   : 0.070116165,
    #     'head'                   : -0.14275753,
    #     'body'                   : -0.130790689,
    #     'straights'              : 0.07080815,
    #     'hooks'                  : 0.075105579,
    #     'overhands'              : 0.014522384,
    #     'uppercuts'              : 0.063754613,
    # }



    # # best gd weights for 16 parameters
    # # avg cost: 0.40182
    # bestValues = {
    #     'min': -0.0153518,
    #     'low': 0.0208653,
    #     'mid': 0.0865790,
    #     'high': 0.1815607,
    #     'max': 0.3181467,
    #     'accuracy': 0.0425241,
    #     'minMiss': 0.0259915,
    #     'lowMiss': -0.0036636,
    #     'midMiss': 0.0322325,
    #     'highMiss': 0.0167854,
    #     'maxMiss': -0.1380295,
    #     'pressureDistance': -0.0201412,
    #     'pressureMovement': -0.0022465,
    #     'pressurePosition': 0.0155406,
    #     'aggressionCombinations': 0.0073154,
    #     'aggressionExchanges': -0.0169417,
    #     'aggressionPower': 0.0841620,
    # }

    # # best gd weights for 12 parameters
    # # avg cost: 0.41099
    # # TODO if using -best params should be key values of bestValues
    # bestValues = {
    #     'min': 0.0308119,
    #     'low': 0.0522517,
    #     'mid': 0.1194264,
    #     'high': 0.2067649,
    #     'max': 0.3527453,
    #     'accuracy': 0.0182009,
    #     'pressureDistance': -0.0151177,
    #     'pressureMovement': 0.0021346,
    #     'pressurePosition': 0.0122336,
    #     'aggressionCombinations': 0.0135037,
    #     'aggressionExchanges': -0.0073459,
    #     'aggressionPower': 0.0796084,
    # }



    # specify stats that should be graphed
    allParams = ['min', 'low', 'mid', 'high', 'max', 'highImpact',
                  'minMiss', 'lowMiss', 'midMiss', 'highMiss', 'maxMiss', 'lowCommitMiss', 'highCommitMiss',
                  'pressure', 'pressureDistance', 'pressureMovement', 'pressurePosition',
                  'aggression', 'aggressionCombinations', 'aggressionExchanges', 'aggressionPower',
                  'singles', 'doubles', 'triples', 'quadsPlus',
                  'outside', 'midrange', 'inside', 'clinch',
                  'backfoot', 'neutral', 'frontfoot',
                  'southpaw', 'squared', 'orthodox',
                  'lead', 'rear', 'head', 'body',
                  'straights', 'hooks', 'overhands', 'uppercuts',
                  'thrown', 'landed', 'missed', 'accuracy', 'heuristic']

    # for use in 4 arg cycleParams, base parameter set that gets added to
    startingParams = ['min', 'low', 'mid', 'high', 'max', 'aggressionPower', 'pressurePosition', 'accuracy']


    # params to be used in combo
    parameters = ['totalLanded', 'landed', 'thrown', 'missed', 'accuracy', 'highImpact',
                'min', 'low', 'mid', 'high', 'max',
                'minMiss', 'lowMiss', 'midMiss', 'highMiss', 'maxMiss', 'lowCommitMiss', 'highCommitMiss',
                'pressure', 'pressureDistance', 'pressureMovement', 'pressurePosition',
                'aggression', 'aggressionCombinations', 'aggressionExchanges', 'aggressionPower',
                'singles', 'doubles', 'triples', 'quadsPlus',
                'outside', 'midrange', 'inside', 'clinch',
                'backfoot', 'neutral', 'frontfoot',
                'southpaw', 'squared', 'orthodox',
                'lead', 'rear', 'head', 'body',
                'straights', 'hooks', 'overhands', 'uppercuts']
    
    # params to be used in combo
    parameters = ['landed', 'thrown', 'accuracy', 'highImpact',
                'min', 'low', 'mid', 'high', 'max',
                'lowCommitMiss', 'highCommitMiss',
                'pressure', 'pressureDistance', 'pressureMovement', 'pressurePosition',
                'aggression', 'aggressionCombinations', 'aggressionExchanges', 'aggressionPower',
                'singles', 'doubles', 'triples', 'quadsPlus',
                'outside', 'midrange', 'inside', 'clinch',
                'backfoot', 'neutral', 'frontfoot',
                'southpaw', 'squared', 'orthodox',
                'lead', 'rear', 'head', 'body',
                'straights', 'hooks', 'overhands', 'uppercuts']
        

    # # fewer params give solid cost
    # parameters = ['min', 'low', 'mid', 'high', 'max', 'thrownShare',
    #               'pressureDistance', 'pressureMovement', 'pressurePosition',
    #               'aggressionCombinations', 'aggressionExchanges', 'aggressionPower']
    
    # fewer params give solid cost
    # parameters = ['min', 'low', 'mid', 'high', 'max']
    # parameters = ['landed']

    
    # -best flag
    parameters = ['min', 'low', 'mid', 'high', 'max', 'accuracy',
                'minMiss', 'lowMiss', 'midMiss', 'highMiss', 'maxMiss',
                'pressureDistance', 'pressureMovement', 'pressurePosition',
                'aggressionCombinations', 'aggressionExchanges', 'aggressionPower',
                'singles', 'doubles', 'triples', 'quadsPlus',
                'outside', 'midrange', 'inside', 'clinch',
                'backfoot', 'neutral', 'frontfoot',
                'southpaw', 'squared', 'orthodox',
                'lead', 'rear', 'head', 'body',
                'straights', 'hooks', 'overhands', 'uppercuts']


    # # 15 param from l1
    # parameters = ['landedShare', 'thrownShare', 'aggressionPower',
    #               'max', 'high', 'mid', 'lowCommitMiss',
    #               'body', 'hooks', 'inside',
    #               'neutral', 'orthodox','squared',
    #               'pressurePosition', 'pressureDistance']
    
    # # minimal 6
    # parameters = ['landedShare', 'highImpact', 'mid', 'head',
    #             'aggressionPower', 'neutral']
    

    # minimal 6
    # parameters = ['head', 'thrown', 'highImpact', 'backfoot', 'orthodox', 'pressurePosition']
    parameters = ['min', 'low', 'mid', 'high', 'max', 'aggressionPower', 'pressurePosition']
    # parameters = ['head', 'low', 'mid', 'highImpact', 'aggressionPower', 'pressurePosition', 'neutral']



    # # l1 30
    # parameters = ['accuracy', 'landedShare', 'thrownShare',
    #             'min', 'low', 'mid', 'high', 'max', 
    #             'lowCommitMiss', 'highCommitMiss',
    #             'singles', 
    #             'lead', 'rear', 'head', 'body',
    #             'straights', 'hooks', 'overhands', 'uppercuts',
    #             'pressureDistance', 'pressureMovement', 'pressurePosition',
    #             'aggressionCombinations', 'aggressionExchanges', 'aggressionPower',
    #             'inside',
    #             'backfoot', 'neutral',
    #             'squared', 'orthodox']
    # # l1 15
    # parameters = ['landedShare', 'thrownShare',
    #         'min', 'low', 'mid', 'high', 'max', 
    #         'lead', 'rear', 'head', 'body',
    #         'straights', 'hooks', 'overhands', 'uppercuts',
    #         'pressureDistance', 'pressureMovement', 'pressurePosition',
    #         'aggressionCombinations', 'aggressionExchanges', 'aggressionPower',
    #         'backfoot', 'neutral', 'frontfoot',
    #         'southpaw', 'squared', 'orthodox']

    
    # # giant params list
    # parameters = ['landedShare', 'thrownShare', 'accuracy',
    #             'min', 'low', 'mid', 'high', 'max', 
    #             'minMiss', 'lowMiss', 'midMiss', 'highMiss', 'maxMiss',
    #             'pressureDistance', 'pressureMovement', 'pressurePosition',
    #             'aggressionCombinations', 'aggressionExchanges', 'aggressionPower',
    #             'singles', 'doubles', 'triples', 'quadsPlus',
    #             'outside', 'midrange', 'inside', 'clinch',
    #             'backfoot', 'neutral', 'frontfoot',
    #             'southpaw', 'squared', 'orthodox',
    #             'lead', 'rear', 'head', 'body',
    #             'straights', 'hooks', 'overhands', 'uppercuts']
    
    # run neural network if -mlp is in command line
    if '-mlp' in sys.argv:

        # split data into training & testing sets for mlp
        if testSplit != 0.0:
            trainingRatio = 1 - testSplit - validationSplit
            
            # ensure we have enough data for all splits
            if trainingRatio <= 0:
                print(f"{Fore.RED}ERROR:\tNot enough data for the specified split ratios{Fore.WHITE}")
                return
                
            training, testing, validation = dictSplit(data, trainingRatio, testSplit, validationSplit, randomSeed=randomSeed)

        # use default ratios if not specified using -split
        else:
            training, testing, validation = dictSplit(data, 0.8, 0.1, 0.1, randomSeed=randomSeed)

        # run mlp
        mlpData, mlpPredictions, testPredictions = mlp(training, testing, validation, parameters, judgeName, randomSeed=randomSeed)

        # rank judges
        if '-costrank' in sys.argv:
            rankJudges(mlpData, mlpPredictions, -1, judgeName)
        else:
            disagreeThreshold = float(sys.argv[i+1]) if'-dt' in sys.argv else 0
            # make sure dt value is valid
            if disagreeThreshold < 0 or disagreeThreshold > 1:
                print(f"{Fore.RED}ERROR:\tDisagreement threshold must be between 0 and 1{Fore.WHITE}")
                return
            rankJudges(mlpData, mlpPredictions, disagreeThreshold, judgeName)

        # count num of judges
        uniqueJudgeCount = countUniqueJudges(data)
        print(f"Number of judges: {uniqueJudgeCount}")

        # comparison of mlp predictions to judges
        # ! We need to get testPredictions after mlp is completed
        pairwiseComparison(mlpData, mlpPredictions, testing, testPredictions, shrinkageK=shrinkageK)
        
        # grade rounds
        gradeRounds(mlpData, mlpPredictions, parameters, None)
        
        # grade fights
        gradeFights(mlpData, parameters, None, mlpPredictions)
        
        print("\nCompleted: Multi-layer Perceptron analysis\n")

        # prompt user to search for a particular fight
        if '-lookup' in sys.argv: fightLookup(data, mlpPredictions, parameters, None)

        return




    
    # check command line for cyclic gradient descent (switching out parameters to find best combo)
    combosFlag = False
    combostartFlag = False
    paramCount = None

    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '-combos' or sys.argv[i] == '-combo':
            combosFlag = True
            paramCount = int(sys.argv[i+1])
            
            # check to make sure that a positive int appears after combo flag
            if paramCount < 1 or paramCount > 16:
                print(f"{Fore.RED}ERROR: Integer after -combos must be between 1 and 16{Fore.WHITE}")
                return

        elif sys.argv[i] == '-combostart':
            combostartFlag = True
            paramCount = int(sys.argv[i+1])
            
            # check to make sure that a positive int appears after combo flag
            if paramCount < 1 or paramCount > 16:
                print(f"{Fore.RED}ERROR: Integer after -combostart must be between 1 and 16{Fore.WHITE}")
                return


    # use raw data directly for gradient descent
    if testSplit != 0.0:
        gradientData = training
    else:
        gradientData = data

    # use presaved coefficent values
    best = False
    if '-best' in sys.argv:
        optimizedParams = np.array([bestValues[param] for param in parameters])
        best = True
        # TODO check that params uses same params as best

    elif combosFlag:
        # combos uses normalized data for gradient descent
        combos(gradientData, parameters, paramCount, dampener=dampener, sharpness=sharpness)
        return

    elif combostartFlag:
        # comboStart uses normalized data for gradient descent
        comboStart(gradientData, startingParams, allParams, paramCount, dampener=dampener, sharpness=sharpness)
        return
    
    else:
        # gradient descent now returns (bestParams, minCost) tuple
        gradientResult = gradientDescent(gradientData, gradientData['scores'], parameters, 
                                        includeMappingParams=includeMappingParams, 
                                        dampener=dampener, sharpness=sharpness)
        
        # extract just the parameters from the tuple
        if isinstance(gradientResult, tuple):
            optimizedParams, finalCost = gradientResult
        else:
            # fallback for backward compatibility
            optimizedParams = gradientResult



    # create string with judge name in parenthesis if exists
    judgeString = " ("+judgeName+")" if (judgeName != None) else ""

    # print out training results
    trainPM = getPMs(gradientData, parameters, optimizedParams, allParams, dampener, sharpness)
    print(f"\n\n\t\tTraining Results{judgeString}")
    print("------------------------------------------------------------")
    printAccuracy(trainPM, disagreeThreshold)
    print(f"\tAverage Cost:\t\t\t\t\t{round(costCalc(optimizedParams, gradientData, gradientData['scores'], parameters, True, dampener, sharpness),5)}")
    print(f"\tMedian Cost:\t\t\t\t\t{round(costCalc(optimizedParams, gradientData, gradientData['scores'], parameters, False, dampener, sharpness),5)}\n")
    
    # ensure optimizedParams is a numpy array before passing to printValues
    if not isinstance(optimizedParams, np.ndarray):
        optimizedParams = np.array(optimizedParams)
    
    printValues(parameters, optimizedParams, bestValues, best, includeMappingParams)
    
    # print out testing results
    testPM = None
    if testSplit != 0.0:
        # use raw testing data
        testPM = getPMs(testing, parameters, optimizedParams, allParams, dampener, sharpness)
        print(f"\n\n\t\tTesting Results{judgeString}")
        print("------------------------------------------------------------")
        printAccuracy(testPM, disagreeThreshold)
        print(f"\tAverage Cost:\t\t\t\t\t{round(costCalc(optimizedParams, testing, testing['scores'], parameters, True, dampener, sharpness),5)}")
        print(f"\tMedian Cost:\t\t\t\t\t{round(costCalc(optimizedParams, testing, testing['scores'], parameters, False, dampener, sharpness),5)}\n")

        # use raw data for predictions
        pmData = getPMs(data, parameters, optimizedParams, allParams, dampener, sharpness)
    # if no testing data, set pmData to trainPM since it holds all data
    else:
        pmData = trainPM

    if best:    print("\nCompleted:\tUse preset coefficents\n")
    else:       print("\nCompleted:\tGradient descent\n")



    # normalize data and get correlations
    normalizedData, correlations = normalizeData(pmData, allParams)

    # print southpaw vs orthodox win rates
    printStanceWinRate(data)

    # get average stat values & standard deviations
    getAverageStatValues(data)

    # get average stat values & standard deviations by score
    getStatValuesByScore(data)

    # print correlation analysis
    printCorrelations(correlations)

    # get highest total impact
    printHighestImpactFights(data)

    uniqueJudgeCount = countUniqueJudges(data)
    print(f"Number of judges: {uniqueJudgeCount}")

    # stop here if using quad-cam data
    if useQuadcam or useSinglecam:
        # use raw data for grading
        gradeRounds(data, None, parameters, optimizedParams, dampener=dampener, sharpness=sharpness)
        gradeFights(data, parameters, optimizedParams, None, dampener=dampener, sharpness=sharpness)
        pairwiseComparison(data, pmData['heuristic'], None, None, shrinkageK=shrinkageK)
        # for i in range (1,1001):
        #     pairwiseComparison(data, pmData['heuristic'], None, None)
        print(f"\nDone\t\t{len(fights)} fights,\t\t{int(len(data['scores'])/2)} rounds")
        return

    # plot a correlation matrix of the stats
    plotCorrMatrix(data, parameters)

    # make scatter plots
    # plotScatters(pmData, allParams, 'scatters')                    # non-normalized
    # plotScatters(normalizedData, allParams, 'normalizedScatters')  # normalized
    print(f"\nCompleted:\tScatter plots\n")

    # make histograms
    # plotHistograms(pmData, allParams)                              # non-normalized
    # plotNormalHistograms(normalizedData, allParams)                # normalized
    print(f"\nCompleted:\tHistograms\n")


    # prints names of judges with most rounds
    judgeFrequency(data, judgeName)

    # set randRank bool which indicates if all the judges rounds should be used or a random sample
    sampleRank = True if '-sampleRank' in sys.argv else False
    # set disagreeThreshold to -1 if cost rank is being used
    disagreeThreshold = -1 if '-costrank' in sys.argv else disagreeThreshold
    # print judges ranked
    rankJudges(data, pmData['heuristic'], disagreeThreshold, judgeName, sampleRank)
    # print accuracy of predictive system relative to judges
    if testSplit != 0.0:
        pairwiseComparison(data, pmData['heuristic'], testing, testPM['heuristic'], roundThreshold=100, shrinkageK=shrinkageK)
        pairwiseComparison(data, pmData['heuristic'], testing, testPM['heuristic'], roundThreshold=50, shrinkageK=shrinkageK)
        pairwiseComparison(data, pmData['heuristic'], testing, testPM['heuristic'], roundThreshold=20, shrinkageK=shrinkageK)
    else:
        pairwiseComparison(data, pmData['heuristic'], None, None, shrinkageK=shrinkageK)

    # print rounds with biggest difference between actual and predicted score
    # use raw data
    gradeRounds(data, None, parameters, optimizedParams, dampener=dampener, sharpness=sharpness)

    # print fights with biggest average difference between actual and predicted score
    gradeFights(data, parameters, optimizedParams, None, dampener=dampener, sharpness=sharpness)
    
    print(f"\nDone\t\t{len(fights)} fights,\t\t{int(len(data['scores'])/2)} rounds")
    print(f"\t\t{getTenTenCount(fights,judgeName)} 10-10 scores,\t{3*int(len(data['scores'])/2)} 10-9 & 9-10 scores\n")




    # params to be used in l1 logistic regression (all 50)
    lassoParams50 = ['totalLanded', 'landedShare', 'thrownShare',
                'landed', 'thrown', 'missed', 'accuracy', 'highImpact',
                'min', 'low', 'mid', 'high', 'max',
                'minMiss', 'lowMiss', 'midMiss', 'highMiss', 'maxMiss', 'lowCommitMiss', 'highCommitMiss',
                'singles', 'doubles', 'triples', 'quadsPlus',
                'lead', 'rear', 'head', 'body',
                'straights', 'hooks', 'overhands', 'uppercuts',
                'pressure', 'pressureDistance', 'pressureMovement', 'pressurePosition',
                'aggression', 'aggressionCombinations', 'aggressionExchanges', 'aggressionPower',
                'outside', 'midrange', 'inside', 'clinch',
                'backfoot', 'neutral', 'frontfoot',
                'southpaw', 'squared', 'orthodox']
                

    lassoParams35 = ['accuracy', 'landedShare', 'thrownShare',
                'min', 'low', 'mid', 'high', 'max', 
                'lowCommitMiss', 'highCommitMiss',
                'singles', 'doubles', 'triples', 'quadsPlus',
                'lead', 'rear', 'head', 'body',
                'straights', 'hooks', 'overhands', 'uppercuts',
                'pressureDistance', 'pressureMovement', 'pressurePosition',
                'aggressionCombinations', 'aggressionExchanges', 'aggressionPower',
                'inside', 'midrange', 'outside',
                'backfoot', 'neutral',
                'squared', 'orthodox']

    lassoParams30 = ['accuracy', 'landedShare', 'thrownShare',
                'min', 'low', 'mid', 'high', 'max', 
                'lowCommitMiss', 'highCommitMiss',
                'singles', 
                'lead', 'rear', 'head', 'body',
                'straights', 'hooks', 'overhands', 'uppercuts',
                'pressureDistance', 'pressureMovement', 'pressurePosition',
                'aggressionCombinations', 'aggressionExchanges', 'aggressionPower',
                'inside',
                'backfoot', 'neutral',
                'squared', 'orthodox']
    
    lassoParams20 = ['landedShare', 'thrownShare',
            'min', 'low', 'mid', 'highImpact', 
            'lead', 'head',
            'straights', 'hooks',
            'pressureDistance', 'pressureMovement', 'pressurePosition',
            'aggressionCombinations', 'aggressionExchanges', 'aggressionPower',
            'backfoot', 'neutral',
            'squared', 'orthodox']
    
    lassoParams15 = ['landedShare', 'thrownShare',
        'mid', 'highImpact', 
        'head',
        'straights', 'hooks',
        'pressureDistance', 'pressureMovement', 'pressurePosition',
        'aggressionCombinations', 'aggressionExchanges', 'aggressionPower',
        'neutral',
        'orthodox']


    # run L1 analysis with the same parameters used in gradient descent
    l1Model, l1Results, l1Scaler = runL1Analysis(data, lassoParams30)
    # compare results across methods
    print(f"\n{Fore.CYAN}Method Comparison:{Fore.WHITE}")
    print(f"L1 Logistic Regression Test Accuracy: {l1Results['testAccuracy']:.3f}")
    # print(f"Gradient Descent Test Accuracy: {round(100* (unanCorrect+splitCorrect) / (unanTotal+splitTotal),3)}%")



    # prompt user to search for a particular fight
    if '-lookup' in sys.argv:
        # use raw data
        fightLookup(data, None, parameters, optimizedParams, dampener, sharpness)


main()