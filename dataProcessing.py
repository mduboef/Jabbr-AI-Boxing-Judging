import pandas as pd
import os, csv
from datetime import datetime
import numpy as np
from colorama import Fore
from tqdm import tqdm
from scipy import stats                             # for calculating correlation metrics
from sklearn.preprocessing import StandardScaler    # for normalizing pmData

from gradientDescent import heuristic

# class representing a single fighters stats for a single round
class Performance:
    def __init__(self, thrown, landed, highImpact, pressure, aggression, accuracy,
                 min, low, mid, high, max, minMiss, lowMiss, midMiss, highMiss, maxMiss,
                 singles, doubles, triples, quadsPlus,
                 backfoot, frontfoot, neutral, outside, midrange, inside, clinch,
                 aggressionCombinations, aggressionExchanges, aggressionPower,
                 pressureDistance, pressureMovement, pressurePosition,
                 orthodox, southpaw, squared,
                 lead, rear, head, body, straights, hooks, overhands, uppercuts):
        self.thrown = thrown
        self.landed = landed
        self.highImpact = highImpact
        self.pressure = pressure
        self.aggression = aggression
        self.accuracy = accuracy
        self.min = min
        self.low = low
        self.mid = mid
        self.high = high
        self.max = max
        self.minMiss = minMiss
        self.lowMiss = lowMiss
        self.midMiss = midMiss
        self.highMiss = highMiss
        self.maxMiss = maxMiss
        self.singles = singles
        self.doubles = doubles
        self.triples = triples
        self.quadsPlus = quadsPlus
        self.backfoot = backfoot
        self.frontfoot = frontfoot
        self.neutral = neutral
        self.outside = outside
        self.midrange = midrange
        self.inside = inside
        self.clinch = clinch
        self.aggressionCombinations = aggressionCombinations
        self.aggressionExchanges = aggressionExchanges
        self.aggressionPower = aggressionPower
        self.pressureDistance = pressureDistance
        self.pressureMovement = pressureMovement
        self.pressurePosition = pressurePosition
        self.orthodox = orthodox
        self.southpaw = southpaw
        self.squared = squared
        self.lead = lead
        self.rear = rear
        self.head = head
        self.body = body
        self.straights = straights
        self.hooks = hooks
        self.overhands = overhands
        self.uppercuts = uppercuts



    def excludeRound(self, num):
        # NOTE if there is an out of range error it is because of stoppage
        # causing a score for a round that didn't happen (Donaire vs Vetyeka rd 5)
        self.thrown.pop(num)
        self.landed.pop(num)
        self.highImpact.pop(num)
        self.pressure.pop(num)
        self.aggression.pop(num)
        self.accuracy.pop(num)
        self.min.pop(num)
        self.low.pop(num)
        self.mid.pop(num)
        self.high.pop(num)
        self.max.pop(num)
        self.minMiss.pop(num)
        self.lowMiss.pop(num)
        self.midMiss.pop(num)
        self.highMiss.pop(num)
        self.maxMiss.pop(num)
        self.singles.pop(num)
        self.doubles.pop(num)
        self.triples.pop(num)
        self.quadsPlus.pop(num)
        self.backfoot.pop(num)
        self.frontfoot.pop(num)
        self.neutral.pop(num)
        self.outside.pop(num)
        self.midrange.pop(num)
        self.inside.pop(num)
        self.clinch.pop(num)
        self.aggressionCombinations.pop(num)
        self.aggressionExchanges.pop(num)
        self.aggressionPower.pop(num)
        self.pressureDistance.pop(num)
        self.pressureMovement.pop(num)
        self.pressurePosition.pop(num)
        self.orthodox.pop(num)
        self.southpaw.pop(num)
        self.squared.pop(num)
        self.lead.pop(num)
        self.rear.pop(num)
        self.head.pop(num)
        self.body.pop(num)
        self.straights.pop(num)
        self.hooks.pop(num)
        self.overhands.pop(num)
        self.uppercuts.pop(num)


# class holding general fight info and 
class Fight:
    def __init__(self, red, blue, date, rounds, decision, winner, file):
        self.red = red
        self.blue = blue
        self.date = date
        self.rounds = rounds
        self.decision = decision
        self.winner = winner
        self.file = file
        self.pR = None
        self.pB = None

        self.judge1 = None
        self.judge2 = None
        self.judge3 = None

        self.rScore1 = None
        self.bScore1 = None
        self.rScore2 = None
        self.bScore2 = None
        self.rScore3 = None
        self.bScore3 = None

        self.excludedRounds = []
    
    def addRed(self, pR):
        self.pR = pR

    def addBlue(self, pB):
        self.pB = pB

    def addCard(self, judges, scores):
        self.judge1 = judges[0]
        self.judge2 = judges[1]
        self.judge3 = judges[2]

        self.rScore1 = scores[0]
        self.bScore1 = scores[1]
        self.rScore2 = scores[2]
        self.bScore2 = scores[3]
        self.rScore3 = scores[4]
        self.bScore3 = scores[5]

    def excludeRound(self, num):
        self.pR.excludeRound(num)
        self.pB.excludeRound(num)
        self.excludedRounds.append(num+1)

    # turns raw score data (ie [10][9]) into binary data (ie [1]) & skips over excluded rounds
    # returns a array of 3 binary score arrays, 1 for each judge
    def combineSingleScores(self):
        cards = dict(
            r1 = [],
            b1 = [],
            r2 = [],
            b2 = [],
            r3 = [],
            b3 = []
        )

        for i in range(0,len(self.rScore1)):
            # skip over excluded rounds
            if i+1 not in self.excludedRounds and self.rScore1[i] != '':
                if int(self.rScore1[i]) > int(self.bScore1[i]):
                    cards['r1'].append(1)
                    cards['b1'].append(-1)
                else:
                    cards['r1'].append(-1)
                    cards['b1'].append(1)

                if int(self.rScore2[i]) > int(self.bScore2[i]):
                    cards['r2'].append(1)
                    cards['b2'].append(-1)
                else:
                    cards['r2'].append(-1)
                    cards['b2'].append(1)

                if int(self.rScore3[i]) > int(self.bScore3[i]):
                    cards['r3'].append(1)
                    cards['b3'].append(-1)
                else:
                    cards['r3'].append(-1)
                    cards['b3'].append(1)


        return cards
            


    

# holds scores for a single fight before scores added to Fight object
class Card:
    def __init__(self, red, blue, date, judge1, rScore1, bScore1, judge2, rScore2, bScore2, judge3, rScore3, bScore3):
        self.red = red
        self.blue = blue
        self.date = date
        self.judge1 = None
        self.rScore1 = None
        self.bScore1 = None
        self.judge2 = None
        self.rScore2 = None
        self.bScore2 = None
        self.judge3 = None
        self.rScore3 = None
        self.bScore3 = None


# open Combos sheet and count misses by power commit
def readMisses(filepath):

    misses = dict (
        minR = [],
        lowR = [],
        midR = [],
        highR = [],
        maxR = [],
        minB = [],
        lowB = [],
        midB = [],
        highB = [],
        maxB = []
    )

    combos = pd.read_excel(filepath, sheet_name="Combos")  # Specify sheet name

    currentRound = 0    # tracks what round is being looked at

    # loop through combos sheet rows to find misses by power commit
    for i in range(0, len(combos)):

        round = int(combos.iloc[i,0])

        # new round
        if round > currentRound:
            # add 0s to arrays (will be incremented later)
            misses['minR'].append(0)
            misses['lowR'].append(0)
            misses['midR'].append(0)
            misses['highR'].append(0)
            misses['maxR'].append(0)

            misses['minB'].append(0)
            misses['lowB'].append(0)
            misses['midB'].append(0)
            misses['highB'].append(0)
            misses['maxB'].append(0)
            
            currentRound += 1       # update currentRound

            # error check to make sure rounds strickly increase by 1
            if currentRound != round:
                print(f"{Fore.RED}ERROR: Rounds in combo sheet don't increment{Fore.WHITE}")


        # increment total if punch is a miss
        if combos.iloc[i,5] == 0:

            commit = combos.iloc[i,6]

            if commit == 1:
                if combos.iloc[i,1] == 'red':
                    misses['minR'][round-1] = misses['minR'][round-1]+1
                else:
                    misses['minB'][round-1] = misses['minB'][round-1]+1

            elif commit == 2:
                if combos.iloc[i,1] == 'red':
                    misses['lowR'][round-1] = misses['lowR'][round-1]+1
                else:
                    misses['lowB'][round-1] = misses['lowB'][round-1]+1

            elif commit == 3:
                if combos.iloc[i,1] == 'red':
                    misses['midR'][round-1] = misses['midR'][round-1]+1
                else:
                    misses['midB'][round-1] = misses['midB'][round-1]+1

            elif commit == 4:
                if combos.iloc[i,1] == 'red':
                    misses['highR'][round-1] = misses['highR'][round-1]+1
                else:
                    misses['highB'][round-1] = misses['highB'][round-1]+1

            elif commit == 5:
                if combos.iloc[i,1] == 'red':
                    misses['maxR'][round-1] = misses['maxR'][round-1]+1
                else:
                    misses['maxB'][round-1] = misses['maxB'][round-1]+1

            else:
                print(f"{Fore.RED}ERROR: Power Commit Over 5{Fore.WHITE}")
    
    return misses


# read additional punch landed stats from Type Counts sheet
def getTypeCounts(filepath, orthodoxR, southpawR, orthodoxB, southpawB):

    # need to know stance for each fighter in each round to determine lead and rear punches
    redStance = []
    blueStance = []
    for i in range(0, len(orthodoxR)):
        redStance.append('orthodox') if orthodoxR[i] > southpawR[i] else redStance.append('southpaw')
        blueStance.append('orthodox') if orthodoxB[i] > southpawB[i] else blueStance.append('southpaw')

    typeCounts = pd.read_excel(filepath, sheet_name="Type Counts")  # Specify sheet name

    # initialize arrays for the number of rounds we have from orthodox/southpaw data
    numRounds = len(orthodoxR)
    types = dict (
        leadR = [0] * numRounds,
        rearR = [0] * numRounds,
        headR = [0] * numRounds,
        bodyR = [0] * numRounds,
        straightsR = [0] * numRounds,
        hooksR = [0] * numRounds,
        overhandsR = [0] * numRounds,
        uppercutsR = [0] * numRounds,

        leadB = [0] * numRounds,
        rearB = [0] * numRounds,
        headB = [0] * numRounds,
        bodyB = [0] * numRounds,
        straightsB = [0] * numRounds,
        hooksB = [0] * numRounds,
        overhandsB = [0] * numRounds,
        uppercutsB = [0] * numRounds
    )

    # loop through combos sheet to find totals for each fighters in each round
    row = 1
    while row+13 < len(typeCounts):


        round = int(typeCounts.iloc[row,0])
        color = typeCounts.iloc[row,1]
        startingType = typeCounts.iloc[row,2]   # should be lHookBody

        # check if this round exists in our main data
        if round > numRounds:
            break

        lHookBody = int(typeCounts.iloc[row,4])
        lHookHead = int(typeCounts.iloc[row+1,4])
        lOverhandHead = int(typeCounts.iloc[row+2,4])
        lStraightBody = int(typeCounts.iloc[row+3,4])
        lStraightHead = int(typeCounts.iloc[row+4,4])
        lUppercutBody = int(typeCounts.iloc[row+5,4])
        lUppercutHead = int(typeCounts.iloc[row+6,4])
        rHookBody = int(typeCounts.iloc[row+7,4])
        rHookHead = int(typeCounts.iloc[row+8,4])
        rOverhandHead = int(typeCounts.iloc[row+9,4])
        rStraightBody = int(typeCounts.iloc[row+10,4])
        rStraightHead = int(typeCounts.iloc[row+11,4])
        rUppercutBody = int(typeCounts.iloc[row+12,4])
        rUppercutHead = int(typeCounts.iloc[row+13,4])

        # add lead and rear punches totals based on stance
        if color == 'red':
            if redStance[round-1] == 'orthodox':
                types['leadR'][round-1] = lHookBody + lHookHead + lOverhandHead + lStraightBody + lStraightHead + lUppercutBody + lUppercutHead
                types['rearR'][round-1] = rHookBody + rHookHead + rOverhandHead + rStraightBody + rStraightHead + rUppercutBody + rUppercutHead
            else:
                types['leadR'][round-1] = rHookBody + rHookHead + rOverhandHead + rStraightBody + rStraightHead + rUppercutBody + rUppercutHead
                types['rearR'][round-1] = lHookBody + lHookHead + lOverhandHead + lStraightBody + lStraightHead + lUppercutBody + lUppercutHead
        else:
            if blueStance[round-1] == 'orthodox':
                types['leadB'][round-1] = lHookBody + lHookHead + lOverhandHead + lStraightBody + lStraightHead + lUppercutBody + lUppercutHead
                types['rearB'][round-1] = rHookBody + rHookHead + rOverhandHead + rStraightBody + rStraightHead + rUppercutBody + rUppercutHead
            else:
                types['leadB'][round-1] = rHookBody + rHookHead + rOverhandHead + rStraightBody + rStraightHead + rUppercutBody + rUppercutHead
                types['rearB'][round-1] = lHookBody + lHookHead + lOverhandHead + lStraightBody + lStraightHead + lUppercutBody + lUppercutHead

        colorChar = 'R' if color == 'red' else 'B'
        
        # add totals for target and type of punches
        types['head'+colorChar][round-1] = lHookHead + lOverhandHead + lStraightHead + lUppercutHead + rHookHead + rOverhandHead + rStraightHead + rUppercutHead
        types['body'+colorChar][round-1] = lHookBody + lStraightBody + lUppercutBody + rHookBody + rStraightBody + rUppercutBody
        types['straights'+colorChar][round-1] = lStraightBody + lStraightHead + rStraightBody + rStraightHead
        types['hooks'+colorChar][round-1] = lHookBody + lHookHead + rHookBody + rHookHead
        types['overhands'+colorChar][round-1] = lOverhandHead + rOverhandHead
        types['uppercuts'+colorChar][round-1] = lUppercutBody + lUppercutHead + rUppercutBody + rUppercutHead

        row += 14  # move to next round/next fighter
    

    # check if num of rounds in typeCounts matches num of rounds in orthodox/southpaw data
    if len(types['leadR']) != numRounds:
        print(f"{Fore.RED}ERROR: Number of rounds in Type Counts does not match number of rounds in orthodox/southpaw data{Fore.WHITE}")
        return None

    return types

# reads in data from excel files
# returns list of Fight objects with all stats
def readStats(commandLine):

    fights = []

    # get the directory containing the Excel files
    if '-singlecam' in commandLine:
        directory = os.getcwd() + "/singleStats/"
    else:
        directory = os.getcwd() + "/stats/"  # overall stats folder called "stats" in the current working directory

    # list all Excel files in the directory (handles various extensions)
    excelFiles = [f for f in os.listdir(directory) if f.endswith((".xlsx", ".xlsm", ".xlsb"))]

    pbar = tqdm(total=len(excelFiles), desc="Reading stats", unit="file")

    # check for -includeInserted flag and set includeInserted accordingly
    icludeInserted = True if '-includeinserted' in commandLine else False

    for filename in excelFiles:

        # skip files with (INSERTED)_ in the name if -includeInserted flag is not set
        if not icludeInserted and '(INSERTED)_' in filename:
            continue

        # construct the full path to the Excel file
        filepath = os.path.join(directory, filename)

        # read specific sheet "Match Info"
        info = pd.read_excel(filepath, sheet_name="Match Info")  # Specify sheet name

        # read specific sheet "Summary" into a pandas DataFrame
        df = pd.read_excel(filepath, sheet_name="Summary")  # Specify sheet name

        # get misses by power commit and store in a dict
        misses = readMisses(filepath)

        rName = info.iloc[0,0].replace('’', "'")    # change apostrophes in names to match scorecards
        bName = info.iloc[0,1].replace('’', "'")
        date = cleanDate(info.iloc[0,2])            # format date to match scorecards

        # construct Fight object
        fight = Fight(rName, bName, date, info.iloc[0,4], info.iloc[0,6], info.iloc[0,7], filename)

        # initalize red stats
        thrownR = []
        landedR = []
        highImpactR = []
        pressureR = []
        aggressionR = []
        accuracyR = []
        minR = []
        lowR = []
        midR = []
        highR = []
        maxR = []
        minMissR = misses['minR']
        lowMissR = misses['lowR']
        midMissR = misses['midR']
        highMissR = misses['highR']
        maxMissR = misses['maxR']
        singlesR = []
        doublesR = []
        triplesR = []
        quadsPlusR = []
        backfootR = []
        frontfootR = []
        neutralR = []
        outsideR = []
        midrangeR = []
        insideR = []
        clinchR = []
        aggressionCombinationsR = []
        aggressionExchangesR = []
        aggressionPowerR = []
        pressureDistanceR = []
        pressureMovementR = []
        pressurePositionR = []
        orthodoxR = []
        southpawR = []
        squaredR = []

        # initalize blue stats
        thrownB = []
        landedB = []
        highImpactB = []
        pressureB = []
        aggressionB = []
        accuracyB = []
        minB = []
        lowB = []
        midB = []
        highB = []
        maxB = []
        minMissB = misses['minB']
        lowMissB = misses['lowB']
        midMissB = misses['midB']
        highMissB = misses['highB']
        maxMissB = misses['maxB']
        singlesB = []
        doublesB = []
        triplesB = []
        quadsPlusB = []
        backfootB = []
        frontfootB = []
        neutralB = []
        outsideB = []
        midrangeB = []
        insideB = []
        clinchB = []
        aggressionCombinationsB = []
        aggressionExchangesB = []
        aggressionPowerB = []
        pressureDistanceB = []
        pressureMovementB = []
        pressurePositionB = []
        orthodoxB = []
        southpawB = []
        squaredB = []


        # loop through columns
        roundsScheduled = int(info.iloc[0,5])
        for i in range(0,roundsScheduled*2):

            # check if we're going out of bounds
            if i+2 >= df.shape[1]:
                print(f"Debug: Would go out of bounds at column {i+2}, df has {df.shape[1]} columns")
                break

            # '-' indicates fight ended early
            if df.iloc[2,i+2] == '-':
                break

            # append to red stats
            elif i % 2 == 0:
                thrownR.append(df.iloc[1,i+2])
                landedR.append(df.iloc[2,i+2])
                highImpactR.append(df.iloc[3,i+2])
                pressureR.append(df.iloc[4,i+2] * 100)
                aggressionR.append(df.iloc[5,i+2] * 100)
                accuracyR.append(df.iloc[6,i+2])
                minR.append(df.iloc[7,i+2])
                lowR.append(df.iloc[8,i+2])
                midR.append(df.iloc[9,i+2])
                highR.append(df.iloc[10,i+2])
                maxR.append(df.iloc[11,i+2])
                singlesR.append(df.iloc[12,i+2] * 100)
                doublesR.append(df.iloc[13,i+2] * 100)
                triplesR.append(df.iloc[14,i+2] * 100)
                quadsPlusR.append(df.iloc[15,i+2] * 100)
                backfootR.append(df.iloc[16,i+2] * 100)
                frontfootR.append(df.iloc[17,i+2] * 100)
                neutralR.append(df.iloc[18,i+2] * 100)
                outsideR.append(df.iloc[19,i+2] * 100)
                midrangeR.append(df.iloc[20,i+2] * 100)
                insideR.append(df.iloc[21,i+2] * 100)
                clinchR.append(df.iloc[22,i+2] * 100)
                aggressionCombinationsR.append(df.iloc[23,i+2] * 100)
                aggressionExchangesR.append(df.iloc[24,i+2] * 100)
                aggressionPowerR.append(df.iloc[25,i+2] * 100)
                pressureDistanceR.append(df.iloc[26,i+2] * 100)
                pressureMovementR.append(df.iloc[27,i+2] * 100)
                pressurePositionR.append(df.iloc[28,i+2] * 100)
                orthodoxR.append(df.iloc[29,i+2] * 100)
                southpawR.append(df.iloc[30,i+2] * 100)
                squaredR.append(df.iloc[31,i+2] * 100)


            # append to blue stats
            else:
                thrownB.append(df.iloc[1,i+2])
                landedB.append(df.iloc[2,i+2])
                highImpactB.append(df.iloc[3,i+2])
                pressureB.append(df.iloc[4,i+2] * 100)
                aggressionB.append(df.iloc[5,i+2] * 100)
                accuracyB.append(df.iloc[6,i+2])
                minB.append(df.iloc[7,i+2])
                lowB.append(df.iloc[8,i+2])
                midB.append(df.iloc[9,i+2])
                highB.append(df.iloc[10,i+2])
                maxB.append(df.iloc[11,i+2])
                singlesB.append(df.iloc[12,i+2] * 100)
                doublesB.append(df.iloc[13,i+2] * 100)
                triplesB.append(df.iloc[14,i+2] * 100)
                quadsPlusB.append(df.iloc[15,i+2] * 100)
                backfootB.append(df.iloc[16,i+2] * 100)
                frontfootB.append(df.iloc[17,i+2] * 100)
                neutralB.append(df.iloc[18,i+2] * 100)
                outsideB.append(df.iloc[19,i+2] * 100)
                midrangeB.append(df.iloc[20,i+2] * 100)
                insideB.append(df.iloc[21,i+2] * 100)
                clinchB.append(df.iloc[22,i+2] * 100)
                aggressionCombinationsB.append(df.iloc[23,i+2] * 100)
                aggressionExchangesB.append(df.iloc[24,i+2] * 100)
                aggressionPowerB.append(df.iloc[25,i+2] * 100)
                pressureDistanceB.append(df.iloc[26,i+2] * 100)
                pressureMovementB.append(df.iloc[27,i+2] * 100)
                pressurePositionB.append(df.iloc[28,i+2] * 100)
                orthodoxB.append(df.iloc[29,i+2] * 100)
                southpawB.append(df.iloc[30,i+2] * 100)
                squaredB.append(df.iloc[31,i+2] * 100)

        # populate stats from Type Counts sheet
        typeCounts = getTypeCounts(filepath, orthodoxR, southpawR, orthodoxB, southpawB)
        leadR = typeCounts['leadR']
        rearR = typeCounts['rearR']
        headR = typeCounts['headR']
        bodyR = typeCounts['bodyR']
        straightsR = typeCounts['straightsR']
        hooksR = typeCounts['hooksR']
        overhandsR = typeCounts['overhandsR']
        uppercutsR = typeCounts['uppercutsR']
        leadB = typeCounts['leadB']
        rearB = typeCounts['rearB']
        headB = typeCounts['headB']
        bodyB = typeCounts['bodyB']
        straightsB = typeCounts['straightsB']
        hooksB = typeCounts['hooksB']
        overhandsB = typeCounts['overhandsB']
        uppercutsB = typeCounts['uppercutsB']


        pR = Performance(thrownR, landedR, highImpactR, pressureR, aggressionR, accuracyR,
            minR, lowR, midR, highR, maxR, minMissR, lowMissR, midMissR, highMissR, maxMissR,
            singlesR, doublesR, triplesR, quadsPlusR,
            backfootR, frontfootR, neutralR, outsideR, midrangeR, insideR, clinchR,
            aggressionCombinationsR, aggressionExchangesR, aggressionPowerR,
            pressureDistanceR, pressureMovementR, pressurePositionR,
            orthodoxR, southpawR, squaredR,
            leadR, rearR, headR, bodyR, straightsR, hooksR, overhandsR, uppercutsR)
        
        pB = Performance(thrownB, landedB, highImpactB, pressureB, aggressionB, accuracyB,
            minB, lowB, midB, highB, maxB, minMissB, lowMissB, midMissB, highMissB, maxMissB,
            singlesB, doublesB, triplesB, quadsPlusB,
            backfootB, frontfootB, neutralB, outsideB, midrangeB, insideB, clinchB,
            aggressionCombinationsB, aggressionExchangesB, aggressionPowerB,
            pressureDistanceB, pressureMovementB, pressurePositionB,
            orthodoxB, southpawB, squaredB,
            leadB, rearB, headB, bodyB, straightsB, hooksB, overhandsB, uppercutsB)

        fight.addRed(pR)
        fight.addBlue(pB)
        fights.append(fight)

        # check that thrown = landed + missed
        for i in range(0, len(thrownR)):
            missedR = minMissR[i]+lowMissR[i]+midMissR[i]+highMissR[i]+maxMissR[i]  # total number of missed punches
            missedB = minMissB[i]+lowMissB[i]+midMissB[i]+highMissB[i]+maxMissB[i]
            if (thrownR[i] != missedR + landedR[i]) or (thrownB[i] != missedB + landedB[i]):
                print(f"{Fore.RED}ERROR:\tthrown != landed + missed")
                print(f"\t{rName} vs {bName} Round {i+1})")
                print(f"\tRed : {thrownR[i]} != {landedR[i]} + {missedR}")
                print(f"\tBlue: {thrownB[i]} != {landedB[i]} + {missedB}{Fore.WHITE}")

        # Update the progress bar
        pbar.update(1)


    # Close the progress bar
    pbar.close()

    # search for duplicate fights
    for i in range(0, len(fights)-1):
        for j in range(i+1, len(fights)):
            if fights[i].red==fights[j].red and fights[i].blue==fights[j].blue and fights[i].date==fights[j].date:
                print(f"Duplicate found in DeepStrike Data: {fights[i].red} vs. {fights[i].blue} ({fights[i].date})")

    return fights
                    

# reads CSV with scores and returns list of Card objects
def readCards(useQuadcam, useSingleCam):
    cards = []
    filename = "quadCards.csv" if (useQuadcam or useSingleCam)  else "pairedData.csv"

    try:
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)    # skip header
            for row in reader:

                # Extract data from info row
                redCorner = row[0]
                blueCorner = row[1]
                date = row[3]
                judge1 = row[6]
                judge2 = row[7]
                judge3 = row[8]

                next(reader)    # skip link

                # read next 6 rows and save each full row to its own list
                scores = []
                for i in range(0,6):  # Skip the link row and read 6 rounds
                    scores.append(next(reader))

                card = {
                    "red": redCorner,
                    "blue": blueCorner,
                    "date": date,
                    "judges": [judge1, judge2, judge3],
                    "scores": scores
                }

                if scores [0] == None:
                    print(f"NONE SCORES\n{scores}")
                else:
                    cards.append(card)
    except FileNotFoundError:
        raise ValueError(f"{Fore.RED}ERROR: CSV file pairedData.csv not found.{Fore.WHITE}")
    except IndexError:
        raise ValueError(f"{Fore.RED}ERROR: CSV file pairedData.csv may be empty or have incorrect format.{Fore.WHITE}")



    return cards


# add corresponding score info to fights
def pairFightsToCards(fights, cards):
    # add score data from cards to Fight objects
    for i in range(0,len(fights)):
        matched = False

        # look through cards to find matching fight
        for j in range(0,len(cards)):
            if fights[i].red == cards[j]["red"] and fights[i].blue == cards[j]["blue"]:
                # Convert dates to datetime objects for comparison
                fightDate = datetime.strptime(fights[i].date, "%m/%d/%Y")  # Adjust format as needed
                cardDate = datetime.strptime(cards[j]["date"], "%m/%d/%Y")  # Adjust format as needed

                # match found
                if fightDate == cardDate:
                    fights[i].addCard(cards[j]["judges"], cards[j]["scores"])
                    matched = True
                    break

        # no matching card (fight date probably incorrect)
        if not matched:
            print(f"{Fore.RED}ERROR: NO CARD FOUND    {fights[i].red} vs. {fights[i].blue}    ({fights[i].date})\nPlease check date given to DeepStrike")
            print(f"{fights[i].file}{Fore.WHITE}\n")

    # remove any fights that dont have scores
    fights = [fight for fight in fights if fight.rScore1 is not None]

    return fights


# takes date strings output by DeepStrike and returns formmatted version
def cleanDate(dateStr):
    # Define a mapping of month abbreviations to numeric values
    month_mapping = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
        'May': '05', 'Jun': '06', 'June': '06', 'Jul': '07', 'July': '07', 'Aug': '08',
        'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
    }
    # Split the input date string into day, month, and year
    day, month, year = dateStr.split()
    # Convert the month abbreviation to its numeric value
    month_numeric = month_mapping.get(month, '00')  # Default to '00' if not found
    # Format the date in "month/day/year" format
    formatted_date = f"{month_numeric}/{day}/{year}"
    return formatted_date


# take array of fights and return number of 10-10 scores
# compare output to number of rounds used after exclusion (just the 10-9 & 9-10s)
def getTenTenCount(fights, judgeName):
    tenTenCount = 0
    # if there is a judge specified, count only their 10-10s
    if judgeName != None:
        for i in range(0, len(fights)):
            # check for judge
            if judgeName == fights[i].judge1:
                scoresR = fights[i].rScore1
                scoresB = fights[i].bScore1
            elif judgeName == fights[i].judge2:
                scoresR = fights[i].rScore2
                scoresB = fights[i].bScore2
            elif judgeName == fights[i].judge3:
                scoresR = fights[i].rScore3
                scoresB = fights[i].bScore3
            # if judge not found, move to next fight
            else:
                continue

            for j in range(0, len(scoresR)):
                # break if fight over
                if scoresR[j] == '':
                    break
                # count 10-10s
                if int(scoresR[j]) == int(scoresB[j]) == 10:
                    tenTenCount += 1
        
    # if no judge specified count all 10-10s
    else: 
        for i in range(0, len(fights)):
            j1R = fights[i].rScore1
            j1B = fights[i].bScore1
            j2R = fights[i].rScore2
            j2B = fights[i].bScore2
            j3R = fights[i].rScore3
            j3B = fights[i].bScore3

            for j in range(0, len(j1R)):
                # break if fight over
                if j1R[j] == '':
                    break
                # count 10-10s
                if int(j1R[j]) == int(j1B[j]) == 10:
                    tenTenCount += 1
                if int(j2R[j]) == int(j2B[j]) == 10:
                    tenTenCount += 1
                if int(j3R[j]) == int(j3B[j]) == 10:
                    tenTenCount += 1

    return tenTenCount

# takes all score lists and reduces them to a single list
# 3/3 means all 3 judges scored for red, 0/3 means none did
def combineScores(j1R, j2R, j3R, j1B, j2B, j3B, judgeNum):

    combined = []

    # cut off cards at end of fight
    end = 12
    for i in range(0,len(j1R)):
        if j1R[i] == '':
            end = i
            break

    if end != 12:
        j1R = j1R[:end]
        j2R = j2R[:end]
        j3R = j3R[:end]
        j1B = j1B[:end]
        j2B = j2B[:end]
        j3B = j3B[:end]


    for i in range(0,end):

        # break if fight over
        if j1R[i] == '':
            print(f"{Fore.RED}ERROR: Non-scored round{Fore.WHITE}")
            break

        score = 0
        score1 = 0
        score2 = 0
        score3 = 0

        # add 1 for red, subtract 1 for bule
        if int(j1R[i]) > int(j1B[i]):
            score1 += 1
        else:
            score1 -= 1
        if int(j2R[i]) > int(j2B[i]):
            score2 += 1
        else:
            score2 -= 1
        if int(j3R[i]) > int(j3B[i]):
            score3 += 1
        else:
            score3 -= 1
        if judgeNum == 1 or judgeNum == None:
            score += score1
        if judgeNum == 2 or judgeNum == None:
            score += score2
        if judgeNum == 3 or judgeNum == None:
            score += score3

        # round exclusion when all judges considered
        if judgeNum == None:

            # exclude even rounds
            if int(j1R[i]) == int(j1B[i]) or int(j2R[i]) == int(j2B[i]) or int(j3R[i]) == int(j3B[i]):
                score = -30

            # exclude round with anything under 9
            elif int(j1R[i])<9 or int(j1B[i])<9 or int(j2R[i])<9 or int(j2B[i])<9 or int(j3R[i])<9 or int(j3B[i])<9:
                score = -30
    
        # round exclusion when -j flag is set
        elif judgeNum == 1:
            if int(j1R[i]) == int(j1B[i]):      # exclude even rounds
                score = -10
                print("EVEN ROUND")
            if int(j1R[i])<9 or int(j1B[i])<9:  # exclude non 10-9 rounds
                score = -10
        elif judgeNum == 2:
            if int(j2R[i]) == int(j2B[i]):      # exclude even rounds
                score = -10
                print("EVEN ROUND")
            if int(j2R[i])<9 or int(j2B[i])<9:  # exclude non 10-9 rounds
                score = -10
        else:
            if int(j3R[i]) == int(j3B[i]):      # exclude even rounds
                score = -10
                print("EVEN ROUND")
            if int(j3R[i])<9 or int(j3B[i])<9:  # exclude non 10-9 rounds
                score = -10

        if judgeNum == None:
            combined.append(score/3)
        else:
            combined.append(score/1)

    return combined


# takes combined score list in terms of one corner and returns form other fighers perspective
def reverseScores(rScore):
    bScore = []
    for i in range(0,len(rScore)):
        if rScore[i] == -1.0:
            bScore.append(1.0)
        elif rScore[i] == -1/3:
            bScore.append(1/3)
        elif rScore[i] == 1/3:
            bScore.append(-1/3)
        elif rScore[i] == 1.0:
            bScore.append(-1.0)
        else:
            print(f"{Fore.RED}ERROR: SCORE NOT A MULTIPLE OF 1/3{Fore.WHITE}")
    return bScore


# creates a data dictionary from Fight objects
def fightToData(fight, data, judgeNum):

    # compress score data into single list in terms of red
    # NOTE this could be cleaner if combineScores spit out two arrays 1 for red, 1 for blue then they both had rounds excluded
    redScore = combineScores(fight.rScore1, fight.rScore2, fight.rScore3, fight.bScore1, fight.bScore2, fight.bScore3, judgeNum)


    # Get the actual number of rounds we have performance data for
    performanceRounds = len(fight.pR.landed)

    # Check if we have a mismatch between scorecard and performance data
    if len(redScore) > performanceRounds:
        print(f"{Fore.YELLOW}Warning: Mismatch in round count for {fight.red} vs {fight.blue} ({fight.date})")
        print(f"Scorecard rounds: {len(redScore)}, Performance data rounds: {performanceRounds}{Fore.WHITE}")
        # Truncate redScore to match performance data
        redScore = redScore[:performanceRounds]

    # NOTE probably makes more sense to do exclusion inside combineScores
    # remove -10 rounds from redScore
    r = list(range(1, len(redScore) + 1))
    for i in range(len(redScore) - 1, -1, -1):
        if redScore[i] == -10:
            redScore.pop(i)
            fight.excludeRound(i)

    roundCount = []         # store round counter in roundCount (ie. [1,2,4,5,6,7,8])
    for i in range(0,len(r)):
        if r[i] not in fight.excludedRounds:
            roundCount.append(r[i])

    # get scores in terms of blue
    blueScore = reverseScores(redScore)

    # get binary versions of individual judge scorecards
    individualCards = fight.combineSingleScores()

    # append stats from fight objects to data dict
    for i in range(0,len(roundCount)):
        # store general info
        data["name"].append( fight.red )
        data["name"].append( fight.blue )
        data["nameO"].append( fight.blue )
        data["nameO"].append( fight.red )
        data["date"].append( fight.date )
        data["date"].append( fight.date )
        data["round"].append( roundCount[i] )
        data["round"].append( roundCount[i] )
        data["color"].append( "red" )
        data["color"].append( "blue" )
        data["scores"].append( redScore[i] )
        data["scores"].append( blueScore[i] )

        # store judge specifc scores
        data["score1"].append(individualCards['r1'][i])
        data["score1"].append(individualCards['b1'][i])
        data["score2"].append(individualCards['r2'][i])
        data["score2"].append(individualCards['b2'][i])
        data["score3"].append(individualCards['r3'][i])
        data["score3"].append(individualCards['b3'][i])

        # store judges' names
        if judgeNum == 1 or judgeNum == None:
            j1 = fight.judge1
        else:
            j1 = None
        if judgeNum == 2 or judgeNum == None:
            j2 = fight.judge2
        else:
            j2 = None
        if judgeNum == 3 or judgeNum == None:
            j3 = fight.judge3
        else:
            j3 = None
        data["judge1"].append( j1 )
        data["judge1"].append( j1 )
        data["judge2"].append( j2 )
        data["judge2"].append( j2 )
        data["judge3"].append( j3 )
        data["judge3"].append( j3 )

        # store stats that apply to both fighters
        totalThrown = fight.pR.thrown[i] + fight.pB.thrown[i]
        data["totalThrown"].append( totalThrown )
        data["totalThrown"].append( totalThrown )
        totalLanded = fight.pR.landed[i] + fight.pB.landed[i]
        data["totalLanded"].append( totalLanded )
        data["totalLanded"].append( totalLanded )


        # store ratio stats for primary fighter
        tt = totalThrown if totalThrown != 0 else 1     # avoid division by zero
        tl = totalLanded if totalLanded != 0 else 1
        data["thrownShare"].append( fight.pR.thrown[i] / tt )
        data["thrownShare"].append( fight.pB.thrown[i] / tt )
        data["landedShare"].append( fight.pR.landed[i] / tl )
        data["landedShare"].append( fight.pB.landed[i] / tl )


        # store stats for primary fighter
        data["thrown"].append( fight.pR.thrown[i] )
        data["thrown"].append( fight.pB.thrown[i] )
        data["landed"].append( fight.pR.landed[i] )
        data["landed"].append( fight.pB.landed[i] )
        data["missed"].append( fight.pR.thrown[i] - fight.pR.landed[i] )
        data["missed"].append( fight.pB.thrown[i] - fight.pB.landed[i] )
        if fight.pR.thrown[i] < 1:
            data["accuracy"].append( 0 )
        else:
            data["accuracy"].append( 100* (fight.pR.landed[i] / fight.pR.thrown[i]) )
        if fight.pB.thrown[i] < 1:
            data["accuracy"].append( 0 )
        else:
            data["accuracy"].append( 100* (fight.pB.landed[i] / fight.pB.thrown[i]) )
        data["highImpact"].append( fight.pR.highImpact[i] )
        data["highImpact"].append( fight.pB.highImpact[i] )
        data["pressure"].append( fight.pR.pressure[i] )
        data["pressure"].append( fight.pB.pressure[i] )
        data["pressureDistance"].append( fight.pR.pressureDistance[i] )
        data["pressureDistance"].append( fight.pB.pressureDistance[i] )
        data["pressureMovement"].append( fight.pR.pressureMovement[i] )
        data["pressureMovement"].append( fight.pB.pressureMovement[i] )
        data["pressurePosition"].append( fight.pR.pressurePosition[i] )
        data["pressurePosition"].append( fight.pB.pressurePosition[i] )
        data["aggression"].append( fight.pR.aggression[i] )
        data["aggression"].append( fight.pB.aggression[i] )
        data["aggressionCombinations"].append( fight.pR.aggressionCombinations[i] )
        data["aggressionCombinations"].append( fight.pB.aggressionCombinations[i] )
        data["aggressionExchanges"].append( fight.pR.aggressionExchanges[i] )
        data["aggressionExchanges"].append( fight.pB.aggressionExchanges[i] )
        data["aggressionPower"].append( fight.pR.aggressionPower[i] )
        data["aggressionPower"].append( fight.pB.aggressionPower[i] )
        data["orthodox"].append( fight.pR.orthodox[i] )
        data["orthodox"].append( fight.pB.orthodox[i] )
        data["southpaw"].append( fight.pR.southpaw[i] )
        data["southpaw"].append( fight.pB.southpaw[i] )
        data["squared"].append( fight.pR.squared[i] )
        data["squared"].append( fight.pB.squared[i] )
        data["singles"].append( fight.pR.singles[i] )
        data["singles"].append( fight.pB.singles[i] )
        data["doubles"].append( fight.pR.doubles[i] )
        data["doubles"].append( fight.pB.doubles[i] )
        data["triples"].append( fight.pR.triples[i] )
        data["triples"].append( fight.pB.triples[i] )
        data["quadsPlus"].append( fight.pR.quadsPlus[i] )
        data["quadsPlus"].append( fight.pB.quadsPlus[i] )
        data["outside"].append( fight.pR.outside[i] )
        data["outside"].append( fight.pB.outside[i] )
        data["midrange"].append( fight.pR.midrange[i] )
        data["midrange"].append( fight.pB.midrange[i] )
        data["inside"].append( fight.pR.inside[i] )
        data["inside"].append( fight.pB.inside[i] )
        data["clinch"].append( fight.pR.clinch[i] )
        data["clinch"].append( fight.pB.clinch[i] )
        data["backfoot"].append( fight.pR.backfoot[i] )
        data["backfoot"].append( fight.pB.backfoot[i] )
        data["frontfoot"].append( fight.pR.frontfoot[i] )
        data["frontfoot"].append( fight.pB.frontfoot[i] )
        data["neutral"].append( fight.pR.neutral[i] )
        data["neutral"].append( fight.pB.neutral[i] )
        data["lead"].append( fight.pR.lead[i] )
        data["lead"].append( fight.pB.lead[i] )
        data["rear"].append( fight.pR.rear[i] )
        data["rear"].append( fight.pB.rear[i] )
        data["head"].append( fight.pR.head[i] )
        data["head"].append( fight.pB.head[i] )
        data["body"].append( fight.pR.body[i] )
        data["body"].append( fight.pB.body[i] )
        data["straights"].append( fight.pR.straights[i] )
        data["straights"].append( fight.pB.straights[i] )
        data["hooks"].append( fight.pR.hooks[i] )
        data["hooks"].append( fight.pB.hooks[i] )
        data["overhands"].append( fight.pR.overhands[i] )
        data["overhands"].append( fight.pB.overhands[i] )
        data["uppercuts"].append( fight.pR.uppercuts[i] )
        data["uppercuts"].append( fight.pB.uppercuts[i] )

        data["min"].append( fight.pR.min[i] )
        data["min"].append( fight.pB.min[i] )
        data["low"].append( fight.pR.low[i] )
        data["low"].append( fight.pB.low[i] )
        data["mid"].append( fight.pR.mid[i] )
        data["mid"].append( fight.pB.mid[i] )
        data["high"].append( fight.pR.high[i] )
        data["high"].append( fight.pB.high[i] )
        data["max"].append( fight.pR.max[i] )
        data["max"].append( fight.pB.max[i] )
        data["minMiss"].append( fight.pR.minMiss[i] )
        data["minMiss"].append( fight.pB.minMiss[i] )
        data["lowMiss"].append( fight.pR.lowMiss[i] )
        data["lowMiss"].append( fight.pB.lowMiss[i] )
        data["midMiss"].append( fight.pR.midMiss[i] )
        data["midMiss"].append( fight.pB.midMiss[i] )
        data["highMiss"].append( fight.pR.highMiss[i] )
        data["highMiss"].append( fight.pB.highMiss[i] )
        data["maxMiss"].append( fight.pR.maxMiss[i] )
        data["maxMiss"].append( fight.pB.maxMiss[i] )
        data["lowCommitMiss"].append( fight.pR.minMiss[i] + fight.pR.lowMiss[i] )
        data["lowCommitMiss"].append( fight.pB.minMiss[i] + fight.pB.lowMiss[i] )
        data["highCommitMiss"].append( fight.pR.midMiss[i] + fight.pR.highMiss[i] + fight.pR.maxMiss[i] )
        data["highCommitMiss"].append( fight.pB.midMiss[i] + fight.pB.highMiss[i] + fight.pB.maxMiss[i] )



        # store ratio stats for opponent fighter
        data["thrownShareO"].append( fight.pB.thrown[i] / tt )
        data["thrownShareO"].append( fight.pR.thrown[i] / tt )
        data["landedShareO"].append( fight.pB.landed[i] / tl )
        data["landedShareO"].append( fight.pR.landed[i] / tl )

        # store stats for opponent fighter
        data["thrownO"].append( fight.pB.thrown[i] )
        data["thrownO"].append( fight.pR.thrown[i] )
        data["landedO"].append( fight.pB.landed[i] )
        data["landedO"].append( fight.pR.landed[i] )
        data["missedO"].append( fight.pB.thrown[i] - fight.pB.landed[i] )
        data["missedO"].append( fight.pR.thrown[i] - fight.pR.landed[i] )
        if fight.pB.thrown[i] < 1:
            data["accuracyO"].append( 0 )
        else:
            data["accuracyO"].append( 100* (fight.pB.landed[i] / fight.pB.thrown[i]) )
        if fight.pR.thrown[i] < 1:
            data["accuracyO"].append( 0 )
        else:
            data["accuracyO"].append( 100* (fight.pR.landed[i] / fight.pR.thrown[i]) )
        data["highImpactO"].append( fight.pB.highImpact[i] )
        data["highImpactO"].append( fight.pR.highImpact[i] )
        data["pressureO"].append( fight.pB.pressure[i] )
        data["pressureO"].append( fight.pR.pressure[i] )
        data["pressureDistanceO"].append( fight.pB.pressureDistance[i] )
        data["pressureDistanceO"].append( fight.pR.pressureDistance[i] )
        data["pressureMovementO"].append( fight.pB.pressureMovement[i] )
        data["pressureMovementO"].append( fight.pR.pressureMovement[i] )
        data["pressurePositionO"].append( fight.pB.pressurePosition[i] )
        data["pressurePositionO"].append( fight.pR.pressurePosition[i] )
        data["aggressionO"].append( fight.pB.aggression[i] )
        data["aggressionO"].append( fight.pR.aggression[i] )
        data["aggressionCombinationsO"].append( fight.pB.aggressionCombinations[i] )
        data["aggressionCombinationsO"].append( fight.pR.aggressionCombinations[i] )
        data["aggressionExchangesO"].append( fight.pB.aggressionExchanges[i] )
        data["aggressionExchangesO"].append( fight.pR.aggressionExchanges[i] )
        data["aggressionPowerO"].append( fight.pB.aggressionPower[i] )
        data["aggressionPowerO"].append( fight.pR.aggressionPower[i] )
        data["orthodoxO"].append( fight.pB.orthodox[i] )
        data["orthodoxO"].append( fight.pR.orthodox[i] )
        data["southpawO"].append( fight.pB.southpaw[i] )
        data["southpawO"].append( fight.pR.southpaw[i] )
        data["squaredO"].append( fight.pB.squared[i] )
        data["squaredO"].append( fight.pR.squared[i] )
        data["singlesO"].append( fight.pB.singles[i] )
        data["singlesO"].append( fight.pR.singles[i] )
        data["doublesO"].append( fight.pB.doubles[i] )
        data["doublesO"].append( fight.pR.doubles[i] )
        data["triplesO"].append( fight.pB.triples[i] )
        data["triplesO"].append( fight.pR.triples[i] )
        data["quadsPlusO"].append( fight.pB.quadsPlus[i] )
        data["quadsPlusO"].append( fight.pR.quadsPlus[i] )
        data["outsideO"].append( fight.pB.outside[i] )
        data["outsideO"].append( fight.pR.outside[i] )
        data["midrangeO"].append( fight.pB.midrange[i] )
        data["midrangeO"].append( fight.pR.midrange[i] )
        data["insideO"].append( fight.pB.inside[i] )
        data["insideO"].append( fight.pR.inside[i] )
        data["clinchO"].append( fight.pB.clinch[i] )
        data["clinchO"].append( fight.pR.clinch[i] )
        data["backfootO"].append( fight.pB.backfoot[i] )
        data["backfootO"].append( fight.pR.backfoot[i] )
        data["frontfootO"].append( fight.pB.frontfoot[i] )
        data["frontfootO"].append( fight.pR.frontfoot[i] )
        data["neutralO"].append( fight.pB.neutral[i] )
        data["neutralO"].append( fight.pR.neutral[i] )
        data["leadO"].append( fight.pB.lead[i] )
        data["leadO"].append( fight.pR.lead[i] )
        data["rearO"].append( fight.pB.rear[i] )
        data["rearO"].append( fight.pR.rear[i] )
        data["headO"].append( fight.pB.head[i] )
        data["headO"].append( fight.pR.head[i] )
        data["bodyO"].append( fight.pB.body[i] )
        data["bodyO"].append( fight.pR.body[i] )
        data["straightsO"].append( fight.pB.straights[i] )
        data["straightsO"].append( fight.pR.straights[i] )
        data["hooksO"].append( fight.pB.hooks[i] )
        data["hooksO"].append( fight.pR.hooks[i] )
        data["overhandsO"].append( fight.pB.overhands[i] )
        data["overhandsO"].append( fight.pR.overhands[i] )
        data["uppercutsO"].append( fight.pB.uppercuts[i] )
        data["uppercutsO"].append( fight.pR.uppercuts[i] )

        data["minO"].append( fight.pB.min[i] )
        data["minO"].append( fight.pR.min[i] )
        data["lowO"].append( fight.pB.low[i] )
        data["lowO"].append( fight.pR.low[i] )
        data["midO"].append( fight.pB.mid[i] )
        data["midO"].append( fight.pR.mid[i] )
        data["highO"].append( fight.pB.high[i] )
        data["highO"].append( fight.pR.high[i] )
        data["maxO"].append( fight.pB.max[i] )
        data["maxO"].append( fight.pR.max[i] )
        data["minMissO"].append( fight.pB.minMiss[i] )
        data["minMissO"].append( fight.pR.minMiss[i] )
        data["lowMissO"].append( fight.pB.lowMiss[i] )
        data["lowMissO"].append( fight.pR.lowMiss[i] )
        data["midMissO"].append( fight.pB.midMiss[i] )
        data["midMissO"].append( fight.pR.midMiss[i] )
        data["highMissO"].append( fight.pB.highMiss[i] )
        data["highMissO"].append( fight.pR.highMiss[i] )
        data["maxMissO"].append( fight.pB.maxMiss[i] )
        data["maxMissO"].append( fight.pR.maxMiss[i] )
        data["lowCommitMissO"].append( fight.pB.minMiss[i] + fight.pB.lowMiss[i] )
        data["lowCommitMissO"].append( fight.pR.minMiss[i] + fight.pR.lowMiss[i] )
        data["highCommitMissO"].append( fight.pB.midMiss[i] + fight.pB.highMiss[i] + fight.pB.maxMiss[i] )
        data["highCommitMissO"].append( fight.pR.midMiss[i] + fight.pR.highMiss[i] + fight.pR.maxMiss[i] )

    for key, value in data.items():
        if len(value) != len(data["round"]):
            print(f"{Fore.RED}ERROR: Unequal list length for key: {key} ({len(value)} != {len(data['round'])}){Fore.WHITE}")

    return data



# takes in neutralData and makes a new dict with corresponding +/-s
def getPMs(data, params, paramValues, allParams, dampener=50.0, sharpness=2.0):
    """
    takes in neutralData and makes a new dict with corresponding +/-s
    now supports the new ratio-based scoring equation
    """
    pmData = dict(
        scores=data["scores"],
        color=data["color"],

        # stats for both fighters
        totalThrown=data["totalThrown"],
        totalLanded=data["totalLanded"],

        # ratio stats
        thrownShare=data["thrownShare"],
        landedShare=data["landedShare"],

        # fighter stats
        thrown=[],
        landed=[],
        missed = [],
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
        lowCommitMiss=[],
        highCommitMiss=[],
        orthodox = [],
        southpaw = [],
        squared = [],
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
        head = [],
        body = [],
        straights = [],
        hooks = [],
        overhands = [],
        uppercuts = [],
        heuristic=[]
    )

    if paramValues is not None:
        # import the heuristic function for calculations
        from gradientDescent import heuristic
        
        # calculate heuristic using new equation
        heuristicScores = heuristic(paramValues, data, params, dampener, sharpness)

        # ensure heuristicScores is a flat list of numbers
        if isinstance(heuristicScores, np.ndarray):
            pmData['heuristic'] = heuristicScores.flatten().tolist()
        else:
            pmData['heuristic'] = list(heuristicScores)


    # iterate over each stat and calculate round-by-round difference
    for stat in allParams:
        for i in range(len(data['scores'])):
            if stat == 'heuristic':  # when graphing heuristic we don't plot +/-
                if paramValues is not None:
                    pm = pmData['heuristic'][i]  # already calculated above
                else:
                    pm = 0  # default value if no parameters provided
            elif 'total' in stat or 'Share' in stat:     # use raw values for total stats and ratio stats
                pm = data[stat][i]
            else:
                pm = data[stat][i] - data[stat + 'O'][i]
            pmData[stat].append(pm)

    return pmData


# takes pmData and normalizes it
# returns normalized data and correlation metrics (Pearson and Spearman)
def normalizeData(pmData, attributes):
    scaler = StandardScaler()       # initialize scaler
    
    # Create arrays for normalization
    dataArray = np.array([[pmData[attr][i] for attr in attributes] for i in range(len(pmData['scores']))])
    
    # Fit and transform the data
    normalizedArray = scaler.fit_transform(dataArray)
    
    # Create new dictionary with normalized data
    normalizedData = pmData.copy()  # Copy all original data
    for i, attr in enumerate(attributes):
        if attr != 'scores':  # Don't normalize scores
            normalizedData[attr] = normalizedArray[:, i]
        # ? should I be normalizing heuristic values (ie my predictions)
    
    # Calculate correlations
    correlations = {}
    for attr in attributes:
        if attr != 'scores' and 'total' not in attr:
            # Pearson correlation
            pearsonR, pearsonP = stats.pearsonr(normalizedData[attr], normalizedData['scores'])
            # Spearman correlation
            spearmanR, spearmanP = stats.spearmanr(normalizedData[attr], normalizedData['scores'])
            correlations[attr] = {
                'pearson': {'r': pearsonR, 'p': pearsonP},
                'spearman': {'r': spearmanR, 'p': spearmanP}
            }
    
    return normalizedData, correlations


def normalizeRawData(data, parameters):
    """
    normalize raw data dictionary for gradient descent while preserving original values

    This function normalizes all stat parameters (both fighter and opponent versions)
    using StandardScaler to put all features on the same scale. This makes gradient
    descent more stable and weights directly interpretable.

    Args:
        data: original data dictionary with raw values
        parameters: list of parameter names to normalize

    Returns:
        normalizedData: copy of data dictionary with normalized stat values
        scaler: fitted StandardScaler object for potential later use
    """
    from sklearn.preprocessing import StandardScaler

    # create a copy of the data to avoid modifying original
    normalizedData = {}
    for key in data.keys():
        if isinstance(data[key], list):
            normalizedData[key] = data[key].copy()
        else:
            normalizedData[key] = data[key]

    # collect all stat columns that need normalization
    # we need to normalize both fighter and opponent versions together
    statsToNormalize = []
    for param in parameters:
        if param not in ['heuristic']:  # skip heuristic as it's not raw data
            statsToNormalize.append(param)
            # also add opponent version if it exists and isn't a total/share stat
            if 'total' not in param and 'Share' not in param:
                if param + 'O' in data:
                    statsToNormalize.append(param + 'O')

    # remove duplicates while preserving order
    statsToNormalize = list(dict.fromkeys(statsToNormalize))

    # create matrix for normalization: rows = samples, columns = features
    numSamples = len(data['scores'])
    dataMatrix = np.zeros((numSamples, len(statsToNormalize)))

    for i, stat in enumerate(statsToNormalize):
        if stat in data:
            dataMatrix[:, i] = data[stat]

    # fit scaler and transform data
    scaler = StandardScaler()
    normalizedMatrix = scaler.fit_transform(dataMatrix)

    # put normalized values back into the dictionary
    for i, stat in enumerate(statsToNormalize):
        normalizedData[stat] = normalizedMatrix[:, i].tolist()

    print(f"Normalized {len(statsToNormalize)} stat columns for gradient descent")
    print(f"  Features are now on same scale (mean=0, std=1)")
    print(f"  This makes weights directly interpretable and GD more stable")

    return normalizedData, scaler

