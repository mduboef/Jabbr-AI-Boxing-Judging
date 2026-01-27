import json
import os
from datetime import datetime
from colorama import Fore
from tqdm import tqdm
from dataProcessing import Fight, Performance

def formatDate(dateArray):
    if dateArray and len(dateArray) == 3:
        day, month, year = dateArray
        # return MM/DD/YYYY format to match what pairFightsToCards expects
        return f"{month:02d}/{day:02d}/{year}"
    else:
        # return a default date if invalid
        return datetime.now().strftime("%m/%d/%Y")


# parse a single JSON file
def parseQuadJsonFile(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data             # return dict containing parsed fight data
    except Exception as e:
        print(f"{Fore.RED}ERROR: Failed to parse {filepath}: {str(e)}{Fore.WHITE}")
        return None


# extract stats for a single fighter from a single round
# returns dict containing all extracted stats
def extractFighterStatsFromRound(fighterData):
    stats = {}
    
    # extract punch stats
    punches = fighterData.get('Punches', {})
    stats['thrown'] = punches.get('Thrown', 0)
    stats['landed'] = punches.get('Landed', 0)
    stats['accuracy'] = punches.get('Accuracy', 0) * 100  # convert to percentage
    stats['highImpact'] = punches.get('Landed-High-Impact', 0)
    
    # extract impact categories (min, low, mid, high, max)
    impactCategory = punches.get('Impact-Category', {})
    stats['min'] = impactCategory.get('1-Min', 0)
    stats['low'] = impactCategory.get('2-Low', 0)
    stats['mid'] = impactCategory.get('3-Mid', 0)
    stats['high'] = impactCategory.get('4-High', 0)
    stats['max'] = impactCategory.get('5-Max', 0)
    
    # initialize miss counters
    stats['minMiss'] = 0
    stats['lowMiss'] = 0
    stats['midMiss'] = 0
    stats['highMiss'] = 0
    stats['maxMiss'] = 0
    
    # extract miss counts from Combinations section
    # each combination entry represents a punch: ["punch type", quality, powerCommit]
    # quality 0 = miss, powerCommit 1-5 maps to minMiss-maxMiss
    combinations = fighterData.get('Combinations', {})
    combinationsList = combinations.get('List', [])
    
    for combination in combinationsList:
        # each combination is a list of punches in that combination
        if isinstance(combination, list):
            for punch in combination:
                if isinstance(punch, list) and len(punch) >= 3:
                    punchType = punch[0]  # string like "L Hook Head"
                    quality = punch[1]    # 0 = miss, 1-5 = landed quality
                    powerCommit = punch[2]  # 1-5 power commit level
                    
                    # only process misses (quality == 0)
                    if quality == 0:
                        # map powerCommit (1-5) to appropriate miss counter
                        if powerCommit == 1:
                            stats['minMiss'] += 1
                        elif powerCommit == 2:
                            stats['lowMiss'] += 1
                        elif powerCommit == 3:
                            stats['midMiss'] += 1
                        elif powerCommit == 4:
                            stats['highMiss'] += 1
                        elif powerCommit == 5:
                            stats['maxMiss'] += 1
    
    # extract compound metrics
    compoundMetrics = fighterData.get('Compound Metrics', {})
    stats['pressure'] = compoundMetrics.get('Pressure', 0) * 100
    stats['pressureDistance'] = compoundMetrics.get('Pressure-distance', 0) * 100
    stats['pressureMovement'] = compoundMetrics.get('Pressure-movement', 0) * 100
    stats['pressurePosition'] = compoundMetrics.get('Pressure-position', 0) * 100
    stats['aggression'] = compoundMetrics.get('Aggression', 0) * 100
    stats['aggressionCombinations'] = compoundMetrics.get('Aggression-combinations', 0) * 100
    stats['aggressionExchanges'] = compoundMetrics.get('Aggression-exchanges', 0) * 100
    stats['aggressionPower'] = compoundMetrics.get('Aggression-power', 0) * 100
    
    # extract combinations overview
    combinations = fighterData.get('Combinations', {}).get('Overview', {})
    stats['singles'] = combinations.get('Singles', [0, 0, 0])[2] * 100 if combinations.get('Singles') else 0
    stats['doubles'] = combinations.get('Doubles', [0, 0, 0])[2] * 100 if combinations.get('Doubles') else 0
    stats['triples'] = combinations.get('Triples', [0, 0, 0])[2] * 100 if combinations.get('Triples') else 0
    stats['quadsPlus'] = combinations.get('Quads+', [0, 0, 0])[2] * 100 if combinations.get('Quads+') else 0
    
    # extract distance stats
    distance = fighterData.get('Distance', {})
    stats['outside'] = distance.get('Outside', 0) * 100
    stats['midrange'] = distance.get('Mid-Range', 0) * 100
    stats['inside'] = distance.get('Inside', 0) * 100
    stats['clinch'] = distance.get('Clinch', 0) * 100
    
    # extract stance stats
    stance = fighterData.get('Stance', {})
    stats['orthodox'] = stance.get('Orthodox', 0) * 100
    stats['southpaw'] = stance.get('Southpaw', 0) * 100
    stats['squared'] = stance.get('Squared', 0) * 100

    # extract balance stats
    balance = fighterData.get('Balance', {})
    stats['backfoot'] = balance.get('Back-Foot', 0) * 100
    stats['frontfoot'] = balance.get('Front-Foot', 0) * 100
    stats['neutral'] = balance.get('Neutral', 0) * 100
    
    # extract type counts for punch types
    typeCount = punches.get('Type-Count', {})
    
    # initialize punch type counters
    stats['lead'] = 0
    stats['rear'] = 0
    stats['head'] = 0
    stats['body'] = 0
    stats['straights'] = 0
    stats['hooks'] = 0
    stats['overhands'] = 0
    stats['uppercuts'] = 0
    
    # determine dominant stance for lead/rear calculation
    orthodoxPercent = stance.get('Orthodox', 0)
    southpawPercent = stance.get('Southpaw', 0)
    isOrthodox = orthodoxPercent >= southpawPercent
    
    for punchType, punchData in typeCount.items():
        if len(punchData) >= 2:
            landedCount = punchData[0]  # landed punches of this type
            
            # determine punch category
            if 'Straight' in punchType:
                stats['straights'] += landedCount
            elif 'Hook' in punchType:
                stats['hooks'] += landedCount
            elif 'Overhand' in punchType:
                stats['overhands'] += landedCount
            elif 'Uppercut' in punchType:
                stats['uppercuts'] += landedCount
            
            # determine target
            if 'Head' in punchType:
                stats['head'] += landedCount
            elif 'Body' in punchType:
                stats['body'] += landedCount
            
            # determine lead/rear based on hand and stance
            if punchType.startswith('L'):
                if isOrthodox:
                    stats['lead'] += landedCount  # left is lead for orthodox
                else:
                    stats['rear'] += landedCount  # left is rear for southpaw
            elif punchType.startswith('R'):
                if isOrthodox:
                    stats['rear'] += landedCount  # right is rear for orthodox
                else:
                    stats['lead'] += landedCount  # right is lead for southpaw
    
    return stats


# create a single fight object from parsed JSON data
def createFightFromJson(jsonData, filename):
    # extract fighter names from the Names section
    names = jsonData.get('Names', {})
    redName = names.get('red', '')
    blueName = names.get('blue', '')
    
    if not redName or not blueName:
        print(f"{Fore.RED}ERROR: Could not extract fighter names from {filename}{Fore.WHITE}")
        return None
    
    # extract match info from A-Match-Info section
    matchInfo = jsonData.get('A-Match-Info', {})
    
    # extract and format date
    matchDate = matchInfo.get('match-date', [1, 1, 2024])
    date = formatDate(matchDate)
    
    # extract rounds information
    roundsFought = matchInfo.get('rounds-fought', 0)
    roundsScheduled = matchInfo.get('rounds-scheduled', 0)
    rounds = roundsFought  # use rounds actually fought
    
    # extract decision and winner
    decision = matchInfo.get('decision', 'UD')
    winnerColor = matchInfo.get('winner', '')
    
    # determine winner name based on color
    if winnerColor == 'red':
        winner = redName
    elif winnerColor == 'blue':
        winner = blueName
    else:
        winner = ''  # draw or no contest

    # validate rounds count by checking what round keys actually exist
    actualRounds = [k for k in jsonData.keys() if k.startswith('Round ') and len(k.split()) == 2 and k.split()[1].isdigit()]
    actualRoundsCount = len(actualRounds)
    
    if rounds == 0:
        # fallback to counting actual round keys in the JSON
        rounds = actualRoundsCount
        if rounds == 0:
            print(f"{Fore.RED}ERROR: No round data found in {filename}{Fore.WHITE}")
            return None
    elif actualRoundsCount < rounds:
        # metadata says more rounds than actually exist, use actual count
        print(f"{Fore.YELLOW}WARNING: {filename} metadata indicates {rounds} rounds but only {actualRoundsCount} found. Using {actualRoundsCount}.{Fore.WHITE}")
        rounds = actualRoundsCount

    # create Fight object
    fight = Fight(redName, blueName, date, rounds, decision, winner, filename)
    
    # extract stats for each round
    redStats = {
        'thrown': [], 'landed': [], 'highImpact': [], 'pressure': [], 'aggression': [],
        'accuracy': [], 'min': [], 'low': [], 'mid': [], 'high': [], 'max': [],
        'minMiss': [], 'lowMiss': [], 'midMiss': [], 'highMiss': [], 'maxMiss': [],
        'singles': [], 'doubles': [], 'triples': [], 'quadsPlus': [],
        'backfoot': [], 'frontfoot': [], 'neutral': [],
        'outside': [], 'midrange': [], 'inside': [], 'clinch': [],
        'aggressionCombinations': [], 'aggressionExchanges': [], 'aggressionPower': [],
        'pressureDistance': [], 'pressureMovement': [], 'pressurePosition': [],
        'orthodox': [], 'southpaw': [], 'squared': [],
        'lead': [], 'rear': [], 'head': [], 'body': [],
        'straights': [], 'hooks': [], 'overhands': [], 'uppercuts': []
    }
    
    blueStats = {
        'thrown': [], 'landed': [], 'highImpact': [], 'pressure': [], 'aggression': [],
        'accuracy': [], 'min': [], 'low': [], 'mid': [], 'high': [], 'max': [],
        'minMiss': [], 'lowMiss': [], 'midMiss': [], 'highMiss': [], 'maxMiss': [],
        'singles': [], 'doubles': [], 'triples': [], 'quadsPlus': [],
        'backfoot': [], 'frontfoot': [], 'neutral': [],
        'outside': [], 'midrange': [], 'inside': [], 'clinch': [],
        'aggressionCombinations': [], 'aggressionExchanges': [], 'aggressionPower': [],
        'pressureDistance': [], 'pressureMovement': [], 'pressurePosition': [],
        'orthodox': [], 'southpaw': [], 'squared': [],
        'lead': [], 'rear': [], 'head': [], 'body': [],
        'straights': [], 'hooks': [], 'overhands': [], 'uppercuts': []
    }
    
    # process each round
    for roundNum in range(1, rounds + 1):
        roundKey = f'Round {roundNum:02d}'  # creates "Round 01", "Round 02", etc.
        
        if roundKey not in jsonData:
            print(f"{Fore.YELLOW}WARNING: {roundKey} not found in {filename}{Fore.WHITE}")
            # pad with zeros for missing round
            for key in redStats:
                redStats[key].append(0)
            for key in blueStats:
                blueStats[key].append(0)
            continue  # skip to next round instead of trying to access missing data
        
        # only access roundData if the round exists
        roundData = jsonData[roundKey]

        # extract red fighter stats using the actual fighter name
        if "red" in roundData:
            redRoundStats = extractFighterStatsFromRound(roundData["red"])
            for key in redStats:
                redStats[key].append(redRoundStats.get(key, 0))
        else:
            # pad with zeros if fighter data missing
            for key in redStats:
                redStats[key].append(0)
        
        # extract blue fighter stats using the actual fighter name
        if "blue" in roundData:
            blueRoundStats = extractFighterStatsFromRound(roundData["blue"])
            for key in blueStats:
                blueStats[key].append(blueRoundStats.get(key, 0))
        else:
            # pad with zeros if fighter data missing
            for key in blueStats:
                blueStats[key].append(0)
    
    # create Performance objects
    redPerformance = Performance(
        redStats['thrown'], redStats['landed'], redStats['highImpact'],
        redStats['pressure'], redStats['aggression'], redStats['accuracy'],
        redStats['min'], redStats['low'], redStats['mid'], redStats['high'], redStats['max'],
        redStats['minMiss'], redStats['lowMiss'], redStats['midMiss'], redStats['highMiss'], redStats['maxMiss'],
        redStats['singles'], redStats['doubles'], redStats['triples'], redStats['quadsPlus'],
        redStats['backfoot'], redStats['frontfoot'], redStats['neutral'],
        redStats['outside'], redStats['midrange'], redStats['inside'], redStats['clinch'],
        redStats['aggressionCombinations'], redStats['aggressionExchanges'], redStats['aggressionPower'],
        redStats['pressureDistance'], redStats['pressureMovement'], redStats['pressurePosition'],
        redStats['orthodox'], redStats['southpaw'], redStats['squared'],
        redStats['lead'], redStats['rear'], redStats['head'], redStats['body'],
        redStats['straights'], redStats['hooks'], redStats['overhands'], redStats['uppercuts']
    )
    
    bluePerformance = Performance(
        blueStats['thrown'], blueStats['landed'], blueStats['highImpact'],
        blueStats['pressure'], blueStats['aggression'], blueStats['accuracy'],
        blueStats['min'], blueStats['low'], blueStats['mid'], blueStats['high'], blueStats['max'],
        blueStats['minMiss'], blueStats['lowMiss'], blueStats['midMiss'], blueStats['highMiss'], blueStats['maxMiss'],
        blueStats['singles'], blueStats['doubles'], blueStats['triples'], blueStats['quadsPlus'],
        blueStats['backfoot'], blueStats['frontfoot'], blueStats['neutral'],
        blueStats['outside'], blueStats['midrange'], blueStats['inside'], blueStats['clinch'],
        blueStats['aggressionCombinations'], blueStats['aggressionExchanges'], blueStats['aggressionPower'],
        blueStats['pressureDistance'], blueStats['pressureMovement'], blueStats['pressurePosition'],
        blueStats['orthodox'], blueStats['southpaw'], blueStats['squared'],
        blueStats['lead'], blueStats['rear'], blueStats['head'], blueStats['body'],
        blueStats['straights'], blueStats['hooks'], blueStats['overhands'], blueStats['uppercuts']
    )
    
    # add performances to fight
    fight.addRed(redPerformance)
    fight.addBlue(bluePerformance)
    
    return fight



# read json stat files and return list of Fight objects
def readQuadStats(commandLine):

    # impossible to run with -includeinserted flag
    if '-includeinserted' in commandLine:
        raise ValueError(f"{Fore.RED}ERROR: -includeinserted flag cannot be used with -quadcam{Fore.WHITE}")
    
    fights = []
    
    # get the directory containing the JSON files
    directory = os.getcwd() + "/quadStats/"
    
    # check if directory exists
    if not os.path.exists(directory):
        print(f"{Fore.RED}ERROR: quadStats directory not found at {directory}{Fore.WHITE}")
        return fights
    
    # list all JSON files in the directory
    jsonFiles = [f for f in os.listdir(directory) if f.endswith(".json")]
    
    if len(jsonFiles) == 0:
        print(f"{Fore.YELLOW}WARNING: No JSON files found in {directory}{Fore.WHITE}")
        return fights
    
    pbar = tqdm(total=len(jsonFiles), desc="Reading quadStats", unit="file")
    
    for filename in jsonFiles:
        # construct the full path to the JSON file
        filepath = os.path.join(directory, filename)
        
        # parse JSON file
        jsonData = parseQuadJsonFile(filepath)
        if jsonData is None:
            pbar.update(1)
            continue
        
        # create Fight object from JSON data
        fight = createFightFromJson(jsonData, filename)
        if fight is not None:
            fights.append(fight)
        
        pbar.update(1)
    
    pbar.close()
    
    print(f"{Fore.GREEN}Successfully loaded {len(fights)} fights from quadStats{Fore.WHITE}")
    
    return fights




# exclude specific rounds from specific files when using -ignorescores and -singlecam
def excludeSpecificRounds(fights, useIgnoreScores, useSingleCam):
    if not (useIgnoreScores and useSingleCam):
        return
    
    # define rounds to exclude for specific files
    exclusionRules = {
        'jabbr-statistics-K5VlNMVAbvqgoJk.xlsx': [1, 6, 7, 8],
        'jabbr-statistics-LGwZx3Xj6MaDQb9.xlsx': [4, 5],
        'jabbr-statistics-apZYV376BWqK5jm.xlsx': [5, 6, 7, 8, 9, 10],
        'jabbr-statistics-GVpgmW1xQM5QJao.xlsx': [1, 2, 3, 8],
        'jabbr-statistics-K5VlNMVxb3qgoJk.xlsx': [3, 4, 5, 6]
    }
    
    numToExclude = sum(len(v) for v in exclusionRules.values())
    excludedCount = 0
    
    for fight in fights:
        if fight.file in exclusionRules:
            roundsToExclude = exclusionRules[fight.file][:]  # create a copy to iterate over
            originalLength = len(fight.pR.landed)  # store original length before exclusions
            
            # sort rounds in reverse order so we exclude from highest to lowest
            # this prevents index shifting issues
            roundsToExclude.sort(reverse=True)
            
            for roundNum in roundsToExclude:
                # convert to 0-based index for excludeRound method
                if roundNum <= originalLength:  # use original length, not current length
                    fight.excludeRound(roundNum-1)
                    excludedCount += 1
                    exclusionRules[fight.file].remove(roundNum)  # remove from original for tracking

    
    if excludedCount > 0:
        print(f"{Fore.YELLOW}INFO: Excluded {excludedCount} specified rounds from single-cam files{Fore.WHITE}")
    if excludedCount < numToExclude:
        print(f"{Fore.YELLOW}WARNING: Some specified rounds to exclude were not found in the files{Fore.WHITE}")
        print(f"{Fore.YELLOW}Remaining exclusions: {exclusionRules}{Fore.WHITE}")