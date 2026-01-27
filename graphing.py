import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# take the raw data, normalizes it and plot the correlation matrix of the specified parameters
def plotCorrMatrix(data, parameters):
    # convert data into pandas DataFrame
    rawData = {}
    for param in parameters:
        rawData[param] = np.array(data[param])
    df = pd.DataFrame(rawData)
    
    # normalize data for better visualization
    # doesn't affect correlation calculation but helps with interpretation?
    scaler = StandardScaler()
    normalizedData = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns
    )
    
    # calculate the correlation matrix values
    corrMatrix = normalizedData.corr(method='pearson')
    
    # dynamically size the figure based on number of parameters
    numParams = len(parameters)
    figSize = max(12, numParams * 0.8)  # minimum 12, scale with parameter count
    
    # create plot with larger size
    plt.figure(figsize=(figSize, figSize))
        
    # create heatmap with adjusted font sizes
    cmap = sns.diverging_palette(10, 130, as_cmap=True)  # red to Green
    
    # adjust annotation font size based on number of parameters
    annotationFontSize = max(6, 14 - (numParams * 0.15))  # smaller font for more params
    labelFontSize = max(8, 16 - (numParams * 0.1))
    
    sns.heatmap(
        corrMatrix, 
        cmap=cmap, 
        vmax=1.0, 
        vmin=-1.0,
        center=0,
        square=True, 
        linewidths=.5, 
        annot=True,  # add correlation values in cells
        fmt=".2f",   # format as 2 decimal places
        annot_kws={"size": annotationFontSize},  # adjust annotation font size
        cbar_kws={"shrink": .8}
    )
    
    plt.title('Correlation Matrix of Boxing Performance Metrics', fontsize=16, pad=20)
    
    # rotate labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=labelFontSize)
    plt.yticks(rotation=0, fontsize=labelFontSize)
    
    # adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # save the plot to current working directory with high DPI for better quality
    plt.savefig('correlationMatrix.png', dpi=300, bbox_inches='tight')
    
    # show the plot
    # plt.show()
    
    print("\nCompleted:\tCorrelation matrix of parameters\n")
    
    return corrMatrix   # return the correlation matrix in case we wanna analyize it later



# create scatter plots
def plotScatters(pmData, attributes, save_dir):
    df = pd.DataFrame(pmData)

    # Create a directory for saving plots if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Plot each attribute against 'scores' and save the plots
    for attribute in attributes:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=attribute, y='scores')
        if attribute == 'heuristic':
            plt.title('Label vs Prediction')
            plt.xlabel('Predicted Value')
        else:
            plt.title(f'Label vs {attribute} +/-')
            plt.xlabel(f'{attribute} +/-')
        plt.ylabel('Label')
        plt.savefig(os.path.join(save_dir, f'{attribute}.png'))  # Save the figure in 'scatters' folder
        plt.close()


# plot histograms with non-normalized data
def plotHistograms(pmData, attributes, outputDir='histograms', numBins=20):

    os.makedirs(outputDir, exist_ok=True)

    for attribute in attributes:
        dataByScore = {score: [] for score in [-1, -1 / 3, 1 / 3, 1]}
        for i, score in enumerate(pmData["scores"]):
            dataByScore[score].append(pmData[attribute][i])

        allData = sum(dataByScore.values(), [])
        minValue = min(allData)
        maxValue = max(allData)
        halfBins = numBins // 2

        if attribute in ['thrown', 'landed', 'missed', 'min', 'low', 'mid', 'high', 'max', 'highImpact',
                         'minMiss', 'lowMiss', 'midMiss', 'highMiss', 'maxMiss', 'lowCommitMiss', 'highCommitMiss']:
            maxBound = max(abs(minValue), abs(maxValue))
            step = 1
            while ((step+2) * halfBins)-((step+2)/2) < maxBound:
                step += 2

            negativeBins = np.linspace((-step * halfBins)+(step/2), -step/2, halfBins)
            positiveBins = np.linspace(step/2, (step * halfBins)-(step/2), halfBins)
            bins = np.concatenate((negativeBins, positiveBins))
            
            # Set xlim to the full range of the bins
            plot_min = (-step * halfBins)+(step/2) - step/2
            plot_max = (step * halfBins)-(step/2) + step/2
        else:
            if minValue < 0 < maxValue:
                maxBound = max(abs(minValue), abs(maxValue))
                negativeBins = np.linspace(-maxBound, 0, halfBins + 1)
                positiveBins = np.linspace(0, maxBound, halfBins + 1)[1:]
                bins = np.concatenate((negativeBins, positiveBins))
                plot_min, plot_max = -maxBound, maxBound
            else:
                bins = np.linspace(minValue, maxValue, numBins + 1)
                plot_min, plot_max = minValue, maxValue

        colors = {-1: 'darkred', -1/3: 'tomato', 1/3: 'mediumseagreen', 1: 'darkgreen'}

        fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        plt.subplots_adjust(hspace=0)

        for idx, (score, data) in enumerate(dataByScore.items()):
            frac = "0" if score == -1 else "1" if score == -1/3 else "2" if score == 1/3 else "3" if score == 1 else str(score)
            axs[idx].hist(data, bins=bins, color=colors[score], alpha=0.5, edgecolor='black')
            axs[idx].set_xlim(plot_min, plot_max)  # Set xlim to the full range
            axs[idx].set_ylabel(f"Label: {frac}/3 Judges\nFrequency")
            
            if idx == 0:
                axs[idx].set_title(f"Distribution of Predicted Values" if attribute == 'heuristic' else f"{attribute} Distribution")

            if idx < len(dataByScore) - 1:
                axs[idx].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            else:
                axs[idx].set_xticks(bins)
                axs[idx].set_xticklabels([f'{bin:.1f}' for bin in bins])
                if attribute == 'heuristic':
                    axs[idx].set_xlabel("Predicted Value")
                else:
                    axs[idx].set_xlabel(f"{attribute} +/-")

        maxYValue = max(ax.get_ylim()[1] for ax in axs)
        for ax in axs:
            ax.set_ylim(0, maxYValue)

        plt.savefig(os.path.join(outputDir, f"{attribute}.png"))
        plt.clf()
        plt.close()


# plot histograms with normalized data
def plotNormalHistograms(normalizedData, attributes, outputDir='normalizedHistograms'):

    os.makedirs(outputDir, exist_ok=True)
    
    # Create consistent bins for normalized histograms
    normalizedBins = np.linspace(-3, 3, 20)
    
    # Create custom bins for heuristic with specified boundaries
    heuristicBins = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 
                     0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    for attribute in attributes:
        if attribute == 'scores' or attribute == 'color':
            continue  # Skip non-normalized attributes
            
        # Select the appropriate bins based on attribute
        if attribute == 'heuristic':
            bins = heuristicBins
        else:
            bins = normalizedBins
        
        # Organize data by score
        dataByScore = {score: [] for score in [-1, -1/3, 1/3, 1]}
        for i, score in enumerate(normalizedData["scores"]):
            dataByScore[score].append(normalizedData[attribute][i])
        
        # Create figure with 4 subplots (one for each score value)
        fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        plt.subplots_adjust(hspace=0)
        
        # Define colors for each score category
        colors = {-1: 'darkred', -1/3: 'tomato', 1/3: 'mediumseagreen', 1: 'darkgreen'}
        
        # Plot histograms for each score value
        for idx, (score, data) in enumerate(dataByScore.items()):
            # Convert score to fraction string for label
            frac = "0" if score == -1 else "1" if score == -1/3 else "2" if score == 1/3 else "3"
            
            # Create histogram
            axs[idx].hist(data, bins=bins, color=colors[score], alpha=0.5, edgecolor='black')
            axs[idx].set_xlim(bins[0], bins[-1])
            axs[idx].set_ylabel(f"Label: {frac}/3 Judges\nFrequency")
            
            # Only add title to top subplot
            if idx == 0:
                if attribute == 'heuristic':
                    axs[idx].set_title(f"Distribution of Predicted Values")
                else:
                    axs[idx].set_title(f"Normalized {attribute} Distribution")
            
            # Only show x-axis labels on bottom subplot
            if idx < len(dataByScore) - 1:
                axs[idx].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            else:
                if attribute == 'heuristic':
                    axs[idx].set_xlabel("Predicted Value")
                else:
                    axs[idx].set_xlabel(f"Normalized {attribute} +/- (Standard Deviations)")
        
        # Make y-axis limits consistent across subplots
        maxYValue = max(ax.get_ylim()[1] for ax in axs)
        for ax in axs:
            ax.set_ylim(0, maxYValue)
        
        # Save the figure
        plt.savefig(os.path.join(outputDir, f"{attribute}.png"))
        plt.close(fig)

    