import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Import the data from medical_examination.csv and assign it to the df variable.
df = pd.read_csv('medical_examination.csv')

# 2. Add an overweight column to the data.
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3. Normalize data by making 0 always good and 1 always bad for cholesterol and gluc.
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

def draw_cat_plot():
    # 4. Draw the Categorical Plot in the draw_cat_plot function.
    # 5. Create a DataFrame for the cat plot using pd.melt with values from cholesterol, gluc, smoke, alco, active, and overweight in the df_cat variable.
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6. Group and reformat the data in df_cat to split it by cardio. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7. Convert the data into long format and create a chart that shows the value counts of the categorical features using sns.catplot().
    g = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar'
    )

    # 8. Get the figure for the output and store it in the fig variable.
    fig = g.fig
    # Do not modify the next two lines.
    return fig

def draw_heat_map():
    # 9. Draw the Heat Map in the draw_heat_map function.
    # 10. Clean the data in the df_heat variable by filtering out incorrect data.
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # 11. Calculate the correlation matrix and store it in the corr variable.
    corr = df_heat.corr()

    # 12. Generate a mask for the upper triangle and store it in the mask variable.
    import numpy as np
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 13. Set up the matplotlib figure.
    fig, ax = plt.subplots(figsize=(12, 8))

    # 14. Plot the correlation matrix using sns.heatmap().
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', center=0, vmax=.3, vmin=-.1, square=True, linewidths=.5, cbar_kws={"shrink": .5})

    # Do not modify the next two lines.
    return fig

if __name__ == "__main__":
    # Draw and save categorical plot
    cat_fig = draw_cat_plot()
    cat_fig.savefig("catplot.png")

    # Draw and save heatmap
    heat_fig = draw_heat_map()
    heat_fig.savefig("heatmap.png") 