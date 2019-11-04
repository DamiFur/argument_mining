import seaborn as sns
import colors

sns.set_style('whitegrid')
sns.set_palette('colorblind')
sns.set_context('paper', font_scale=2)

att_palette = {
    'No Attention': colors.dark_blue,
    'Word Attention': colors.green,
    'Context Attention': colors.blue,
    'Self Attention': colors.yellow,
    'Bert': colors.red
}

activation_palette = {
    'sigmoid': colors.yellow,
    'tanh': colors.green,
    'linear': colors.blue,
    'relu': colors.red,
    'na': colors.gray
}
