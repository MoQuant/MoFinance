import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import rcParams

rcParams['figure.autolayout'] = True

        
def Presidents():
    names = ('FDR', 'HarryTruman', 'DwightEisenhower', 'JFK', 'LBJ', 'RichardNixon',
             'GeraldFord', 'JimmyCarter', 'RonaldReagan', 'GHWB', 'BillClinton', 'GWB', 'Obama', 'DT','Biden')
    labels = ('Franklin D. Roosevelt', 'Harry Truman', 'Dwight Eisenhower',
                     'John F. Kennedy', 'Lyndon Johnson', 'Richard Nixon', 'Gerald Ford', 'Jimmy Carter', 'Ronald Reagan',
                     'George H.W. Bush', 'Bill Clinton', 'George W. Bush', 'Barack Obama', 'Donald Trump','Joe Biden')

    terms = ((1933, 1945), (1945, 1953), (1953, 1961), (1961, 1963), (1963, 1969),
             (1969, 1974), (1974, 1977), (1977, 1981), (1981, 1989), (1989, 1993), (1993, 2001),
             (2001, 2009), (2009, 2017), (2017, 2021), (2021, 2025))

    return names, labels, terms


def getPresident(names, i, point):
    image = plt.imread('Presidents/{}.png'.format(names[i]))
    result = OffsetImage(image, zoom=point)
    return result

def plotImage(ax, picture, stdev, mean):
    usa = AnnotationBbox(picture, (stdev, mean), xycoords='data', frameon=False)
    ax.add_artist(usa)

def fetchData(df, t0, t1):
    df = df[(df['Date'] >= t0) & (df['Date'] <= t1)]
    df = df['Price'].values
    ror = df[1:]/df[:-1] - 1.0
    sd = np.std(ror)
    mu = np.mean(ror)
    return sd, mu, mu / sd

names, labels, terms = Presidents()

fig = plt.figure()
ax = fig.add_subplot(111)

bg = 'black'
fg = 'limegreen'

fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)

for p in ('x', 'y'):
    ax.tick_params(p, colors=fg)

ax.set_title("Presidential Performance via Sharpe Ratio", color=fg)
ax.set_xlabel("Presidential Risk", color=fg)
ax.set_ylabel("Presidential Return", color=fg)

df = pd.read_csv('SP500.csv')[::-1]

df['Date'] = list(map(lambda x: int(x.split(' ')[-1]), df['Date']))
df['Price'] = list(map(lambda x: float(x.replace(',','')) if ',' in x else float(x), df['Price']))

sd, mu, sharpe = [], [], []

for i, (t0, t1) in enumerate(terms):
    x, y, z = fetchData(df, t0, t1)
    sd.append(x)
    mu.append(y)
    sharpe.append(z)

shp = [(s - np.min(sharpe))/(np.max(sharpe) - np.min(sharpe)) for s in sharpe]

alpha = 0.2
ax.set_xlim(np.min(sd)*(1-alpha), np.max(sd)*(1+alpha))
ax.set_ylim(np.min(mu)*(1-alpha), np.max(mu)*(1+alpha))

for i, (x, y, z) in enumerate(zip(sd, mu, shp)):
    prez = getPresident(names, i, (alpha*2*z))
    plotImage(ax, prez, x, y)



plt.show()
