"""Render the case study from the saved evaluation; no refitting needed."""
import json
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd
import seaborn as sns

from data import ROOT
from study import BANDS

OUT, WORK = ROOT/'outputs', ROOT/'data'
INK, MUTED, BLUE, TEAL, ORANGE, LINE = '#152c42', '#425a6d', '#4978a2', '#007e78', '#a84d24', '#d9e4eb'


def page(number, label, title):
    fig = plt.figure(figsize=(8.5,11),facecolor='white')
    fig.text(.095,.95,f'RENT PREDICTION LAB  /  {number:02}  /  {label.upper()}',fontsize=9.5,weight='bold',color=TEAL)
    fig.text(.094,.905,title,fontsize=29,weight='bold',color=INK,va='top',linespacing=1.08)
    fig.add_artist(Line2D([.095,.905],[.059,.059],transform=fig.transFigure,color=LINE,lw=1))
    fig.text(.095,.038,'HISTORICAL ASKING RENTS  /  UCI APARTMENT CLASSIFIEDS',fontsize=8,color=MUTED)
    fig.text(.905,.038,f'{number} / 6',fontsize=8,color=MUTED,ha='right')
    return fig


def note(fig, y, title, text, width=87):
    fig.text(.096,y,title,fontsize=11,weight='bold',color=TEAL,va='top')
    fig.text(.096,y-.025,textwrap.fill(text,width),fontsize=11,color=INK,va='top',linespacing=1.45)


def add_chart(fig, bounds, name, draw, size=(8,3)):
    draw(fig.add_axes(bounds))
    image, ax = plt.subplots(figsize=size,layout='constrained')
    draw(ax)
    image.savefig(OUT/'figures'/f'{name}.png',dpi=150,facecolor='white')
    plt.close(image)


def horizontal(ax, labels, values, xlabel, colors=None):
    bars = ax.barh(labels, values, color=colors or BLUE, height=.62)
    ax.bar_label(bars,fmt='$%.0f',padding=5,fontsize=9,color=INK)
    ax.set(xlabel=xlabel,xlim=(0,max(values)*1.2))
    ax.invert_yaxis()
    ax.yaxis.grid(False)
    sns.despine(ax=ax)


def render():
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    info = json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    ranks = pd.read_csv(OUT/'cv_ranked.csv')
    slices = pd.read_csv(OUT/'holdout_slices.csv')
    shap = pd.read_csv(OUT/'shap.csv').head(8)
    dev = pd.read_csv(WORK/'development_eda.csv')
    pred = pd.read_csv(WORK/'holdout_predictions.csv')
    cv = pd.DataFrame(info['cv'])
    test = pd.DataFrame(info['holdout']).set_index('method').loc[cv.method].reset_index()
    picked = info['champion']
    cb, rf = (test.set_index('method').loc[name] for name in ('CatBoost','Random forest'))
    chosen = test.set_index('method').loc[picked]
    cat_cv = cv.set_index('method').loc['CatBoost','cv_rmse']
    tuning_delta = info['cat_fixed_cv'] - cat_cv
    ci = info['cat_minus_rf_ci']
    selected = info['cat_params']
    sns.set_theme(style='whitegrid',rc={'grid.color':LINE,'text.color':INK,'axes.labelcolor':INK,
        'xtick.color':MUTED,'ytick.color':MUTED,'axes.edgecolor':LINE,'pdf.fonttype':42,'text.parse_math':False,
        'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9})

    def rent_hist(ax):
        sns.histplot(dev.loc[dev.rent<=6000,'rent'],binwidth=200,binrange=(0,6000),color=BLUE,ax=ax)
        ax.axvline(info['dev_median'],lw=1.4,color=ORANGE,label=f"Median ${info['dev_median']:,.0f}")
        ax.set(xlabel='Monthly asking rent (USD; shown up to $6,000)',ylabel='Development listings')
        ax.xaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
        ax.legend(frameon=False,fontsize=9)
        sns.despine(ax=ax)

    def split_chart(ax):
        splits = pd.DataFrame(info['folds'])
        rows = [f"Fold {r}" for r in splits.fold] + ['Final holdout']
        fit = [*splits.train,info['row_counts']['development']]
        evaluate = [*splits.validation,info['row_counts']['holdout']]
        ax.barh(rows,fit,color=BLUE,height=.54,label='Training rows')
        ax.barh(rows,evaluate,left=fit,color=[TEAL]*len(splits)+[ORANGE],height=.54)
        for i,(a,b) in enumerate(zip(fit,evaluate)):
            ax.text(a/2,i,f'{a:,}',color='white',ha='center',va='center',fontsize=9)
            ax.text(a+b/2,i,f'{b:,}',color='white',ha='center',va='center',fontsize=9)
        ax.set(xlabel='Listing rows (not a time axis)',xlim=(0,sum([fit[-1],evaluate[-1]])*1.03))
        ax.invert_yaxis(); ax.yaxis.grid(False)
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.legend(handles=[Line2D([],[],color=c,lw=6,label=s) for c,s in
                           [(BLUE,'Train'),(TEAL,'Validation'),(ORANGE,'Final test')]],ncol=3,frameon=False,
                  loc='lower left',bbox_to_anchor=(0,1.01),fontsize=9)
        sns.despine(ax=ax)

    def paths(ax):
        curves = ranks[ranks.method.eq('CatBoost')].sort_values('trees')
        top = int(curves.loc[curves.cv_rmse.idxmin(),'trial'])
        for trial,part in curves.groupby('trial'):
            ax.plot(part.trees,part.cv_rmse,marker='o',markersize=3.5,
                lw=2 if trial==top else 1,color=TEAL if trial==top else BLUE,
                alpha=1 if trial==top else .32,label=f'Chosen configuration (#{top})' if trial==top else None)
        ax.scatter(selected['iterations'],cat_cv,color=ORANGE,s=48,zorder=5,label='Chosen tree count')
        ax.set(xticks=sorted(curves.trees.unique()),xlabel='Trees used for prediction',ylabel='Pooled CV RMSE')
        ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
        ax.legend(frameon=False,fontsize=8.8,ncol=2,loc='lower left',bbox_to_anchor=(0,1.02))
        sns.despine(ax=ax)

    def variants(ax):
        vals = [cat_cv,info['cat_fixed_cv'],*[x['cv_rmse'] for x in info['ablations']]]
        labels = ['Tuned CatBoost','Fixed CatBoost',*[x['method'] for x in info['ablations']]]
        horizontal(ax,labels,vals,'Pooled development CV RMSE (USD)',[TEAL,BLUE,BLUE,BLUE])

    def final_compare(ax):
        y = np.arange(len(cv)); width=.31
        p = ax.barh(y-width/2,cv.cv_rmse,height=width,color=BLUE,label='Grouped CV')
        q = ax.barh(y+width/2,test.rmse,height=width,color=TEAL,label='Final holdout')
        for bars in (p,q): ax.bar_label(bars,fmt='$%.0f',padding=3,fontsize=8.2)
        ax.set(yticks=y,yticklabels=cv.method,xlim=(0,max(cv.cv_rmse.max(),test.rmse.max())*1.14),xlabel='RMSE (USD; lower is better)')
        ax.invert_yaxis(); ax.yaxis.grid(False)
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),frameon=False,ncol=2,fontsize=9)
        sns.despine(ax=ax)

    def fit_scatter(ax):
        shown = pred.actual.le(6500) & pred[picked].le(6500)
        ax.hexbin(pred.loc[shown,'actual'],pred.loc[shown,picked],gridsize=27,bins='log',mincnt=3,cmap='Blues',extent=(0,6500,0,6500))
        ax.plot([0,6500],[0,6500],color=ORANGE,lw=1.1,ls='--',label='Perfect prediction')
        ax.set(xlabel='Actual asking rent (USD)',ylabel='Predicted rent (USD)',xlim=(0,6500),ylim=(0,6500))
        ax.xaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
        ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
        ax.legend(frameon=False,fontsize=8.5,loc='upper left')
        sns.despine(ax=ax)

    def by_band(ax):
        band = slices[slices.dimension.eq('price_band')].set_index('segment').reindex(BANDS).dropna()
        labels = [f'{name} (n={row.n:,.0f})' for name,row in band.iterrows()]
        horizontal(ax,labels,band.mae.tolist(),'Holdout MAE (USD), by actual asking rent',[BLUE]*(len(band)-1)+[ORANGE])

    def attribution(ax):
        labels = {'comp_city':'City rent comparable','comp_city_sqft':'City rent/sq ft comparable',
                  'comp_state':'State rent comparable','comp_city_count':'City comp count (log)',
                  'square_feet':'Square feet','log_sqft':'Log square feet','bed_bath':'Bedrooms × bathrooms',
                  'latitude':'Latitude','longitude':'Longitude','market':'City and state'}
        friendly = [labels.get(s,s.replace('_',' ').capitalize()) for s in shap.feature]
        horizontal(ax,friendly,shap.mean_abs_shap.tolist(),'Mean absolute SHAP contribution (USD)',[TEAL]+[BLUE]*(len(shap)-1))

    def save(pdf,fig):
        pdf.savefig(fig); plt.close(fig)

    with PdfPages(OUT/'rent-prediction-case-study.pdf',metadata={
        'Title':'Estimating apartment asking rent','Subject':'Grouped model tuning and geographic evaluation','CreationDate':None}) as pdf:
        fig = page(1,'Case','Estimating apartment\nasking rent')
        note(fig,.785,'Decision','Can a listing analyst estimate monthly asking rent for a listing in an area whose prices were not used for training? The estimate could flag unusual listings for review, not set rents automatically.')
        for x,value,label in [(.1,f"{info['audit']['retained']:,}",'Eligible listings'),(.38,f"{info['row_counts']['holdout']:,}",'Final holdout listings'),(.665,f'${chosen.rmse:,.0f}','Selected model RMSE')]:
            fig.text(x,.642,value,color=TEAL,fontsize=26,weight='bold'); fig.text(x,.614,label,fontsize=9.5,color=MUTED)
        add_chart(fig,[.171,.278,.714,.265],'01-rent-distribution',rent_hist)
        note(fig,.197,'Scope of the data [1]',f"UCI’s 10K classified-listing snapshot, {info['date_start']} to {info['date_end']}. Only monthly USD apartment listings with positive rent and size are included; {info['audit']['excluded']} rows were excluded. The chart hides the upper tail for readability, but evaluation does not trim it.")
        save(pdf,fig)

        fig = page(2,'Evaluation','Keep each grid cell\non one side of the split')
        note(fig,.784,'Split by coordinate cell, not by row [3]','Use a 0.02° latitude/longitude grid. A cell stays on one side of a split. Around 20% of cells go to the final test; the rest enter four grouped CV folds. Listings can still be geographically close across cell borders.')
        add_chart(fig,[.21,.362,.672,.247],'02-grouped-validation',split_chart)
        note(fig,.287,'Rebuild the competition’s strongest idea without its leakage','Use city rent, city rent per square foot, state rent and city evidence counts. A training row’s statistics exclude its entire cell. Validation and test features use training rents only. Sparse city averages shrink toward state and overall averages.')
        note(fig,.151,'Other features','Use size, beds, baths, coordinates, amenity flags, listing metadata and city/state. Drop target-adjacent text and fields, ID, address and timestamp.')
        save(pdf,fig)

        fig = page(3,'Search','Tune the model,\nthen test the features')
        note(fig,.786,'Same folds and dollar RMSE for every candidate [2, 5]','Compare median rent, Ridge, tuned random forest and CatBoost. Score 9 CatBoost settings at 150, 300, 500 and 750 trees; score 6 forest settings. Select the smallest pooled CV RMSE before opening the holdout.')
        add_chart(fig,[.184,.48,.706,.177],'03-tuning-curves',paths)
        add_chart(fig,[.299,.226,.586,.156],'04-feature-and-target-ablation',variants)
        note(fig,.155,'Selected CatBoost settings',f"Depth {selected['depth']}, learning rate {selected['learning_rate']}, L2 penalty {selected['l2_leaf_reg']}, split randomness {selected['random_strength']}, {selected['iterations']} trees. Tuning cut CV RMSE by ${tuning_delta:,.1f}. Removing rent comparables improved CV further at these settings; that variant was not tested on the holdout.")
        save(pdf,fig)

        fig = page(4,'Final holdout','How much error\nremains on new cells?')
        note(fig,.784,'Comparison after configurations were locked',f'The CV pick was {picked}. It scored ${chosen.rmse:,.0f} RMSE and ${chosen.mae:,.0f} MAE on the final test. RMSE penalizes large errors more than MAE; both are in dollars per month.')
        add_chart(fig,[.236,.50,.644,.166],'05-model-comparison',final_compare)
        add_chart(fig,[.184,.216,.707,.171],'06-observed-vs-predicted',fit_scatter)
        note(fig,.153,'How firm is the comparison?',f"The paired 95% interval for CatBoost minus forest RMSE was {ci[0]:+,.0f} to {ci[1]:+,.0f} USD. It covers zero. The plot stops at $6,500 per axis and shows bins with 3+ listings; all rows are included in the metrics.")
        save(pdf,fig)

        fig = page(5,'Diagnostics','Where the model\nmisses most')
        bymarket = slices[slices.dimension.eq('seen_market')].set_index('segment')
        unseen, seen = bymarket.loc['False'], bymarket.loc['True']
        note(fig,.784,'A held-out cell can still be in a familiar city',f"Of the {len(pred):,} test listings, {int(seen.n):,} were in a city/state seen during training (MAE ${seen.mae:,.0f}). The other {int(unseen.n):,} were in unseen markets (MAE ${unseen.mae:,.0f}). These groups can have different mixes of rents.")
        add_chart(fig,[.29,.518,.596,.181],'07-error-by-price-band',by_band)
        add_chart(fig,[.30,.23,.586,.195],'08-catboost-shap',attribution)
        note(fig,.15,'Read SHAP as model behavior, not cause [4]','The bottom chart shows the mean absolute contribution to CatBoost predictions on 300 sampled holdout rows. Correlated inputs can share attribution. The ranking cannot show what would happen to rent if an amenity were added.')
        save(pdf,fig)

        fig = page(6,'Decision','What the study\ncan support')
        note(fig,.785,'A benchmark, not a valuation product','These are historical asking prices, not signed leases or current fair market rents. The snapshot is not a census. Geographic grouping reduces local overlap but cannot guarantee that every reposted property stays on one side.')
        note(fig,.648,'What to validate next','Check a newer time holdout and a whole-market holdout. Audit errors by market and rent band, especially in the luxury tail. For a production listing tool, remove or justify source and geography signals and evaluate fairness before deployment.')
        note(fig,.511,'Reproduce','Run python study.py and python report.py in the pinned environment. The repo includes fold scores, ranked settings, parameters, error slices and global attributions. Raw listings, row-level holdout data and original competition files are not published.')
        sources = [('[1] UCI Apartment for Rent Classified (2019). DOI 10.24432/C5X623. CC BY 4.0.', 'https://archive.ics.uci.edu/dataset/555/apartment%2Bfor%2Brent%2Bclassified'),
            ('[2] Prokhorenkova et al. (2018). CatBoost: unbiased boosting with categorical features.', 'https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html'),
            ('[3] scikit-learn 1.6 documentation: GroupKFold.', 'https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GroupKFold.html'),
            ('[4] CatBoost documentation: ShapValues.', 'https://catboost.ai/docs/en/concepts/shap-values'),
            ('[5] CatBoost documentation: predict and tree-range parameters.', 'https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_predict')]
        fig.text(.096,.365,'SOURCES',fontsize=10,color=TEAL,weight='bold')
        for i,(text,url) in enumerate(sources):
            fig.text(.096,.335-i*.032,text,fontsize=9,color=INK,url=url)
        save(pdf,fig)


if __name__ == '__main__':
    render()
