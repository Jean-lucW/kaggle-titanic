import pandas as pd, numpy as np, re, itertools
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

c = pd.read_csv(__import__('pathlib').Path(__file__).resolve().parents[2] / 'data' / 'complete_data.csv')
c['TicketCount'] = c.groupby('Ticket').PassengerId.transform('size')
c['FarePP'] = (c.Fare / c.TicketCount).fillna(c.Fare.median())
c['LogFare'] = np.log1p(c.FarePP)
t = c.Name.str.extract(r',\s*([^.]+)\.', expand=False).str.strip()
c['Title'] = t.where(t.isin(['Mr','Mrs','Miss','Master']), 'Other')
c['AgeMissing'] = c.Age.isna().astype(int)
c['AgeF'] = c.Age.fillna(c.groupby('Title').Age.transform('median'))
c['Deck'] = c.Cabin.str[0].fillna('None')
c['Embarked'] = c.Embarked.fillna('S')
def rp(x):
    p = x.split(' ')
    if x.replace(' ','').isdigit(): return 'NUM'
    return re.sub(r'[./ ]','',' '.join(p[:-1]) if p[-1].isdigit() else x).upper()
c['TT'] = c.Ticket.map(rp); vc = c.TT.value_counts(); c.loc[c.TT.isin(vc[vc<10].index),'TT']='OTHER'
c['Group'] = np.where(c.FamilyId.notna(), 'F'+c.FamilyId.astype('Int64').astype(str), 'T'+c.Ticket)
tr = c[c.Dataset=='train'].reset_index(drop=True)
y = tr.Survived.values
cat = ['Sex','Pclass','Title','Deck','Embarked','TT']; num = ['AgeF','AgeMissing','LogFare','SibSp','Parch']
def model():
    return make_pipeline(ColumnTransformer([('c',OneHotEncoder(handle_unknown='ignore'),cat),('n',StandardScaler(),num)]), LogisticRegression(C=1.0, max_iter=2000))
oof = np.zeros(len(tr))
for a,b in GroupKFold(5).split(tr, y, tr.Group):
    m = model().fit(tr.iloc[a], y[a]); oof[b] = m.predict_proba(tr.iloc[b])[:,1]
print('baseline features: logloss %.4f auc %.4f' % (log_loss(y,oof), roc_auc_score(y,oof)))
tr['p']=oof; tr['res']=y-oof
# pair types
idx = range(len(tr)); rows=[]
fam = tr.FamilyId.values; tic = tr.Ticket.values; cls=tr.Pclass.values; sex=tr.Sex.values
deck=tr.Deck.values; fw=pd.cut(tr.FarePP,[-1,7.5,10,20,30,50,1e9]).astype(str).values
for i,j in itertools.combinations(idx,2):
    samefam = not np.isnan(fam[i]) and fam[i]==fam[j]
    if samefam: k='Same family'
    elif tic[i]==tic[j]: k='Same ticket, not family'
    elif deck[i]==deck[j] and deck[i]!='None': k='Same deck, different ticket'
    elif cls[i]==cls[j] and fw[i]==fw[j] and sex[i]==sex[j]: k='Same class, fare window and sex'
    else: continue
    rows.append((k,i,j))
P = pd.DataFrame(rows, columns=['k','i','j'])
def corr(a,b): return np.corrcoef(np.r_[a,b], np.r_[b,a])[0,1]
for k,g in P.groupby('k'):
    print(f"{k:35s} pairs {len(g):6d}  raw r {corr(y[g.i],y[g.j]):+.3f}  residual r {corr(tr.res.values[g.i],tr.res.values[g.j]):+.3f}  same fate {np.mean(y[g.i]==y[g.j]):.2f}")
# predictive: add mates' mean residual (LOO), group folds; mates' residuals come from oof
def mates_feature(df, key):
    s = df.groupby(key).res.transform('sum'); n = df.groupby(key).res.transform('size')
    f = (s - df.res)/(n-1); return f.where(n>1, 0.0), (n>1).astype(int)
for key,label in [('FamilyId','family'),('Ticket','ticket')]:
    d = tr.copy(); 
    if key=='FamilyId': d['FamilyId']=d.FamilyId.fillna(-d.PassengerId)
    d['MateRes'], d['HasMate'] = mates_feature(d, key)
    oof2=np.zeros(len(d))
    for a,b in GroupKFold(5).split(d,y,d.Group):
        m = make_pipeline(ColumnTransformer([('c',OneHotEncoder(handle_unknown='ignore'),cat),('n',StandardScaler(),num+['MateRes','HasMate'])]), LogisticRegression(max_iter=2000)).fit(d.iloc[a],y[a])
        oof2[b]=m.predict_proba(d.iloc[b])[:,1]
    hm = d.HasMate.values==1
    print(f"+{label} mates residual: logloss {log_loss(y,oof2):.4f} auc {roc_auc_score(y,oof2):.4f} | on {hm.sum()} with mates: base {log_loss(y[hm],oof[hm]):.4f} -> {log_loss(y[hm],oof2[hm]):.4f}, acc {np.mean((oof[hm]>.5)==y[hm]):.3f} -> {np.mean((oof2[hm]>.5)==y[hm]):.3f}")

rng=np.random.default_rng(0)
r=tr.res.values
for k in ['Same family','Same ticket, not family','Same class, fare window and sex']:
    g=P[P.k==k].copy(); g['grp']=tr.Group.values[g.i]
    groups=g.grp.unique(); bs=[]
    by={k2:v for k2,v in g.groupby('grp')}
    for _ in range(300):
        s=pd.concat([by[x] for x in rng.choice(groups,len(groups))])
        bs.append(corr(r[s.i],r[s.j]))
    print(k, 'residual r 95%% range %.2f to %.2f' % tuple(np.percentile(bs,[2.5,97.5])))
g=P[P.k=='Same family']
ss = sex[g.i]==sex[g.j]
print('family same-sex pairs',ss.sum(),'resid r %.3f'%corr(r[g.i[ss]],r[g.j[ss]]),' mixed-sex',(~ss).sum(),'resid r %.3f'%corr(r[g.i[~ss]],r[g.j[~ss]]))
for cl in [1,2,3]:
    m=cls[g.i]==cl; print('class',cl,'pairs',m.sum(),'resid r %.3f'%corr(r[g.i[m]],r[g.j[m]]))
