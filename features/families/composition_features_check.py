import pandas as pd, numpy as np, re, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
d=pd.read_csv(__import__('pathlib').Path(__file__).resolve().parents[2] / 'data' / 'complete_data.csv')
d['TicketCount']=d.groupby('Ticket').PassengerId.transform('size'); d['LogFare']=np.log1p((d.Fare/d.TicketCount).fillna(d.Fare.median()))
t=d.Name.str.extract(r',\s*([^.]+)\.',expand=False).str.strip(); d['Title']=t.where(t.isin(['Mr','Mrs','Miss','Master']),'Other')
d['AgeMissing']=d.Age.isna().astype(int); d['AgeF']=d.Age.fillna(d.groupby('Title').Age.transform('median'))
d['DeckF']=d.Cabin.str[0].fillna('None'); d['EmbarkedF']=d.Embarked.fillna('S')
def rp(x):
    p=x.split(' ')
    if x.replace(' ','').isdigit(): return 'NUM'
    return re.sub(r'[./ ]','',' '.join(p[:-1]) if p[-1].isdigit() else x).upper()
d['TT']=d.Ticket.map(rp); vc=d.TT.value_counts(); d.loc[d.TT.isin(vc[vc<10].index),'TT']='OTHER'
d['Group']=np.where(d.FamilyId.notna(),'F'+d.FamilyId.astype('Int64').astype(str),'T'+d.Ticket)
sex=dict(zip(d.PassengerId,d.Sex))
def comp(rs, me):
    out=dict(brother=0,sister=0,spouse=0,sons=0,daughters=0,mother=0,father=0)
    if isinstance(rs,str):
        for part in rs.split('; '):
            role,xs=part.split(' ',1)
            for o in map(int,xs.split(', ')):
                m=sex[o]=='male'
                if role=='sibling': out['brother' if m else 'sister']+=1
                elif role=='spouse': out['spouse']=1
                elif role=='child': out['sons' if m else 'daughters']+=1
                elif role=='parent': out['father' if m else 'mother']+=1
    return out
C=pd.DataFrame([comp(r,p) for r,p in zip(d.Relatives,d.PassengerId)])
d=pd.concat([d,C],axis=1)
d['children']=d.sons+d.daughters
d['HasBrother']=(d.brother>0).astype(int); d['HasSister']=(d.sister>0).astype(int)
d['HasChildren']=(d.children>0).astype(int); d['Children3plus']=(d.children>2).astype(int)
d['HasSpouseAndChildren']=((d.spouse==1)&(d.children>0)).astype(int)
d['HasMother']=(d.mother>0).astype(int); d['HasFather']=(d.father>0).astype(int)
d['MotherNoFather']=((d.mother>0)&(d.father==0)).astype(int)
d['WomanWithHusbandAndChildren']=((d.Sex=='female')&(d.spouse==1)&(d.children>0)).astype(int)
d['ManWithWifeAndChildren']=((d.Sex=='male')&(d.spouse==1)&(d.children>0)).astype(int)
tr=d[d.Dataset=='train'].reset_index(drop=True); y=tr.Survived.astype(int).values
CAT=['Sex','Pclass','Title','DeckF','EmbarkedF','TT']; NUM=['AgeF','AgeMissing','LogFare','SibSp','Parch']
def make(kind,extra):
    if kind=='lr':
        return make_pipeline(ColumnTransformer([('c',OneHotEncoder(handle_unknown='ignore'),CAT),('n',StandardScaler(),NUM+extra)]),LogisticRegression(max_iter=3000))
    return make_pipeline(ColumnTransformer([('c',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),CAT),('n','passthrough',NUM+extra)]),
        HistGradientBoostingClassifier(max_depth=3,learning_rate=0.05,max_iter=200,categorical_features=list(range(len(CAT))),random_state=0))
groups=tr.Group.values; ug=np.unique(groups)
def folds(seed):
    rng=np.random.default_rng(seed); perm=rng.permutation(len(ug)); f=dict(zip(ug[perm],np.arange(len(ug))%5))
    fi=np.array([f[g] for g in groups]); return [(np.where(fi!=k)[0],np.where(fi==k)[0]) for k in range(5)]
FOLDS=[folds(s) for s in range(10)]
def score(kind,extra):
    ll=[];auc=[];acc=[]
    for F in FOLDS:
        p=np.zeros(len(tr))
        for a,b in F: p[b]=make(kind,extra).fit(tr.iloc[a],y[a]).predict_proba(tr.iloc[b])[:,1]
        ll.append(log_loss(y,p)); auc.append(roc_auc_score(y,p)); acc.append(np.mean((p>.5)==y))
    return np.array(ll),np.array(auc),np.array(acc)
SETS={'HasBrother':['HasBrother'],'HasSister':['HasSister'],'HasChildren':['HasChildren'],'Children3plus':['Children3plus'],
 'HasSpouseAndChildren':['HasSpouseAndChildren'],'HasMother':['HasMother'],'MotherNoFather':['MotherNoFather'],
 'WomanWithHusbandAndChildren':['WomanWithHusbandAndChildren'],'ManWithWifeAndChildren':['ManWithWifeAndChildren'],
 'All composition features':['HasBrother','HasSister','HasChildren','Children3plus','HasSpouseAndChildren','HasMother','HasFather','MotherNoFather','WomanWithHusbandAndChildren','ManWithWifeAndChildren']}
for kind in ['lr','gb']:
    b_ll,b_auc,b_acc=score(kind,[])
    print(kind,'baseline logloss %.4f auc %.4f acc %.3f'%(b_ll.mean(),b_auc.mean(),b_acc.mean()))
    for name,ex in SETS.items():
        ll,auc,acc=score(kind,ex); dl=ll-b_ll
        print(f"  {name:30s} n={int(tr[ex].max(axis=1).sum()):4d}  dLogloss {dl.mean():+.4f} (better in {np.sum(dl<0)}/10)  dAUC {(auc-b_auc).mean():+.4f}  dAcc {(acc-b_acc).mean():+.3f}")
