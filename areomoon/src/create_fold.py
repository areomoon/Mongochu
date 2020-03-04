import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

label_path = '../AIMango_sample/label.csv'
save_path  = '../AIMango_sample/train_folds.csv'

if __name__ == '__main__':
    df = pd.read_csv(label_path,encoding = 'iso-8859-1',header=None)
    df.columns = ['image_id','labels']
    df.loc[:,'kfold']=-1
    df = df.sample(frac=1).reset_index(drop=True)

    x = df['image_id'].values
    y = pd.get_dummies(df['labels'].apply(lambda x: x[-1])).values

    mskf = MultilabelStratifiedKFold(n_splits=5,random_state=0)

    for fold ,(trn_, val_) in enumerate(mskf.split(x,y)):
        print('TRAIN ',trn_,'VALID' ,val_)
        df.loc[val_,'kfold'] = fold

    print(df['kfold'].value_counts())
    df.to_csv(save_path, index = False)

