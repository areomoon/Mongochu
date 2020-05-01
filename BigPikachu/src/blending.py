import os
import glob
import pandas as pd

parser = argparse.ArgumentParser(description='blending algo for mango classification')

parser.add_argument('--folder_path', default = "./pred", type=str,
                    help='path to predicted prob data')

parser.add_argument('--voting', default = 'soft', type=str,
                    help='logics of voting')

parser.add_argument('--output_name', default='blend_submission', type=str,
                    help='file name of  blended prediction')

args = parser.parse_args()

def main():
    file_list = os.path.join(args.folder_path, '*.csv')
    file_path = glob.glob(file_list)
    df_all = pd.concat([pd.read_csv(f) for f in file_path]) 

    if args.voting == 'soft':
        soft_label = df_all.groupby('image_id').mean().idxmax(axis=1)
        sub = pd.DataFrame(soft_label, columns = ['label']).reset_index(drop=False)
    
    elif args.voting == 'hard':
        df_all = df_all.set_index('image_ids')
        df_all['label'] = df_all[['A', 'B', 'C']].apply(lambda x: np.argmax(x, axis=0), axis=1)
        hard_label = df_all.groupby('image_ids')['label'].apply(lambda x: np.argmax(np.bincount(x)))

        class_map = {0:'A',1:'B',2:'C'}
        sub = hard_label.map(class_map).reset_index(drop=False)
        
    sub.to_csv(f'{args.output_name}.csv',index=False)

if __name__ == "__main__":
    main()