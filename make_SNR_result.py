import os
import sys
import subprocess
import pandas as pd
import pdb
def search(d_name, li, EXT):
    for (paths, dirs, files) in os.walk(d_name):
        for filename in files:
            ext = filename.split('_')[0]
            if ext == EXT:
                li.append(os.path.join(os.path.join(os.path.abspath(d_name),paths),filename))

READ_ROOT = sys.argv[1]
pdb.set_trace()
res_list2 = []
search(READ_ROOT, res_list2, sys.argv[2])
res_list2 = sorted(res_list2)
pdb.set_trace()
df = pd.DataFrame(columns=['clean','20dB','15dB','10dB','5dB','0dB','m5dB'])
check_list = ['clean','20dB','15dB','10dB','5dB','0dB','m5dB']
res_list = []
for i in range(len(res_list2)):
    if res_list2[i].split('/')[-2][0:3]=='new':
        res_list.append(res_list2[i])
res_list = sorted(res_list)
pdb.set_trace()
# df = pd.DataFrame(columns=['20dB','15dB','10dB','5dB','0dB','m5dB'])
# check_list = ['20dB','15dB','10dB','5dB','0dB','m5dB']
#check_list = ['clean']
for res in res_list:
    ep  = res.split('/')[-3].zfill(3)
    snr = (res.split('/')[-2]).split('_')[-1]
    if not snr in check_list:
        continue
    com = "grep -r 'Mean' " + res + " | awk '{print $11}' | head -1" 
    er  = subprocess.check_output(com, shell=True, universal_newlines=True).rstrip('\n')
    df.loc[ep,snr] = er
#pdb.set_trace()
df = df.fillna(100)
#df = df.astype(float)
#df['Avg.'] = df.mean(axis=1)
#df.sort_index(inplace=True)
#pd.options.display.float_format = '{:.2f}'.format
#print('\n'+READ_ROOT.split('/')[-2]+'\n')
#print(df)

try:
    if not os.path.exists("./scv_file"):
        os.mkdir("./scv_file")
    name="./scv_file/" + str(sys.argv[3]) + ".csv"
    df.to_csv(name, mode='w')
except IndexError as e:
    pass

