import pandas as pd
import numpy as np
import re
from collections import OrderedDict


algotypes = ['hamilton', 'reno', 'bbr', 'cubic']


def get_ttypes(atype, problem='binary'):
    fname = "../data/indis_2019/%s/destination/logs/start_end.log" % atype
    with open(fname) as ffile:
        content = ffile.readlines()
    if problem == 'binary':
        dic = {'normal': [0, 0]}
        if "normal" in content[0] and "normal" in content[1]:
            start = re.findall(r'\d+', content[0])
            dic['normal'][0] = list(map(int, start))[0]
            end = re.findall(r'\d+', content[1])
            dic['normal'][1] = list(map(int, end))[0]
        else:
            raise Exception('log should start with normal type')
    else:
        ttypes = ['normal', 'loss', 'duplicate', 'reorder']
        dic = OrderedDict([('normal',[0,0]), ('loss',[0,0]), ('duplicate',[0,0]), ('reorder',[0,0])])
        index = 0
        for tt in ttypes:
            while tt in content[index] and tt in content[index+1]:
                if dic[tt][0] == 0:
                    start = re.findall(r'\d+', content[index])
                    dic[tt][0] = list(map(int, start))[0]
                    end = re.findall(r'\d+', content[index+1])
                    dic[tt][1] = list(map(int, end))[0]
                else:
                    end = re.findall(r'\d+', content[index+1])
                    dic[tt][1] = list(map(int, end))[0]
                index += 2
                if index > len(content) - 1:
                    break
    return dic


def read_data(problem='binary'):
    data = {}
    unused_columns = ['c_ip:1', 's_ip:15', 'first:29', 'last:30', 'http_res:113',
                      'c_tls_SNI:116', 's_tls_SCN:117', 'fqdn:127', 'dns_rslv:128',
                      'c_rtt_std:48', 's_rtt_std:55', 's_rtt_avg:52', 'c_cwin_max:76', 'c_rtt_avg:45',
                      'c_bytes_retx:11', 's_win_max:96', 's_bytes_retx:25', 's_pkts_unk:106', 'c_pkts_unk:83',
                      'c_pkts_retx:10', 's_pkts_retx:24', 's_pkts_fs:103', 'c_pkts_fs:80', 's_first_ack:37',
                      'c_first_ack:36', 'c_mss_max:71', 's_mss_max:94', 's_pkts_ooo:26', 'c_pkts_ooo:12',
                      's_first:33', 'c_first:32', 'c_rtt_max:47', 's_rtt_max:54', 'c_pkts_push:114', 's_pkts_push:115',
                      's_pkts_reor:104', 'c_pkts_reor:81', 's_sack_cnt:92', 'c_sack_cnt:69', 'c_rtt_cnt:49', 's_rtt_cnt:56',
                      'c_ack_cnt_p:6', 's_ack_cnt_p:20', 'c_pkts_rto:79', 's_pkts_rto:102', 'c_rtt_min:46', 's_rtt_min:53',
                      'c_last:34', 's_last:35', 'c_rtt_min:46', 's_rtt_min:53', 'durat:31', 'c_pkts_data:8', 's_pkts_data:22',
                      'c_ack_cnt:5', 's_ack_cnt:19', 'c_win_min:74', 's_win_min:97', 'c_pkts_all:3', 's_pkts_all:17',
                      'c_port:2', 's_port:16', 'c_bytes_all:9', 's_bytes_all:23',
                      'c_bytes_uniq:7', 's_bytes_uniq:21']
    unused_columns = sorted(unused_columns, key=lambda x: list(map(int, re.findall(r'\d+', x)))[0])
    print(len(unused_columns), unused_columns)
    #unused_columns = ['c_ip:1', 's_ip:15', 'first:29', 'last:30', 'http_res:113',
    #                  'c_tls_SNI:116', 's_tls_SCN:117', 'fqdn:127', 'dns_rslv:128']
    for atype in algotypes:
        dic = get_ttypes(atype, problem=problem)
        rawfname = "../data/indis_2019/%s/source/processed/all.log" % atype
        fname = "../data/%s_all.csv" % atype
        with open(rawfname) as rfile:
            content = rfile.readlines()
        if content[0].startswith('#15#'):
            content[0] = content[0][4:]
            with open(fname, "w", newline="") as ffile:
                for row in content:
                    r = row.split()
                    r_content = ','.join(map(str, r))
                    ffile.write(r_content)
                    ffile.write('\n')
        else:
            raise Exception('Data has the wrong starting???')
        df = pd.read_csv(fname, index_col=None)
        labels = np.array([1] * df.shape[0])
        if problem == 'binary':
            normal_indices = df.index[df['first:29'] < dic['normal'][1] * 1000].tolist()
            labels[normal_indices] = 0
        else:
            index = 0
            for tt in dic.keys():
                normal_indices = df.index[(df['first:29'] < dic[tt][1] * 1000) & (df['first:29'] >= dic[tt][0] * 1000)].tolist()
                labels[normal_indices] = index
                index += 1

        df['Label'] = labels
        df.fillna(0, inplace=True)
        df.drop(unused_columns, axis=1, inplace=True)
        data[atype] = df
    return data


if __name__ == '__main__':
    read_data('multiclass')