import scipy.special

flag = 0
input_={'1':1,'2':0,'3':1} #输入
learn = 0.9 #学习系数
T = 1   #理想输出
list = 0.001	#最小精确度
w = weight={'14':0.2,'15':-0.3,'24':0.4,'25':0.1,'34':-0.5,'35':0.2,'46':-0.3,'56':-0.2} #权重
bias={'4':-0.4,'5':0.2,'6':0.1} #偏置
S={'4':0,'5':0,'6':0} #456总输入
O={'4':0,'5':0,'6':0} #456输出
E={'4':0,'5':0,'6':0} #456误差
#总输入和输出
def output():
    S['4'] = input_['1']*w['14']+input_['2']*w['24']+input_['3']*w['34']+bias['4']
    O['4'] = scipy.special.expit(S['4'])
    S['5'] = input_['1']*w['15']+input_['2']*w['25']+input_['3']*w['35']+bias['5']
    O['5'] = scipy.special.expit(S['5'])
    S['6'] = O['4']*w['46']+O['5']*w['56']+bias['6']
    O['6'] = scipy.special.expit(S['6'])

#误差
def partial():
    E['6'] = O['6']*(1-O['6'])*(1-O['6'])
    E['5'] = O['5']*(1-O['5'])*E['6']*w['56']
    E['4'] = O['4']*(1-O['4'])*E['5']*w['46']

#调整
def train():
    w['46'] = w['46'] + learn*E['6']*E['4']
    w['56'] = w['56'] + learn * E['6'] * E['5']
    w['14'] = w['14'] + learn * E['4'] * input_['1']
    w['15'] = w['15'] + learn * E['5'] * input_['1']
    w['24'] = w['24'] + learn * E['4'] * input_['2']
    w['25'] = w['25'] + learn * E['5'] * input_['2']
    w['34'] = w['34'] + learn * E['4'] * input_['3']
    w['35'] = w['35'] + learn * E['5'] * input_['3']
    bias['6'] = bias['6'] + learn * E['6']
    bias['5'] = bias['5'] + learn * E['5']
    bias['4'] = bias['4'] + learn * E['4']

while True:
    output()
    partial()
    train()
    print(O['6'])
    flag = flag + 1
    if O['6'] >= T - list:
        break

print('times has been gone:',flag)
print('weight now:')
for key,value in w.items():
	print('w',key,':',value) 
print('the last output:',O['6'])





