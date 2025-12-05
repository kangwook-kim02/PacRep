from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt




# task0
cnf_matrix_task0 = np.array([[178,   0],
                            [  0, 422]])
labels_task0 = ['vpn', 'nonvpn']


sns.heatmap(cnf_matrix_task0, annot=True, cmap="Blues", fmt='d', xticklabels=labels_task0, yticklabels=labels_task0) # 소수점 2자리수 까지는 fmt = 'd'
plt.xlabel('Predict')
plt.ylabel('True')

plt.savefig('confusion_matrix_task0.png', dpi=300, bbox_inches='tight')
plt.close() 


# task1

cnf_matrix_task1 = np.array([
    [8, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 29, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0],
    [0, 0, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 49, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 93, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7]
]
)


labels_task1 = [  'aimchat', 'email', 'facebook', 'ftps', 
                  'hangouts', 'icq', 'netflix', 'scp','sftp','skype', 
                  'spotify', 'vimeo', 'voipbuster', 'youtube', 'bittorrent']


sns.heatmap(cnf_matrix_task1, annot=True, cmap="Blues", fmt='d', xticklabels=labels_task1, yticklabels=labels_task1)
plt.xlabel('Predict')
plt.ylabel('True')

plt.savefig('confusion_matrix_task1.png', dpi=300, bbox_inches='tight')
plt.close() 

# task 2

cnf_matrix_task2 = np.array([[62,  0],
                            [ 0, 63]])


labels_task2= ['malicious', 'benign']


sns.heatmap(cnf_matrix_task2, annot=True, cmap="Blues", fmt='d', xticklabels=labels_task2, yticklabels=labels_task2)
plt.xlabel('Predict')
plt.ylabel('True')

plt.savefig('confusion_matrix_task2.png', dpi=300, bbox_inches='tight')
plt.close() 

# task 3

cnf_matrix_task3 = np.array(
           [[33,  0,  0,  0,  0],
            [ 0,  7,  0,  0,  0],
            [ 0,  0, 22,  0,  0],
            [ 0,  0,  0, 32,  3],
            [ 0,  0,  0,  1, 27]]
)

# 정답률 계산 코드

labels_task3 = ['dns2tcp', 'dnscat2', 'iodine', 'chrome', 'firefox']

plt.figure(figsize=(12, 10))
sns.heatmap(cnf_matrix_task3, annot=True, cmap="Blues", fmt='d', xticklabels=labels_task3, yticklabels=labels_task3)
plt.xlabel('Predict')
plt.ylabel('True')

plt.savefig('confusion_matrix_task3.png', dpi=300, bbox_inches='tight')
plt.close() 

# task 4
cnf_matrix_task4 = np.array(
[[255,   0],
 [  0, 220]]
)

# 정답률 계산 코드

labels_task4 = ['malware', 'normal']

plt.figure(figsize=(12, 10))
sns.heatmap(cnf_matrix_task4, annot=True, cmap="Blues", fmt='d', xticklabels=labels_task4, yticklabels=labels_task4)
plt.xlabel('Predict')
plt.ylabel('True')

plt.savefig('confusion_matrix_task4.png', dpi=300, bbox_inches='tight')
plt.close() 

# task 5
cnf_matrix_task5  = np.array(
[
    [21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 22, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 22, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 4, 0, 0, 0, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32]
]
)

# 정답률 계산 코드

labels_task5 = ['cridex', 'geodo', 'htbot', 'miuref', 'neris', 
                'nsis', 'shifu', 'tinba','virut','zeus',
                'bittorrent', 'facetime', 'ftp', 'gmail','mysql',
                'outlook', 'skype','smb','weibo','worldofwarcraft']

plt.figure(figsize=(12, 10))
sns.heatmap(cnf_matrix_task5, annot=True, cmap="Blues", fmt='d', xticklabels=labels_task5, yticklabels=labels_task5)
plt.xlabel('Predict')
plt.ylabel('True')

plt.savefig('confusion_matrix_task5.png', dpi=300, bbox_inches='tight')
plt.close() 