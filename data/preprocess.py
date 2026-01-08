from scapy.all import *
from tqdm import tqdm
import json
import os
import random
from sklearn.model_selection import train_test_split

# 데이터셋 구성
DATASET_CONFIGS = {
  'ISXW2016_flow': {
    # task 0 ISXW2016 VPN label - 2 calss
    '0': {
      'vpn': 0, 'nonvpn': 1
    },
    # task 1 ISXW2016 application label - 16 class
    '1': {
      'aimchat':0 , 'email':1, 'facebook':2, 'ftps':3, 'gmail':4, 
      'hangouts':5, 'icq':6, 'netflix':7, 'scp':8,'sftp':9,'skype':10, 
      'spotify':11, 'vimeo':12, 'voipbuster':13, 'youtube':14, 'bittorrent':15
    }
  },
  'DoHBrw2020_flow': {
    # task 2 DoHBrw2020 query abnormality label - 2 class
    '2': {
      'malicious':0, 'benign':1 
    },
    # task 3 DoHBrw2020 the query generator label - 5 class
    '3': {
      'dns2tcp':0, 'dnscat2':1, 'iodine':2, 'chrome':3, 'firefox':4 
    }
  },
  'USTCTFC2016_flow': {
    # task 4 USTCTFC2016 software abnormality label - 2 class
    '4': {
        'malware':0, 'normal':1
    },
    # task 5 USFCTFC2016 software label - 20 class
    '5': {
        'cridex':0, 'geodo':1, 'htbot':2, 'miuref':3, 'neris':4, 
        'nsis':5, 'shifu':6, 'tinba':7,'virut':8,'zeus':9,'bittorrent':10, 'facetime':11,            
        'ftp':12, 'gmail':13,'mysql':14,'outlook':15, 'skype':16,'smb':17,'weibo':18,'worldofwarcraft':19
    }
  }
}

# pcap 파일에서 Raw 패킷 추출하여 텍스트 파일로 저장
def get_str():
  base_path = '/data2/kangwook/PacRep_DATASETS/'
  base_outdir = '/data2/kangwook/PacRep_PREPROCESS_0930_3/'
  os.makedirs(base_outdir, exist_ok=True)
  
  datasets = ['ISXW2016_flow', 'DoHBrw2020_flow', 'USTCTFC2016_flow']

  num_folders = len(datasets) 
  for i, dataset in enumerate(datasets): 
    print(f"'\n'Processing folder {i+1}/{num_folders}: {dataset}")

    dataset_outdir = os.path.join(base_outdir, dataset)
    os.makedirs(dataset_outdir, exist_ok=True) 
    
    dataset_path = os.path.join(base_path, dataset)
    pcap_folders = os.listdir(dataset_path)
    for pcap_folder in pcap_folders:
      final_path = os.path.join(dataset_path, pcap_folder)
      final_outdir = os.path.join(dataset_outdir, pcap_folder)
      os.makedirs(final_outdir, exist_ok=True) 
      file_names = [f for f in os.listdir(final_path) if f.endswith('.pcap')]
      print("per-flow split: " , pcap_folder)
      for idx, each in enumerate(tqdm(file_names)):  
          pcap_file = os.path.join(final_path, each)
          txtFileName = each
          
          if os.path.isfile(os.path.join(final_outdir, txtFileName + '.txt')): 
            print(f'{idx+1}. already exists: ', txtFileName + '.txt')
            continue
          
          with open(os.path.join(final_outdir, txtFileName + '.txt'), 'w') as f:
              try:
                  pr = PcapReader(pcap_file)
              except Exception as e:
                  print(f'Error reading file {pcap_file}: {e}')
                  continue
              
              pkt = 1
              while pkt: 
                  try:
                      pkt = pr.read_packet()
                      # if 'Raw' in pkt:
                      
                      if pkt.haslayer(TCP):
                        try:
                            del pkt[TCP].payload
                        except Exception:
                            pass
                      if pkt.haslayer(UDP):
                        try:
                            del pkt[UDP].payload
                        except Exception:
                            pass
                          
                      while pkt.haslayer(Padding):
                        try:
                            del pkt[Padding]
                        except Exception:
                            break
                      while pkt.haslayer(Raw):
                        try:
                            del pkt[Raw]          
                        except Exception:
                            break
                          
                      try:
                        f.write(str(repr(pkt)) + '\n')
                      except OverflowError:
                        continue
                  except EOFError:
                      break
        
      
# 각 텍스트 파일에서 토큰화된 데이터를 추출하여 트리플릿 형태로 변환
def triplet_tokens():
  datasets = ['ISXW2016_flow', 'DoHBrw2020_flow', 'USTCTFC2016_flow']
  adrs = [os.path.join('/data2/kangwook/PacRep_PREPROCESS_0930_3/', dataset_name) for dataset_name in datasets] 
  
  triplets = [] # triplet 데이터를 저장할 리스트
  triplets_ISXW2016_train = []
  triplets_ISXW2016_valid = []
  triplets_ISXW2016_test = []
  
  triplets_DoHBrw2020_train = []
  triplets_DoHBrw2020_valid = []
  triplets_DoHBrw2020_test = []
  
  triplets_USTCTFC2016_train = []
  triplets_USTCTFC2016_valid = []
  triplets_USTCTFC2016_test = []
  
  train = []
  test = []
  valid = []
  
  train_data_by_label = defaultdict(list)
  test_data_by_label = defaultdict(list)
  valid_data_by_label = defaultdict(list)

  for i in range(len(adrs)): # len(adrs) = 3
    train_sum_of_flow = 0
    test_sum_of_flow = 0
    valid_sum_of_flow = 0
    # task1 과 task 5 사이에 중복 labeling을 피하기 위해 label_map 매번 초기화
    label_map = {} 
    for dataset, tasks in DATASET_CONFIGS.items():
      if dataset == adrs[i].split("/")[-1]:
        for task, classes in tasks.items():
          for class_name, val in classes.items():
            label_map[class_name] = (task,val) # ex) 'port_scanning' : ('1', 0)

    print(label_map)        
    print(f"Processing dataset {i+1}/{len(adrs)}: {adrs[i]}")
    
    text_folders = os.listdir(adrs[i])
    for text_folder in text_folders:
      address = os.path.join(adrs[i], text_folder)
    
      text_files = [f for f in os.listdir(address) if f.endswith('.txt')] # txt로 끝나는 파일이 있으면 text_files 리스트에 저장
      print("현재 폴더: ", text_folder, " txt 파일의 개수: ", len(text_files))
      
      if (len(text_files) < 500):
        train_files, temp_data = train_test_split(text_files, test_size=0.2, random_state=42)
        valid_files, test_files = train_test_split(temp_data, test_size=0.5, random_state=42)
      else:
        train_files = random.sample(text_files,400) # 랜덤으로 400 flow 추출
        remain = [x for x in text_files if x not in train_files] # train_files에 있는 원소 제거
        
        test_files = random.sample(remain,50) # 랜덤으로 50 flow 추출
        remain = [x for x in remain if x not in test_files] # test_fils에 있는 원소 제거
        
        valid_files = random.sample(remain,50) # 랜덤으로 50 flow 추출
      
      train_sum_of_flow += len(train_files)
      test_sum_of_flow += len(test_files)
      valid_sum_of_flow += len(valid_files)
      
      print("전체 train.txt 파일의 수: ", len(train_files), " 전체 test.txt 파일의 수: ", len(test_files), "전체 valid.txt 파일의 수: ", len(valid_files))
      lists = {"train_files": train_files, "test_files":test_files, "valid_files":valid_files}
      print("Processing folder: ", text_folder)
      for name, lst in lists.items():
        for idx, fname in enumerate(tqdm(lst)): 
          print(f'{idx+1}/{len(text_files)} Processing file: {fname}')
          print("Processing folder: ", text_folder) 
          label_name = text_folder.lower()
          parts = label_name.split('_') 
          labels = {'0': None, '1': None, '2': None, '3': None, '4': None, '5':None}
          for idx, part in enumerate(parts):
            if part in label_map:
              task, val = label_map[part]
              print(f"파일: {fname}, 태스크: {task}, 레이블: {val}")
              labels[task] = val # ex) {'0': 0, '1': 0, '2': None, '3': None}
              print(labels)
            else:
              print(f"파일: {fname}, label_map에 없음: {part}")

          label_tuple = tuple(labels.items())
          with open(os.path.join(address, fname), 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            lines = lines[:10] # 앞에서부터 최대 10개 패킷 추출 
            
        
            for line in tqdm(lines): 
              text = re.split(r'\\| ', line)
              for k in range(len(text)): 
                if 'src' in text[k] or 'dst' in text[k] or 'port' in text[k]:
                  text[k]=''
              tokens = list(filter(None, text))
              if not tokens:
                continue
              if name == "train_files":
                train_data_by_label[label_tuple].append(tokens) # label 기준으로 토큰들 그룹화
              elif name == "test_files":
                test_data_by_label[label_tuple].append(tokens) # label 기준으로 토큰들 그룹화
              elif name == "valid_files":  
                valid_data_by_label[label_tuple].append(tokens) # label 기준으로 토큰들 그룹화
    
    print("데이터셋: ", adrs[i].split("/")[-1], " train_flow: " , train_sum_of_flow, " test_flow: " ,test_sum_of_flow, "valid_flow: ", valid_sum_of_flow)
    # 토큰화
    data_by_label_lists = {"train_data_by_label": train_data_by_label, "test_data_by_label":test_data_by_label, "valid_data_by_label":valid_data_by_label}
    
    for name, data_by_label in data_by_label_lists.items():
      label_tuples = list(data_by_label.keys())
      for anchor_label, anchor_tokens in tqdm(data_by_label.items(), desc="Creating triplets"):
      
        if len(anchor_tokens) < 2:
          continue
              
        for i, anchor_token in enumerate(anchor_tokens):
          pos_idx = random.randrange(len(anchor_tokens) - 1)
          if pos_idx >= i:
            pos_idx += 1
          positive_token = anchor_tokens[pos_idx]

          # Negative 샘플 선택
          negative_label = random.choice(label_tuples)
          while negative_label == anchor_label:
            negative_label = random.choice(label_tuples)
          
          negative_packets = data_by_label[negative_label]
          negative_token = random.choice(negative_packets)
          
          triplet = {
            'anchor': {'text': anchor_token, 'label': dict(anchor_label)},
            'positive': {'text': positive_token, 'label': dict(anchor_label)},
            'negative': {'text': negative_token, 'label': dict(negative_label)}
          }
          
          # Dataset마다 triplet을 따로 저장
          if anchor_label[0][1] is not None:
            if name == "train_data_by_label":
              triplets_ISXW2016_train.append(json.dumps(triplet, ensure_ascii=False))
            elif name == "valid_data_by_label":
              triplets_ISXW2016_valid.append(json.dumps(triplet, ensure_ascii=False))
            elif name == "test_data_by_label":
              triplets_ISXW2016_test.append(json.dumps(triplet, ensure_ascii=False))           
          elif anchor_label[2][1] is not None:
            if name == "train_data_by_label":
              triplets_DoHBrw2020_train.append(json.dumps(triplet, ensure_ascii=False))
            elif name == "valid_data_by_label":
              triplets_DoHBrw2020_valid.append(json.dumps(triplet, ensure_ascii=False))
            elif name == "test_data_by_label":
              triplets_DoHBrw2020_test.append(json.dumps(triplet, ensure_ascii=False))          
          elif anchor_label[4][1] is not None:
            if name == "train_data_by_label":
              triplets_USTCTFC2016_train.append(json.dumps(triplet, ensure_ascii=False))
            elif name == "valid_data_by_label":
              triplets_USTCTFC2016_valid.append(json.dumps(triplet, ensure_ascii=False))
            elif name == "test_data_by_label":
              triplets_USTCTFC2016_test.append(json.dumps(triplet, ensure_ascii=False))    
    
  


  print("ISXW2016 Train: " , len(triplets_ISXW2016_train), " DoHBrw2020 Train: ", len(triplets_DoHBrw2020_train), " USTCTFC2016 Train: ", len(triplets_USTCTFC2016_train))
  print("ISXW2016 Test: ", len(triplets_ISXW2016_test), " DoHBrw2020 Test: ", len(triplets_DoHBrw2020_test), " USTCTFC2016 Test: ", len(triplets_USTCTFC2016_test))
  print("ISXW2016 Valid: ", len(triplets_ISXW2016_valid), " DoHBrw2020 Valid: ", len(triplets_DoHBrw2020_valid), " USTCTFC2016 Valid: ", len(triplets_USTCTFC2016_valid))

  train_data = triplets_ISXW2016_train + triplets_DoHBrw2020_train + triplets_USTCTFC2016_train
  test_data = triplets_ISXW2016_test + triplets_DoHBrw2020_test + triplets_USTCTFC2016_test
  valid_data = triplets_ISXW2016_valid + triplets_DoHBrw2020_valid + triplets_USTCTFC2016_valid
  
  # 데이터셋 섞어주기
  random.seed(42)
  random.shuffle(train_data)
  random.shuffle(test_data)
  random.shuffle(valid_data)
  
  print("Total Train:", len(train_data))
  print("Total Test:", len(test_data))
  print("Total Valid:", len(valid_data))

  output_dir = os.path.join(os.getcwd(), 'PacRep_0930_3') 
  os.makedirs(output_dir, exist_ok=True)
  
  with open(os.path.join(output_dir, 'train.txt'), 'w', encoding= 'utf-8') as f:
      f.write('\n'.join(train_data))

  with open(os.path.join(output_dir, 'valid.txt'), 'w', encoding= 'utf-8') as f:
      f.write('\n'.join(valid_data))

  with open(os.path.join(output_dir, 'test.txt'), 'w', encoding= 'utf-8') as f:
      f.write('\n'.join(test_data))

if __name__ == '__main__':
    # get_str()
    
    triplet_tokens()
