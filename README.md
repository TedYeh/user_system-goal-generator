# Construction of Message Deliver Service Dialog Systems: Schema Guided Dialogue Corpus Collection and Instruction-Guided Model Training (中文訊息傳遞服務對話系統之建構)
此專案包含兩個部份: Schema-Guided Dialogue語料建構 & Instruction-Guided對話系統建置

## Directory Structure
```shell script
├─goal_generation         #Schema-Guided Dialogue語料建構程式
│  ├─backup               #爬蟲備份資料
│  ├─data                 #存放資料庫及爬蟲程式的資料夾
│  │  ├─db                #資料庫，或服務(service)，給Agent存取使用
│  │  ├─json              #資料庫的json版本
│  │  ├─csv               #資料庫的csv版本
|  |  └─build_db.py       #依據爬取的資料來建立服務
│  ├─matrix               #存放轉移矩陣
|  ├─schema               #存放定義好的綱要
│  └─need_labeled         #對話改寫系統
├─tod_system              #對話系統模組
│  ├─convlab
│  │  ├─base_models       #T5-base的TOD模組
│  │  │  └─t5
│  │  │      ├─dst
│  │  │      ├─nlg
│  │  │      ├─nlu
│  │  │      └─policy
│  │  ├─deploy            #對話系統(website)
│  │  │  ├─ctrl
│  │  │  ├─static
│  │  │  ├─templates
│  │  │  └─utils
│  └─data                 #訓練語料
│      └─unified_datasets #存放經過格式統一的語料
│          ├─messagesgd
|          │  ├─db
|          │  └─preprocess.py #進行資料統一化
│          └─messagewoz
└─transistion matrix     #存放轉移矩陣的圖片
```

## Training TOD System

### Data Preprocessing
當你蒐集完語料，需將語料分割成train/validation/test三個目錄，並壓縮成messagesgd.zip，移至messagesgd目錄
```bash
cd tod_system/data/unified_datasets/messagesgd  #需遵守需unified的語料格式
python preprocess.py
```

### Train Model
```bash
cd tod_system/convlab/base_models/t5/nlu/
bash run_nlu.sh [corpus_name] [context_size]
```

```bash
cd tod_system/convlab/base_models/t5/dst/
bash run_dst.sh [corpus_name] [context_size]
```

```bash
cd tod_system/convlab/base_models/t5/policy/
bash run_policy.sh [corpus_name] [context_size]
```

```bash
cd tod_system/convlab/base_models/t5/nlg/
bash run_nlg.sh [corpus_name] [context_size]
```

### Deploy Model
```bash
cd tod_system/convlab/
python ./deploy/run.py
```
