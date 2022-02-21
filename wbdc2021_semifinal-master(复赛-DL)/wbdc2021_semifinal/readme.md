# 环境依赖

1. src1运行环境：

    原始自带tensorflow_py3环境，需要安装以下包，命令下:

    `
     source activate tensorflow_py3
     pip install deepctr==0.8.7 --no-deps
     pip install torch==1.7.0 torchvision==0.8.1 
     pip install tensorflow-gpu==1.13.1
     pip install numba`
     
     初始化脚本见`init1.sh`
     
2. src3运行环境

    由init.sh 构建的wbdc2021_prosper 环境 
    python 3.6
    包见requirements.txt

    
     
# 目录结构

    如比赛规范

# 运行流程

  - 进入wbdc2021-semi目录下执行sh train.sh开始训练（部分训练代码未来得及整理为py文件）
  - 进入wbdc2021-semi目录下执行sh inference.sh开始推理,脚本里有自动切换环境的代码
  
  `cd wbdc2021-semi/`
  `bash inference.sh /home/tione/notebook/wbdc2021-semi/data/wedata/wechat_algo_data2/test_a.csv`
  
# 模型及特征
  - 模型deepfm 
  - 特征
    - feedembdding降维16
    - feedembdding按时间加权降维16
    - user 观看视频序列做w2v 16维
    - user 观看视频anthorid序列做w2v 16维
    - tag、keyword、文本特征
    - user-feed二部图  随机游走 然后w2v
    
  - share bottom 多任务 + 7个单任务deepfm
  - 其他尝试模型 fefm + mmoe , mmoe + multihead ple 
  - 
  -
# 算法性能
  - inference.sh 2000样本平均预测时长22ms ，总时间330s, 其中share bottom 1ms 每个任务，融合与写csv比较耗时
  - 使用batch里merge特征的方式，因此预测性能和cpu有关
  
# 代码说明

  - src/src1 inference.py 644行 submission1 = infer(test_loader,share_bottom_model,os.path.join(MODEL_PATH,'tf_models/share_bottom/model_seed%s' % (SEED)))  #(share bottom )

  - src/inference.py  271行     pred_ans = model.predict(test_loader)  #(deepfm)