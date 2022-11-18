# 目录

- [目录](#目录)
- [QMIX描述](#QMIX描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
- [模型描述](#模型描述)
    - [训练性能](#训练性能)
    - [推理流程](#推理流程)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#ModelZoo主页)

# QMIX 描述

QMIX(Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning),深度确定性策略梯度算法。

QMIX是一种Value-Based的多智能体强化学习算法，使用中心式学习分布式执行的方法。算法大框架为基于AC框架的CTDE（Centralized Training Distributed Execution）模式，整个网络由Mixing Network和Agent Network两部分组成。

<font color='cornflowerblue'>论文：</font> QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning. https://arxiv.org/abs/1803.11485:

# 模型架构

整个网络由Mixing Network和Agent Network两部分组成。Agent Network 采用DRQN结构。网络包括三层：输入层（MLP）—> 中间层(GRU) —> 输出层(MLP)。网络循环输入当前的观测值O_t和上一时刻的动作u_(t-1)，输出为每个Agent的行为效用值Q。Mixing Network输入为各个Agent的Q值和当前全局状态S_t，输出为当前状态下所有Agent联合行为u的行为效用值Q_tot。Mixing Network由两个神经网络组成：（1）推理网络：  输入所有Agent的行为效用值Q，推理出全局效用值Q_tot。网络为有两个隐层的前馈神经网络，隐层的权重和偏置由参数生成网络生成。（2）参数生成网络： 输入全局状态St，输出推理网络中的神经元权重（Weight）和偏置（bias）。为保证权重的非负性，采用一个线性层和绝对值激活函数生成权重。第一层偏置采用线性网络生成，第二层偏置由两层网络和ReLU激活函数得到。

# 数据集

Qmix模型在星际争霸2（StarCraft Ⅱ）的环境下进行强化学习训练。一个智能体控制一个单位，智能体的行为空间为移动、攻击、停止和无操作，奖励由总伤害加上击杀10分加上团灭200分组成。通过引入视距来实现局部可观测性，智能体只能得到视距内的信息。

# 环境要求

- 硬件

    - Ascend910

- 框架

    - [MindSpore](https://www.mindspore.cn)

- 如需查看详情，请参见如下资源：

    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

- 第三方库

    - [PyYAML]

        - 安装PyYAML

            ```python
            pip install PyYAML==6.0
            ```

    - [smac](https://github.com/oxwhirl/smac)

        - 安装smac

            ```python
            git clone https://github.com/oxwhirl/smac.git
            cd smac
            pip install -e ".[dev]"
            pre-commit install
            ```

    - [SC2.4.10](http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip)(只支持x86环境下运行,若出现 `CXXABI_1.3.11' not found，`GLIBCXX_3.4.22' not found 这类问题时,请确认3rdparty/  StarCraftII/Libs/libstdc++.so.6版本的正确性)

        - 安装SC2.4.10

            ```shell
            cd qmix
            mkdir 3rdparty
            cd 3rdparty
            unzip -P iagreetotheeula SC2.4.10.zip
            rm -rf SC2.4.10.zip
            ```

    - [SMAC maps](https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip)

        - 安装smac maps

            ```shell
            cd qmix/3rdparty/StarCraftII/Maps
            unzip SMAC_Maps.zip
            rm -rf SMAC_Maps.zip
            ```

# 快速入门

通过官方网站安装MindSpore后，您可按照以下步骤进行训练和推理

```shell
python main.py --config=qmix --env-config=sc2 with env_args.map_name=3m
# 待训练执行完成后，生成checkpoint文件(CKPT_PATH)后可执行推理脚本：
python main.py --config=qmix --env-config=sc2 with env_args.map_name=3m checkpoint_path=CKPT_PATH evaluate=True
```

# 脚本说明

- 脚本及样例代码

```python
qmix
├── README_CN.md                # 说明文档
├── ascend_310_infer
│   ├── inc
│   │   ├── utils.h
│   ├── src
│   │   ├── utils.cc
│   │   ├── main.cc             # Ascend 310 cpp模型文件
│   ├── build.sh                # 编译脚本
│   ├── CMakeLists.txt          # C++过程文件
├── scripts
│   ├── run_standalone_train.sh        # 训练shell脚本
│   ├── run_eval.sh       # qmix Ascend 310推理shell脚本
│   ├── run_infer_310 .sh       # qmix Ascend 310 推理shell脚本
├── src
│   ├── components
│   │   ├── __init__.py
│   │   ├── action_selectors.py # 动作选择
│   │   ├── episode_buffer.py   # 缓存区
│   │   ├── epsilon_schedules.py# schedules
│   │   ├── transforms.py       # onehot
│   ├── config
│   │   ├── algs
│   │   │   ├── qmix.yaml       # qmix配置
│   │   ├── envs
│   │   │   ├── sc2.yaml        # 环境配置
│   │   ├── default.yaml        # 默认配置
│   ├── controllers
│   │   ├── __init__.py
│   │   ├── basic_controller.py # 训练控制器
│   ├── envs
│   │   ├── __init__.py
│   │   ├── multiagentenv.py    # 环境配置
│   ├── learners
│   │   ├── __init__.py
│   │   ├── q_learner.py        # 学习器
│   ├── modules
│   │   ├── agents
│   │   │   ├── __init__.py
│   │   │   ├── infer_agent.py  # 推理网络
│   │   │   ├── rnn_agent.py    # 训练网络
│   │   ├── mixers
│   │   │   ├── __init__.py
│   │   │   ├── qmix.py         # mix网络
│   │   ├── __init__.py
│   │   ├── cells.py            # netWithLossCell
│   │   ├── grad_clip.py        # 梯度裁剪
│   ├── runners
│   │   ├── __init__.py
│   │   ├── episode_runner.py    # 训练器
│   ├── utils
│   │   ├── logging.py           # 打印日志
│   │   ├── timehelper.py        # 计时
│   ├── main.py                  # 训练_推理_export
│   ├── run_310.py               # 310推理
│   ├── run.py                   # 运行脚本
```

- 脚本参数

```python
gamma = 0.99             #  Q衰减系数
batch_size = 32          #  训练样本大小
lr = 0.0005              #  学习率
optim_alpha = 0.99       #  RMSProp alpha
optim_eps = 0.00001      #  RMSProp epsilon
buffer_size = 5000       #  经验池大小
env_args-map_name = '3m' #  环境地图
checkpoint_path = ''     #  ckpt路径
```

- 训练过程

    ```shell
    python:
        python main.py --config=qmix --env-config=sc2 with env_args.map_name='3m'
    shell:
    cd scripts
    bash run_standalone_train.sh [device_id] [MAP_NAME]
    #示例
    bash run_standalone_train.sh 3 '3m'
    ```

- 推理过程

    ckpt_path设置为models那一级的目录，例如/disk0/qmix/results/models/，会自动选取step数最多的ckpt进行推理。

    ```python
    python：
        python main.py --config=qmix --env-config=sc2 with env_args.map_name='3m' evaluate=True checkpoint_path='./results/qmix/results/models/'
    shell:
    cd scripts
    bash run_eval.sh [device_id] [MAP_NAME] [CKPT_PATH]
    #示例
    bash run_eval.sh 3 '3m' './results/qmix/results/models/'
    ```

- 导出过程

    ckpt_path设置为models那一级的目录，例如/disk0/qmix/results/models/，会自动选取step数最多的ckpt进行导出。
    无需指定FILE_FORMAT ["AIR", "MINDIR"]，运行一遍同时生成两个格式的文件。

    ```python
    python main.py --config=qmix --env-config=sc2 with env_args.map_name='3m' export=True checkpoint_path='./results/qmix/results/models/'
    ```

# 模型描述

## 训练性能

| 参数           | qmix                                                         |
| -------------- | ------------------------------------------------------------ |
| 资源           | Ascend 910; CPU 2.60GHz,192cores;Memory,755G                 |
| 上传日期       | 2022.6.1                                                   |
| MindSpore 版本 | 1.6.1                                                        |
| 训练参数       | Gamma:0.99, buffer_size:5000, lr:0.0005, env_args-map_name:'3m',   batch_size:32, checkpoint_path:'',optim_alpha = 0.99 ,optim_eps = 0.00001|
| 优化器         | RMSProp                                                         |
| 损失函数       | 自定义                                                      |
| 输出           | loss                                                       |
| 速度           | 160ms/step                                                   |
| 参数(M)        | 0.13                                                         |
| 脚本           | main.py                                                     |

## 推理流程

如果您需要使用此训练模型在GPU、Ascend 910、Ascend 310等多个硬件平台上进行推理，可参考此[链接](https://www.mindspore.cn/install/en)。下面是Ascend 910操作步骤示例：

- Ascend 910运行推理脚本

    ```python
    # 执行python文件
    python main.py --config=qmix --env-config=sc2 with env_args.map_name='3m' evaluate=True checkpoint_path='./results/qmix/'
    # 执行shell文件
    cd scripts
    bash run_eval.sh [device_id] [MAP_NAME] [CKPT_PATH]
    ```

- GPU、CPU下运行同上

- Ascend 310 运行推理脚本

    ```python
    # Ascend310推理需导出Ascend910下训练结束后的模型文件和模型参数文件
    python main.py --config=qmix --env-config=sc2 with env_args.map_name='3m' export=True checkpoint_path='./results/qmix/'
    ```

    ```shell
    # 先编译C++文件，再执行Python推理
    bash run_infer_310.sh [mindir] [output_path] [device_id]
    # example
    cd scripts
    bash run_infer_310.sh ../test.mindir ../output 0
    ```

# 随机情况说明

在推理过程和训练过程中，我们都使用到gym环境下的随机种子。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
