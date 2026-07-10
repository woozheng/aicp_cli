# AICP_CLI

> AICP([Agent Interaction & Communication Protocol](https://github.com/woozheng/aicp))协议驱动的 LLM 运行时 CLI —— 世界最小需求执行型 Agent，2000行+ 代码，8个文件。176K，跨平台。由[AICP Engine](https://github.com/woozheng/aicp_engine) 创建.

## 能力边界

以需求为导向，从聊天，网络搜索，指令，任务，

操作你的电脑，检查硬件，打开摄像头，文件整理，下载文件，

能力只取决于你的网络限制，你的硬件限制。

## 与传统AI、Claude Hermers等 CLI 的核心差异

不需要安装任何框架、不需要下载任何插件，不需要安装任何工具。

只需要python3.11+，你的api-key，协议自驱动。

AICP_CLI 只有一个目标:解决你的需求。不仅仅输出而是直接执行！

如果你有任何已有的成熟云端agent-api地址和key，只需要告诉他它都会直接运用，不存记忆永不泄露

不需要设定任何身份，能力，灵魂等冗余配置，

无状态，无记忆AICP CLI运行时，一次谈话，直达结果，不会产生Token燃烧！

## 安装方法

```bash
# 1. 克隆
git clone https://github.com/woozheng/aicp_cli.git && cd aicp_cli

# 2. 安装依赖（最少依赖）
pip install -r requirements.txt


# 3、运行
# windows 
aicp  

# linux&Mac
python aicp.py

# 4.第一次运行自动生成配置 aicp.yaml

# edit aicp.yaml,填写 model 和 api-key. 填写完成，再次运行 # 3


```

## 项目用例

你可以让他到做什么
```text

1、帮我打开摄像头拍张照片。
2、帮我分析c:\project项目，生成README.MD
3、帮我生成股票‘xxxxx’ 的研报MD，数据要最新的
4、帮我检查我的电脑进程，检查可疑的病毒进程，输出MD
5、给我生成一个五子棋游戏，并且打开让我玩
6、给我写一个Go的服务，端口在8888，我已经有环境直接编译运行启动
7、直接给我下载docker，并且安装
8、给我搜索豆瓣排名前十的电影，要有详细介绍，生成md文件
9、给我搜索github上最热的AI项目，生成分析文件
10、给我分析目前 AI代码智能体的主流项目、当前现状、变现能力、发展趋势，未来愿景。输出一个PPT
11、我要火车照片，去网上给我下载几张？
12、把 c:\xxx图片，都给我打上水印（去掉水印）
13、给我生成一份标准小升初语文试卷，重点初中水平，不要答案。难度要难！！
14、读取 xxxxxx试卷.md文件，自动生成答案！
15、给我画个大西瓜。然后打开给我看    【没有远程调用API它真会手绘一个】
16、给我生成一段节奏感的音乐文件，10秒钟。然后播放 【可以听听它的审美】
17、生成 中文 ”我爱你“的声音文件，说十遍，播放
。。。。

聊天？？？有意义么？你的需求越清楚，结果越丰富。一切取决于你要做什么。

```

## 模型推荐

代码编程类模型优先，deepseek、豆包 code、千问 coder

通用型、娱乐型、知识型 不推荐

---
## 🤝 贡献

欢迎通过 Issue / PR 提交问题、可以为CLI增加记忆、CLI内置 -r 命令，可与AICP-engine连接，让CLI 远程互相调用


